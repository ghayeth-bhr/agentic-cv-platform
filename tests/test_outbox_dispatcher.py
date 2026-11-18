from uuid import uuid4

import pytest

from agentic_cv.contracts import EventEnvelope
from agentic_cv.outbox import Outbox
from agentic_cv.outbox_dispatcher import OutboxDispatcher


class FakePublisher:
    def __init__(self, *, fail_first=False):
        self.events = []
        self.fail_first = fail_first

    def publish(self, event):
        self.events.append(event.event_id)
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError("broker unavailable")


def make_event():
    return EventEnvelope(
        event_type="DatasetExportCompleted", tenant_id=uuid4(), aggregate_type="job",
        aggregate_id=uuid4(), payload={"format": "coco"}, idempotency_key=None,
    )


def test_dispatcher_publishes_and_marks_events(tmp_path):
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    event = outbox.enqueue(make_event())
    publisher = FakePublisher()

    processed = OutboxDispatcher(outbox=outbox, publisher=publisher).dispatch_once()

    assert processed == (event.event_id,)
    assert publisher.events == [event.event_id]
    assert outbox.pending() == []
    outbox.close()


def test_dispatcher_records_failure_and_continues(tmp_path):
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    first = outbox.enqueue(make_event())
    second = outbox.enqueue(make_event())
    publisher = FakePublisher(fail_first=True)

    processed = OutboxDispatcher(outbox=outbox, publisher=publisher).dispatch_once(retry_after_seconds=0)

    assert processed == (first.event_id, second.event_id)
    assert publisher.events == [first.event_id, second.event_id]
    assert outbox.get(first.event_id).attempts == 1
    assert outbox.get(first.event_id).published_at is None
    assert outbox.get(second.event_id).published_at is not None
    outbox.close()


def test_dispatcher_validates_retry_delay(tmp_path):
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    with pytest.raises(ValueError, match="retry_after_seconds"):
        OutboxDispatcher(outbox=outbox, publisher=FakePublisher()).dispatch_once(retry_after_seconds=-1)
    outbox.close()