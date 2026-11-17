from uuid import uuid4

import pytest

from agentic_cv.contracts import EventEnvelope
from agentic_cv.outbox import Outbox


def event(*, key: str | None = None) -> EventEnvelope:
    return EventEnvelope(
        event_type="AssetReady",
        tenant_id=uuid4(),
        aggregate_type="asset",
        aggregate_id=uuid4(),
        payload={"asset": "asset-1"},
        idempotency_key=key,
    )


def test_enqueue_is_idempotent_and_pending_until_published(tmp_path):
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    original = event(key="asset-ready:1")
    first = outbox.enqueue(original)
    duplicate = outbox.enqueue(original)

    assert duplicate == first
    assert outbox.pending()[0] == first
    published = outbox.mark_published(first.event_id)
    assert published.published_at is not None
    assert outbox.pending() == []
    outbox.close()


def test_failed_delivery_is_retryable_and_counts_attempts(tmp_path):
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    record = outbox.enqueue(event())

    failed = outbox.mark_failed(record.event_id, "broker unavailable")

    assert failed.attempts == 1
    assert failed.last_error == "broker unavailable"
    assert outbox.pending()[0].event_id == record.event_id
    outbox.close()


def test_same_tenant_idempotency_key_returns_original_event(tmp_path):
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    first = event(key="same-key")
    second = EventEnvelope(
        event_type="AssetReady",
        tenant_id=first.tenant_id,
        aggregate_type="asset",
        aggregate_id=uuid4(),
        payload={"asset": "different"},
        idempotency_key="same-key",
    )

    assert outbox.enqueue(second) == outbox.enqueue(first)
    outbox.close()


def test_pending_limit_must_be_positive(tmp_path):
    outbox = Outbox(str(tmp_path / "events.sqlite3"))

    with pytest.raises(ValueError, match="limit"):
        outbox.pending(limit=0)
    outbox.close()
