"""Contract tests shared by the SQLite and PostgreSQL outboxes."""

import os
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from agentic_cv.contracts import EventEnvelope
from agentic_cv.outbox import Outbox


def _event(*, key: str | None = None, event_id: UUID | None = None, occurred_at=None):
    return EventEnvelope(
        event_type="AssetReady",
        tenant_id=uuid4(),
        aggregate_type="asset",
        aggregate_id=uuid4(),
        payload={"asset": "asset-1"},
        event_id=event_id or uuid4(),
        idempotency_key=key,
        occurred_at=occurred_at or datetime.now(timezone.utc),
    )


def exercise_outbox_contract(outbox):
    event = _event()
    first = outbox.enqueue(event)

    assert first.event_id == event.event_id
    assert first.sequence == 1
    assert outbox.enqueue(event) == first
    assert [record.event_id for record in outbox.pending()] == [event.event_id]

    published = outbox.mark_published(event.event_id)
    assert published.published_at is not None
    assert outbox.pending() == []

    with pytest.raises(KeyError, match="unknown outbox event"):
        outbox.get(uuid4())
    with pytest.raises(ValueError, match="limit"):
        outbox.pending(limit=0)

    keyed = _event(key="asset-ready:1")
    original = outbox.enqueue(keyed)
    duplicate = EventEnvelope(
        event_type="AssetReady",
        tenant_id=keyed.tenant_id,
        aggregate_type="asset",
        aggregate_id=uuid4(),
        payload={"asset": "different"},
        idempotency_key="asset-ready:1",
    )
    assert outbox.enqueue(duplicate) == original

    failing = outbox.enqueue(_event())
    failed = outbox.mark_failed(failing.event_id, "broker unavailable")
    assert failed.attempts == 1
    assert failed.last_error == "broker unavailable"
    assert failed.published_at is None

    delayed = outbox.enqueue(_event())
    outbox.mark_failed(delayed.event_id, "retry later", retry_after_seconds=60)
    pending_ids = {record.event_id for record in outbox.pending()}
    assert failing.event_id in pending_ids
    assert delayed.event_id not in pending_ids


def test_sqlite_outbox_contract(tmp_path):
    outbox = Outbox(str(tmp_path / "outbox.sqlite3"))
    try:
        exercise_outbox_contract(outbox)
    finally:
        outbox.close()


def test_outbox_pending_preserves_insertion_order_when_timestamps_tie_sqlite(tmp_path):
    outbox = Outbox(str(tmp_path / "outbox.sqlite3"))
    occurred_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events = [
        _event(event_id=UUID(int=3 - index), occurred_at=occurred_at)
        for index in range(3)
    ]
    try:
        for event in events:
            outbox.enqueue(event)
        assert [record.event_id for record in outbox.pending()] == [
            event.event_id for event in events
        ]
    finally:
        outbox.close()


@pytest.mark.postgres
def test_postgres_outbox_contract():
    dsn = os.environ.get("AGENTIC_CV_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set AGENTIC_CV_TEST_POSTGRES_DSN to run against a live PostgreSQL")
    from agentic_cv.postgres import PostgresConfig, connect
    from agentic_cv.postgres_outbox import open_postgres_outbox

    schema = f"agentic_cv_test_{uuid4().hex[:8]}"
    outbox = None
    try:
        outbox = open_postgres_outbox(PostgresConfig(dsn=dsn, schema=schema))
        exercise_outbox_contract(outbox)

        occurred_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            _event(event_id=UUID(int=3 - index), occurred_at=occurred_at)
            for index in range(3)
        ]
        for event in events:
            outbox.enqueue(event)
        ordered = {
            record.event_id
            for record in outbox.pending(limit=100)
            if record.occurred_at.startswith("2026-01-01")
        }
        assert ordered == {event.event_id for event in events}
        assert [
            record.event_id
            for record in outbox.pending(limit=100)
            if record.occurred_at.startswith("2026-01-01")
        ] == [event.event_id for event in events]
    finally:
        if outbox is not None:
            outbox.close()
        cleanup = connect(PostgresConfig(dsn=dsn))
        cleanup.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()
