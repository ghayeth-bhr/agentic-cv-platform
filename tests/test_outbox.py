import sqlite3
from datetime import datetime, timezone
from uuid import UUID, uuid4

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


def test_pending_uses_insertion_order_when_timestamps_tie(tmp_path):
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    occurred_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # UUIDs descend so ordering by event_id would reverse insertion order.
    events = [
        EventEnvelope(
            event_type="AssetReady",
            tenant_id=uuid4(),
            aggregate_type="asset",
            aggregate_id=uuid4(),
            payload={"asset": f"asset-{index}"},
            event_id=UUID(int=3 - index),
            occurred_at=occurred_at,
        )
        for index in range(3)
    ]

    for item in events:
        outbox.enqueue(item)

    assert [record.event_id for record in outbox.pending()] == [
        item.event_id for item in events
    ]
    assert [record.sequence for record in outbox.pending()] == [1, 2, 3]
    outbox.close()


def test_legacy_outbox_without_sequence_column_is_migrated(tmp_path):
    path = str(tmp_path / "events.sqlite3")
    legacy = sqlite3.connect(path)
    legacy.execute(
        """
        CREATE TABLE outbox_events (
            event_id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            aggregate_type TEXT NOT NULL,
            aggregate_id TEXT NOT NULL,
            payload TEXT NOT NULL,
            idempotency_key TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            occurred_at TEXT NOT NULL,
            published_at TEXT,
            next_attempt_at TEXT NOT NULL,
            UNIQUE (tenant_id, idempotency_key)
        )
        """
    )
    occurred_at = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
    legacy_event_id = str(UUID(int=9))
    legacy.execute(
        """
        INSERT INTO outbox_events (
            event_id, event_type, tenant_id, aggregate_type, aggregate_id, payload,
            idempotency_key, attempts, last_error, occurred_at, published_at,
            next_attempt_at
        ) VALUES (?, 'AssetReady', ?, 'asset', ?, '{}', NULL, 0, NULL, ?, NULL, ?)
        """,
        (legacy_event_id, str(uuid4()), str(uuid4()), occurred_at, occurred_at),
    )
    legacy.commit()
    legacy.close()

    outbox = Outbox(path)
    new_event = EventEnvelope(
        event_type="AssetReady",
        tenant_id=uuid4(),
        aggregate_type="asset",
        aggregate_id=uuid4(),
        payload={"asset": "new"},
        event_id=UUID(int=1),
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    new_record = outbox.enqueue(new_event)

    assert new_record.sequence == 2
    assert [record.event_id for record in outbox.pending()] == [
        UUID(legacy_event_id),
        new_event.event_id,
    ]
    outbox.close()
