"""PostgreSQL implementation of the durable outbox contract.

Mirrors :class:`agentic_cv.outbox.Outbox` so producers, the dispatcher, and the
export workflow can switch backends without changes. SQLite remains the default;
this store is only constructed when ``AGENTIC_CV_POSTGRES_DSN`` is set.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from .contracts import EventEnvelope
from .outbox import OutboxRecord
from .postgres import PostgresConfig, apply_migrations, connect


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _jsonb(value: dict[str, Any]) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(value)


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def open_postgres_outbox(config: PostgresConfig) -> "PostgresOutbox":
    """Connect and return an outbox scoped to the configured schema."""
    from psycopg.rows import dict_row

    connection = connect(config, row_factory=dict_row)
    try:
        return PostgresOutbox(connection, schema=config.schema)
    except Exception:
        connection.close()
        raise


class PostgresOutbox:
    def __init__(self, connection: Any, *, schema: str = "public") -> None:
        self._connection = connection
        self._lock = threading.RLock()
        apply_migrations(connection, schema=schema)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def enqueue(self, event: EventEnvelope) -> OutboxRecord:
        now = _utc_now()
        payload = dict(event.to_dict()["payload"])
        with self._lock:
            existing = self._fetch_by_event_id(event.event_id)
            if existing is None and event.idempotency_key is not None:
                existing = self._fetch_by_idempotency_key(
                    event.tenant_id, event.idempotency_key
                )
            if existing is not None:
                return existing
            row = self._connection.execute(
                """
                INSERT INTO outbox_events (
                    event_id, event_type, tenant_id, aggregate_type, aggregate_id,
                    payload, idempotency_key, attempts, last_error, occurred_at,
                    published_at, next_attempt_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 0, NULL, %s, NULL, %s)
                RETURNING *
                """,
                (
                    event.event_id, event.event_type, event.tenant_id,
                    event.aggregate_type, event.aggregate_id, _jsonb(payload),
                    event.idempotency_key, event.occurred_at, now,
                ),
            ).fetchone()
            self._connection.commit()
            return self._record(row)

    def get(self, event_id: UUID) -> OutboxRecord:
        with self._lock:
            record = self._fetch_by_event_id(event_id)
        if record is None:
            raise KeyError(f"unknown outbox event: {event_id}")
        return record

    def pending(self, *, limit: int = 100) -> list[OutboxRecord]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM outbox_events
                WHERE published_at IS NULL AND next_attempt_at <= %s
                ORDER BY occurred_at, sequence LIMIT %s
                """,
                (_utc_now(), limit),
            ).fetchall()
        return [self._record(row) for row in rows]

    def mark_published(self, event_id: UUID) -> OutboxRecord:
        with self._lock:
            self.get(event_id)
            row = self._connection.execute(
                """
                UPDATE outbox_events
                SET published_at = %s, last_error = NULL
                WHERE event_id = %s
                RETURNING *
                """,
                (_utc_now(), event_id),
            ).fetchone()
            self._connection.commit()
            return self._record(row)

    def mark_failed(
        self,
        event_id: UUID,
        error: str,
        *,
        retry_after_seconds: int = 0,
    ) -> OutboxRecord:
        if retry_after_seconds < 0:
            raise ValueError("retry_after_seconds must not be negative")
        with self._lock:
            current = self.get(event_id)
            next_attempt = _utc_now() + timedelta(seconds=retry_after_seconds)
            row = self._connection.execute(
                """
                UPDATE outbox_events
                SET attempts = %s, last_error = %s, next_attempt_at = %s
                WHERE event_id = %s
                RETURNING *
                """,
                (current.attempts + 1, error, next_attempt, event_id),
            ).fetchone()
            self._connection.commit()
            return self._record(row)

    def _fetch_by_event_id(self, event_id: UUID) -> OutboxRecord | None:
        row = self._connection.execute(
            "SELECT * FROM outbox_events WHERE event_id = %s", (event_id,)
        ).fetchone()
        return None if row is None else self._record(row)

    def _fetch_by_idempotency_key(
        self, tenant_id: UUID, idempotency_key: str
    ) -> OutboxRecord | None:
        row = self._connection.execute(
            """
            SELECT * FROM outbox_events
            WHERE tenant_id = %s AND idempotency_key = %s
            """,
            (tenant_id, idempotency_key),
        ).fetchone()
        return None if row is None else self._record(row)

    @staticmethod
    def _record(row: Any) -> OutboxRecord:
        payload = row["payload"]
        if not isinstance(payload, dict):
            payload = json.loads(payload)
        published_at = row["published_at"]
        return OutboxRecord(
            event_id=UUID(str(row["event_id"])),
            sequence=row["sequence"],
            event_type=row["event_type"],
            tenant_id=UUID(str(row["tenant_id"])),
            aggregate_type=row["aggregate_type"],
            aggregate_id=UUID(str(row["aggregate_id"])),
            payload=payload,
            idempotency_key=row["idempotency_key"],
            attempts=row["attempts"],
            last_error=row["last_error"],
            occurred_at=_iso(row["occurred_at"]),
            published_at=None if published_at is None else _iso(published_at),
            next_attempt_at=_iso(row["next_attempt_at"]),
        )
