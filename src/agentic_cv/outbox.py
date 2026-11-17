"""Durable local outbox for idempotent integration-event delivery."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from .contracts import EventEnvelope


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, kw_only=True)
class OutboxRecord:
    event_id: UUID
    event_type: str
    tenant_id: UUID
    aggregate_type: str
    aggregate_id: UUID
    payload: dict[str, Any]
    idempotency_key: str | None
    attempts: int
    last_error: str | None
    occurred_at: str
    published_at: str | None
    next_attempt_at: str


class Outbox:
    def __init__(self, path: str) -> None:
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS outbox_events (
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
        self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def enqueue(self, event: EventEnvelope) -> OutboxRecord:
        now = _utc_now().isoformat()
        value = event.to_dict()
        with self._lock:
            existing = self._connection.execute(
                "SELECT * FROM outbox_events WHERE event_id = ?",
                (str(event.event_id),),
            ).fetchone()
            if existing is None and event.idempotency_key is not None:
                existing = self._connection.execute(
                    "SELECT * FROM outbox_events WHERE tenant_id = ? AND idempotency_key = ?",
                    (str(event.tenant_id), event.idempotency_key),
                ).fetchone()
            if existing is not None:
                return self._record(existing)
            self._connection.execute(
                """
                INSERT INTO outbox_events (
                    event_id, event_type, tenant_id, aggregate_type, aggregate_id,
                    payload, idempotency_key, attempts, last_error, occurred_at,
                    published_at, next_attempt_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, NULL, ?)
                """,
                (
                    str(event.event_id), event.event_type, str(event.tenant_id),
                    event.aggregate_type, str(event.aggregate_id), json.dumps(value["payload"], sort_keys=True),
                    event.idempotency_key, event.occurred_at.isoformat(), now,
                ),
            )
            self._connection.commit()
            return self.get(event.event_id)

    def get(self, event_id: UUID) -> OutboxRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM outbox_events WHERE event_id = ?", (str(event_id),)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown outbox event: {event_id}")
        return self._record(row)

    def pending(self, *, limit: int = 100) -> list[OutboxRecord]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM outbox_events
                WHERE published_at IS NULL AND next_attempt_at <= ?
                ORDER BY occurred_at, event_id LIMIT ?
                """,
                (_utc_now().isoformat(), limit),
            ).fetchall()
        return [self._record(row) for row in rows]

    def mark_published(self, event_id: UUID) -> OutboxRecord:
        with self._lock:
            self.get(event_id)
            self._connection.execute(
                "UPDATE outbox_events SET published_at = ?, last_error = NULL WHERE event_id = ?",
                (_utc_now().isoformat(), str(event_id)),
            )
            self._connection.commit()
            return self.get(event_id)

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
            self._connection.execute(
                """
                UPDATE outbox_events
                SET attempts = ?, last_error = ?, next_attempt_at = ?
                WHERE event_id = ?
                """,
                (current.attempts + 1, error, next_attempt.isoformat(), str(event_id)),
            )
            self._connection.commit()
            return self.get(event_id)

    @staticmethod
    def _record(row: sqlite3.Row) -> OutboxRecord:
        return OutboxRecord(
            event_id=UUID(row["event_id"]), event_type=row["event_type"],
            tenant_id=UUID(row["tenant_id"]), aggregate_type=row["aggregate_type"],
            aggregate_id=UUID(row["aggregate_id"]), payload=json.loads(row["payload"]),
            idempotency_key=row["idempotency_key"], attempts=row["attempts"],
            last_error=row["last_error"], occurred_at=row["occurred_at"],
            published_at=row["published_at"], next_attempt_at=row["next_attempt_at"],
        )
