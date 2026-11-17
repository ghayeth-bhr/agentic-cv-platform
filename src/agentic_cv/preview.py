"""Durable tenant-scoped progress events for batch preview clients."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, kw_only=True)
class PreviewEvent:
    event_id: int
    tenant_id: UUID
    job_id: UUID
    image_id: str
    processed: int
    total: int
    status: str
    created_at: str


class PreviewStore:
    def __init__(self, path: str) -> None:
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS preview_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                image_id TEXT NOT NULL,
                processed INTEGER NOT NULL,
                total INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def publish(
        self,
        *,
        tenant_id: UUID,
        job_id: UUID,
        image_id: str,
        processed: int,
        total: int,
        status: str = "preview_ready",
    ) -> PreviewEvent:
        if not image_id.strip() or processed < 1 or total < processed:
            raise ValueError("preview image and progress values are invalid")
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO preview_events
                    (tenant_id, job_id, image_id, processed, total, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (str(tenant_id), str(job_id), image_id, processed, total, status, _utc_now()),
            )
            self._connection.commit()
            return self.get(cursor.lastrowid)

    def get(self, event_id: int) -> PreviewEvent:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM preview_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown preview event: {event_id}")
        return self._record(row)

    def list_for_job(
        self,
        *,
        tenant_id: UUID,
        job_id: UUID,
        after_event_id: int = 0,
    ) -> list[PreviewEvent]:
        if after_event_id < 0:
            raise ValueError("after_event_id must not be negative")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM preview_events
                WHERE tenant_id = ? AND job_id = ? AND event_id > ?
                ORDER BY event_id
                """,
                (str(tenant_id), str(job_id), after_event_id),
            ).fetchall()
        return [self._record(row) for row in rows]

    @staticmethod
    def _record(row: sqlite3.Row) -> PreviewEvent:
        return PreviewEvent(
            event_id=row["event_id"], tenant_id=UUID(row["tenant_id"]),
            job_id=UUID(row["job_id"]), image_id=row["image_id"],
            processed=row["processed"], total=row["total"], status=row["status"],
            created_at=row["created_at"],
        )
