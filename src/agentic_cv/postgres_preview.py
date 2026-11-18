"""PostgreSQL implementation of the durable preview event store.

Mirrors :class:`agentic_cv.preview.PreviewStore` for SSE/replay clients.
SQLite remains the default; this store is opt-in via
``AGENTIC_CV_POSTGRES_DSN``.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from .postgres import PostgresConfig, apply_migrations, connect
from .preview import PreviewEvent


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def open_postgres_preview_store(config: PostgresConfig) -> "PostgresPreviewStore":
    from psycopg.rows import dict_row

    connection = connect(config, row_factory=dict_row)
    try:
        return PostgresPreviewStore(connection, schema=config.schema)
    except Exception:
        connection.close()
        raise


class PostgresPreviewStore:
    def __init__(self, connection: Any, *, schema: str = "public") -> None:
        self._connection = connection
        self._lock = threading.RLock()
        apply_migrations(connection, schema=schema)

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
            row = self._connection.execute(
                """
                INSERT INTO preview_events
                    (tenant_id, job_id, image_id, processed, total, status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (tenant_id, job_id, image_id, processed, total, status, _utc_now()),
            ).fetchone()
            self._connection.commit()
            return self._record(row)

    def get(self, event_id: int) -> PreviewEvent:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM preview_events WHERE event_id = %s", (event_id,)
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
                WHERE tenant_id = %s AND job_id = %s AND event_id > %s
                ORDER BY event_id
                """,
                (tenant_id, job_id, after_event_id),
            ).fetchall()
        return [self._record(row) for row in rows]

    @staticmethod
    def _record(row: Any) -> PreviewEvent:
        return PreviewEvent(
            event_id=int(row["event_id"]),
            tenant_id=UUID(str(row["tenant_id"])),
            job_id=UUID(str(row["job_id"])),
            image_id=row["image_id"],
            processed=int(row["processed"]),
            total=int(row["total"]),
            status=row["status"],
            created_at=_iso(row["created_at"]),
        )
