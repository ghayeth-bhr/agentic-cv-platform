"""Durable local job state for restart-safe control-plane operations."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    "queued": frozenset({"running", "cancelled"}),
    "running": frozenset({"succeeded", "failed", "cancelled", "queued"}),
    "succeeded": frozenset(),
    "failed": frozenset({"queued"}),
    "cancelled": frozenset(),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, kw_only=True)
class JobRecord:
    job_id: UUID
    tenant_id: UUID
    kind: str
    status: JobStatus
    payload: dict[str, Any]
    idempotency_key: str
    attempts: int
    error: str | None
    created_at: str
    updated_at: str


class JobStore:
    def __init__(self, path: str) -> None:
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (tenant_id, idempotency_key)
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create(
        self,
        *,
        tenant_id: UUID,
        kind: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> JobRecord:
        if not kind.strip() or not idempotency_key.strip():
            raise ValueError("kind and idempotency_key must not be empty")
        encoded_payload = json.dumps(payload, sort_keys=True)
        with self._lock:
            existing = self._connection.execute(
                "SELECT * FROM jobs WHERE tenant_id = ? AND idempotency_key = ?",
                (str(tenant_id), idempotency_key),
            ).fetchone()
            if existing is not None:
                if existing["kind"] != kind or existing["payload"] != encoded_payload:
                    raise ValueError("idempotency key already identifies a different job")
                return self._record(existing)
            job_id = uuid4()
            now = _utc_now()
            self._connection.execute(
                """
                INSERT INTO jobs (
                    job_id, tenant_id, kind, status, payload, idempotency_key,
                    attempts, error, created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, 0, NULL, ?, ?)
                """,
                (str(job_id), str(tenant_id), kind, encoded_payload, idempotency_key, now, now),
            )
            self._connection.commit()
            return self.get(job_id)

    def get(self, job_id: UUID) -> JobRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown job: {job_id}")
        return self._record(row)

    def transition(
        self,
        job_id: UUID,
        status: JobStatus,
        *,
        error: str | None = None,
    ) -> JobRecord:
        with self._lock:
            current = self.get(job_id)
            if status not in ALLOWED_TRANSITIONS[current.status]:
                raise ValueError(f"cannot transition job from {current.status} to {status}")
            attempts = current.attempts + (status == "running")
            now = _utc_now()
            self._connection.execute(
                "UPDATE jobs SET status = ?, attempts = ?, error = ?, updated_at = ? WHERE job_id = ?",
                (status, attempts, error, now, str(job_id)),
            )
            self._connection.commit()
            return self.get(job_id)

    @staticmethod
    def _record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=UUID(row["job_id"]), tenant_id=UUID(row["tenant_id"]),
            kind=row["kind"], status=row["status"], payload=json.loads(row["payload"]),
            idempotency_key=row["idempotency_key"], attempts=row["attempts"],
            error=row["error"], created_at=row["created_at"], updated_at=row["updated_at"],
        )
