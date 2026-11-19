"""Durable local job state for restart-safe control-plane operations."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4
from .sqlite_utils import connect_sqlite

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
    lease_owner: str | None
    lease_expires_at: str | None
    cancel_requested: bool


class JobStoreBackend(Protocol):
    def close(self) -> None: ...

    def create(
        self, *, tenant_id: UUID, kind: str, payload: dict[str, Any], idempotency_key: str
    ) -> JobRecord: ...

    def get(self, job_id: UUID) -> JobRecord: ...

    def list(
        self, *, status: JobStatus | None = None, kind: str | None = None
    ) -> list[JobRecord]: ...

    def transition(
        self, job_id: UUID, status: JobStatus, *, error: str | None = None
    ) -> JobRecord: ...

    def claim(self, job_id: UUID, *, worker_id: str, lease_seconds: float = 60) -> JobRecord: ...

    def heartbeat(
        self, job_id: UUID, *, worker_id: str, lease_seconds: float = 60
    ) -> JobRecord: ...

    def requeue_expired_leases(
        self, *, now: datetime | None = None, kind: str | None = None
    ) -> tuple[UUID, ...]: ...

    def request_cancel(self, job_id: UUID) -> JobRecord: ...


class JobStore:
    def __init__(self, path: str) -> None:
        self._connection = connect_sqlite(path)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL,
                tenant_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                lease_owner TEXT,
                lease_expires_at TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (tenant_id, idempotency_key)
            )
            """
        )
        columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(jobs)")}
        for name, definition in (
            ("lease_owner", "TEXT"),
            ("lease_expires_at", "TEXT"),
            ("cancel_requested", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if name not in columns:
                self._connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")
        if "sequence" not in columns:
            self._connection.execute("ALTER TABLE jobs ADD COLUMN sequence INTEGER")
            self._connection.execute("UPDATE jobs SET sequence = rowid")
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
            sequence = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM jobs"
            ).fetchone()[0]
            self._connection.execute(
                """
                INSERT INTO jobs (
                    job_id, sequence, tenant_id, kind, status, payload, idempotency_key,
                    attempts, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'queued', ?, ?, 0, NULL, ?, ?)
                """,
                (
                    str(job_id), sequence, str(tenant_id), kind, encoded_payload,
                    idempotency_key, now, now,
                ),
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

    def list(self, *, status: JobStatus | None = None, kind: str | None = None) -> list[JobRecord]:
        clauses: list[str] = []
        values: list[str] = []
        if status is not None:
            clauses.append("status = ?")
            values.append(status)
        if kind is not None:
            clauses.append("kind = ?")
            values.append(kind)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM jobs{where} ORDER BY created_at, sequence", values
            ).fetchall()
        return [self._record(row) for row in rows]

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

    def claim(self, job_id: UUID, *, worker_id: str, lease_seconds: float = 60) -> JobRecord:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0")
        with self._lock:
            current = self.get(job_id)
            now = datetime.now(timezone.utc)
            if current.status not in ("queued", "running"):
                raise ValueError(f"cannot claim job in {current.status} state")
            if current.status == "running" and current.lease_expires_at:
                expires_at = datetime.fromisoformat(current.lease_expires_at)
                if expires_at > now and current.lease_owner != worker_id:
                    raise RuntimeError("job is leased by another worker")
            expires = (now + timedelta(seconds=lease_seconds)).isoformat()
            attempts = current.attempts + (current.status == "queued")
            self._connection.execute(
                """
                UPDATE jobs SET status = 'running', attempts = ?, lease_owner = ?,
                    lease_expires_at = ?, updated_at = ? WHERE job_id = ?
                """,
                (attempts, worker_id, expires, now.isoformat(), str(job_id)),
            )
            self._connection.commit()
            return self.get(job_id)

    def heartbeat(self, job_id: UUID, *, worker_id: str, lease_seconds: float = 60) -> JobRecord:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0")
        with self._lock:
            current = self.get(job_id)
            self._require_active_lease(current, worker_id)
            now = datetime.now(timezone.utc)
            expires = (now + timedelta(seconds=lease_seconds)).isoformat()
            self._connection.execute(
                "UPDATE jobs SET lease_expires_at = ?, updated_at = ? WHERE job_id = ?",
                (expires, now.isoformat(), str(job_id)),
            )
            self._connection.commit()
            return self.get(job_id)

    def requeue_expired_leases(
        self, *, now: datetime | None = None, kind: str | None = None
    ) -> tuple[UUID, ...]:
        current_time = now or datetime.now(timezone.utc)
        current_text = current_time.isoformat()
        kind_clause = " AND kind = ?" if kind is not None else ""
        values: tuple[str, ...] = (current_text,) if kind is None else (current_text, kind)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT job_id, cancel_requested FROM jobs
                WHERE status = 'running' AND lease_expires_at IS NOT NULL
                    AND lease_expires_at <= ?{kind_clause}
                ORDER BY created_at, sequence
                """,
                values,
            ).fetchall()
            for row in rows:
                status = "cancelled" if row["cancel_requested"] else "queued"
                error = "job cancellation requested while lease expired" if row["cancel_requested"] else None
                self._connection.execute(
                    """
                    UPDATE jobs SET status = ?, error = ?, lease_owner = NULL,
                        lease_expires_at = NULL, updated_at = ? WHERE job_id = ?
                    """,
                    (status, error, current_text, row["job_id"]),
                )
            self._connection.commit()
        return tuple(UUID(row["job_id"]) for row in rows)

    def request_cancel(self, job_id: UUID) -> JobRecord:
        with self._lock:
            current = self.get(job_id)
            if current.status == "queued":
                now = _utc_now()
                self._connection.execute(
                    "UPDATE jobs SET status = 'cancelled', cancel_requested = 1, updated_at = ? WHERE job_id = ?",
                    (now, str(job_id)),
                )
            elif current.status == "running":
                self._connection.execute(
                    "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE job_id = ?",
                    (_utc_now(), str(job_id)),
                )
            self._connection.commit()
            return self.get(job_id)

    @staticmethod
    def _require_active_lease(current: JobRecord, worker_id: str) -> None:
        if current.status != "running" or current.lease_owner != worker_id:
            raise RuntimeError("worker does not own an active job lease")
        if current.lease_expires_at is None or datetime.fromisoformat(current.lease_expires_at) <= datetime.now(timezone.utc):
            raise RuntimeError("job lease has expired")

    @staticmethod
    def _record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=UUID(row["job_id"]), tenant_id=UUID(row["tenant_id"]),
            kind=row["kind"], status=row["status"], payload=json.loads(row["payload"]),
            idempotency_key=row["idempotency_key"], attempts=row["attempts"],
            error=row["error"], created_at=row["created_at"], updated_at=row["updated_at"],
            lease_owner=row["lease_owner"], lease_expires_at=row["lease_expires_at"],
            cancel_requested=bool(row["cancel_requested"]),
        )
