"""PostgreSQL implementation of the durable job store contract.

Mirrors :class:`agentic_cv.job_store.JobStore` so workers, workflows, and the
export API can switch backends without changes. SQLite remains the default;
this store is only constructed when ``AGENTIC_CV_POSTGRES_DSN`` is set.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from .job_store import ALLOWED_TRANSITIONS, JobRecord, JobStatus
from .postgres import PostgresConfig, apply_migrations, connect


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _jsonb(value: dict[str, Any]) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(value)


def open_postgres_job_store(config: PostgresConfig) -> "PostgresJobStore":
    """Connect and return a job store scoped to the configured schema."""
    from psycopg.rows import dict_row

    connection = connect(config, row_factory=dict_row)
    try:
        return PostgresJobStore(connection, schema=config.schema)
    except Exception:
        connection.close()
        raise


class PostgresJobStore:
    def __init__(self, connection: Any, *, schema: str = "public") -> None:
        self._connection = connection
        self._lock = threading.RLock()
        apply_migrations(connection, schema=schema)

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
        with self._lock:
            existing = self._fetch_by_idempotency_key(tenant_id, idempotency_key)
            if existing is not None:
                if existing.kind != kind or existing.payload != payload:
                    raise ValueError("idempotency key already identifies a different job")
                return existing
            job_id = uuid4()
            now = _utc_now()
            row = self._connection.execute(
                """
                INSERT INTO jobs (
                    job_id, tenant_id, kind, status, payload, idempotency_key,
                    attempts, error, lease_owner, lease_expires_at, cancel_requested,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, 'queued', %s, %s, 0, NULL, NULL, NULL, FALSE, %s, %s)
                RETURNING *
                """,
                (job_id, tenant_id, kind, _jsonb(payload), idempotency_key, now, now),
            ).fetchone()
            self._connection.commit()
            return self._record(row)

    def get(self, job_id: UUID) -> JobRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE job_id = %s", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown job: {job_id}")
        return self._record(row)

    def list(
        self, *, status: JobStatus | None = None, kind: str | None = None
    ) -> list[JobRecord]:
        clauses: list[str] = []
        values: list[Any] = []
        if status is not None:
            clauses.append("status = %s")
            values.append(status)
        if kind is not None:
            clauses.append("kind = %s")
            values.append(kind)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM jobs{where} ORDER BY created_at, sequence", values
            ).fetchall()
        return [self._record(row) for row in rows]

    def transition(
        self, job_id: UUID, status: JobStatus, *, error: str | None = None
    ) -> JobRecord:
        with self._lock:
            current = self.get(job_id)
            if status not in ALLOWED_TRANSITIONS[current.status]:
                raise ValueError(f"cannot transition job from {current.status} to {status}")
            attempts = current.attempts + (status == "running")
            row = self._connection.execute(
                """
                UPDATE jobs
                SET status = %s, attempts = %s, error = %s, updated_at = %s
                WHERE job_id = %s
                RETURNING *
                """,
                (status, attempts, error, _utc_now(), job_id),
            ).fetchone()
            self._connection.commit()
            return self._record(row)

    def claim(self, job_id: UUID, *, worker_id: str, lease_seconds: float = 60) -> JobRecord:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0")
        with self._lock:
            current = self.get(job_id)
            now = _utc_now()
            if current.status not in ("queued", "running"):
                raise ValueError(f"cannot claim job in {current.status} state")
            if current.status == "running" and current.lease_expires_at:
                expires_at = datetime.fromisoformat(current.lease_expires_at)
                if expires_at > now and current.lease_owner != worker_id:
                    raise RuntimeError("job is leased by another worker")
            expires = now + timedelta(seconds=lease_seconds)
            attempts = current.attempts + (current.status == "queued")
            row = self._connection.execute(
                """
                UPDATE jobs
                SET status = 'running', attempts = %s, lease_owner = %s,
                    lease_expires_at = %s, updated_at = %s
                WHERE job_id = %s
                RETURNING *
                """,
                (attempts, worker_id, expires, now, job_id),
            ).fetchone()
            self._connection.commit()
            return self._record(row)

    def heartbeat(
        self, job_id: UUID, *, worker_id: str, lease_seconds: float = 60
    ) -> JobRecord:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0")
        with self._lock:
            current = self.get(job_id)
            self._require_active_lease(current, worker_id)
            now = _utc_now()
            expires = now + timedelta(seconds=lease_seconds)
            row = self._connection.execute(
                """
                UPDATE jobs SET lease_expires_at = %s, updated_at = %s
                WHERE job_id = %s
                RETURNING *
                """,
                (expires, now, job_id),
            ).fetchone()
            self._connection.commit()
            return self._record(row)

    def requeue_expired_leases(
        self, *, now: datetime | None = None, kind: str | None = None
    ) -> tuple[UUID, ...]:
        current_time = now or _utc_now()
        where = [
            "status = 'running'",
            "lease_expires_at IS NOT NULL",
            "lease_expires_at <= %s",
        ]
        values: list[Any] = [current_time]
        if kind is not None:
            where.append("kind = %s")
            values.append(kind)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT job_id, cancel_requested FROM jobs
                WHERE {' AND '.join(where)}
                ORDER BY created_at, sequence
                """,
                values,
            ).fetchall()
            for row in rows:
                if row["cancel_requested"]:
                    status = "cancelled"
                    error = "job cancellation requested while lease expired"
                else:
                    status = "queued"
                    error = None
                self._connection.execute(
                    """
                    UPDATE jobs SET status = %s, error = %s, lease_owner = NULL,
                        lease_expires_at = NULL, updated_at = %s
                    WHERE job_id = %s
                    """,
                    (status, error, current_time, row["job_id"]),
                )
            self._connection.commit()
        return tuple(UUID(str(row["job_id"])) for row in rows)

    def request_cancel(self, job_id: UUID) -> JobRecord:
        with self._lock:
            current = self.get(job_id)
            if current.status in ("queued", "paused"):
                self._connection.execute(
                    """
                    UPDATE jobs SET status = 'cancelled', cancel_requested = TRUE, updated_at = %s
                    WHERE job_id = %s
                    """,
                    (_utc_now(), job_id),
                )
            elif current.status == "running":
                self._connection.execute(
                    "UPDATE jobs SET cancel_requested = TRUE, updated_at = %s WHERE job_id = %s",
                    (_utc_now(), job_id),
                )
            self._connection.commit()
            return self.get(job_id)

    @staticmethod
    def _require_active_lease(current: JobRecord, worker_id: str) -> None:
        if current.status != "running" or current.lease_owner != worker_id:
            raise RuntimeError("worker does not own an active job lease")
        if (
            current.lease_expires_at is None
            or datetime.fromisoformat(current.lease_expires_at) <= _utc_now()
        ):
            raise RuntimeError("job lease has expired")

    def _fetch_by_idempotency_key(
        self, tenant_id: UUID, idempotency_key: str
    ) -> JobRecord | None:
        row = self._connection.execute(
            "SELECT * FROM jobs WHERE tenant_id = %s AND idempotency_key = %s",
            (tenant_id, idempotency_key),
        ).fetchone()
        return None if row is None else self._record(row)

    @staticmethod
    def _record(row: Any) -> JobRecord:
        payload = row["payload"]
        if not isinstance(payload, dict):
            payload = json.loads(payload)
        lease = row["lease_expires_at"]
        return JobRecord(
            job_id=UUID(str(row["job_id"])),
            tenant_id=UUID(str(row["tenant_id"])),
            kind=row["kind"],
            status=row["status"],
            payload=payload,
            idempotency_key=row["idempotency_key"],
            attempts=row["attempts"],
            error=row["error"],
            created_at=_iso(row["created_at"]),
            updated_at=_iso(row["updated_at"]),
            lease_owner=row["lease_owner"],
            lease_expires_at=None if lease is None else _iso(lease),
            cancel_requested=bool(row["cancel_requested"]),
        )
