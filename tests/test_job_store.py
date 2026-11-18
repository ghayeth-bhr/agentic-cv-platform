import sqlite3
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from agentic_cv.job_store import JobStore

FIXED_CREATED_AT = "2026-01-01T00:00:00+00:00"


def test_create_is_idempotent_and_survives_reopen(tmp_path):
    path = str(tmp_path / "jobs.sqlite3")
    tenant_id = uuid4()
    store = JobStore(path)
    first = store.create(
        tenant_id=tenant_id,
        kind="label-run",
        payload={"asset_id": "asset-1"},
        idempotency_key="label-run:asset-1:v1",
    )
    second = store.create(
        tenant_id=tenant_id,
        kind="label-run",
        payload={"asset_id": "asset-1"},
        idempotency_key="label-run:asset-1:v1",
    )
    store.close()

    reopened = JobStore(path)
    restored = reopened.get(first.job_id)

    assert second == first
    assert restored.payload == {"asset_id": "asset-1"}
    reopened.close()


def test_transitions_guard_state_and_count_attempts(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    job = store.create(tenant_id=uuid4(), kind="embedding", payload={}, idempotency_key="embedding:1")

    running = store.transition(job.job_id, "running")
    failed = store.transition(job.job_id, "failed", error="worker stopped")
    retry = store.transition(job.job_id, "queued")

    assert running.attempts == 1
    assert failed.error == "worker stopped"
    assert retry.status == "queued"
    with pytest.raises(ValueError, match="cannot transition"):
        store.transition(job.job_id, "succeeded")
    store.close()


def test_idempotency_key_cannot_change_payload(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    tenant_id = uuid4()
    store.create(
        tenant_id=tenant_id,
        kind="extract",
        payload={"source": "one"},
        idempotency_key="extract:1",
    )

    with pytest.raises(ValueError, match="different job"):
        store.create(
            tenant_id=tenant_id,
            kind="extract",
            payload={"source": "two"},
            idempotency_key="extract:1",
        )
    store.close()


def test_claim_heartbeat_and_expired_lease_recovery(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    job = store.create(tenant_id=uuid4(), kind="extract", payload={}, idempotency_key="extract:lease")

    claimed = store.claim(job.job_id, worker_id="worker-a", lease_seconds=60)
    assert claimed.status == "running"
    assert claimed.lease_owner == "worker-a"
    assert claimed.attempts == 1
    renewed = store.heartbeat(job.job_id, worker_id="worker-a", lease_seconds=60)
    assert renewed.lease_owner == "worker-a"

    with pytest.raises(RuntimeError, match="another worker"):
        store.claim(job.job_id, worker_id="worker-b")
    with pytest.raises(RuntimeError, match="own"):
        store.heartbeat(job.job_id, worker_id="worker-b")
    store.close()


def test_cancel_is_durable_and_queued_jobs_cancel_immediately(tmp_path):
    path = str(tmp_path / "jobs.sqlite3")
    store = JobStore(path)
    job = store.create(tenant_id=uuid4(), kind="label", payload={}, idempotency_key="label:cancel")

    cancelled = store.request_cancel(job.job_id)
    assert cancelled.status == "cancelled"
    assert cancelled.cancel_requested
    store.close()

    reopened = JobStore(path)
    assert reopened.get(job.job_id).cancel_requested
    with pytest.raises(ValueError, match="cancelled"):
        reopened.claim(job.job_id, worker_id="worker-a")
    reopened.close()


def test_list_filters_jobs_by_status_and_kind(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    tenant_id = uuid4()
    export = store.create(tenant_id=tenant_id, kind="dataset-export", payload={}, idempotency_key="export")
    store.create(tenant_id=tenant_id, kind="label", payload={}, idempotency_key="label")

    assert [job.job_id for job in store.list(status="queued", kind="dataset-export")] == [export.job_id]
    store.close()


def test_expired_leases_are_requeued_or_cancelled(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    now = datetime.now(timezone.utc)
    queued_job = store.create(tenant_id=uuid4(), kind="dataset-export", payload={}, idempotency_key="expired")
    cancelled_job = store.create(tenant_id=uuid4(), kind="dataset-export", payload={}, idempotency_key="cancelled")
    store.claim(queued_job.job_id, worker_id="worker-a", lease_seconds=60)
    store.claim(cancelled_job.job_id, worker_id="worker-a", lease_seconds=60)
    store.request_cancel(cancelled_job.job_id)
    expired = now + timedelta(seconds=120)

    recovered = store.requeue_expired_leases(now=expired)

    assert recovered == (queued_job.job_id, cancelled_job.job_id)
    assert store.get(queued_job.job_id).status == "queued"
    assert store.get(queued_job.job_id).lease_owner is None
    assert store.get(cancelled_job.job_id).status == "cancelled"
    store.close()


def _force_creation_order(path, first_id, second_id, *, created_at=FIXED_CREATED_AT):
    """Give two existing jobs one timestamp and UUIDs that reverse insertion order."""
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE jobs SET job_id = ?, created_at = ? WHERE job_id = ?",
        (str(UUID(int=2)), created_at, str(first_id)),
    )
    connection.execute(
        "UPDATE jobs SET job_id = ?, created_at = ? WHERE job_id = ?",
        (str(UUID(int=1)), created_at, str(second_id)),
    )
    connection.commit()
    connection.close()


def _legacy_jobs_database(path, rows):
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
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
    for job_id, tenant_id, key in rows:
        connection.execute(
            """
            INSERT INTO jobs (
                job_id, tenant_id, kind, status, payload, idempotency_key, attempts,
                error, lease_owner, lease_expires_at, cancel_requested, created_at,
                updated_at
            ) VALUES (?, ?, 'dataset-export', 'queued', '{}', ?, 0, NULL, NULL, NULL, 0, ?, ?)
            """,
            (str(job_id), str(tenant_id), key, FIXED_CREATED_AT, FIXED_CREATED_AT),
        )
    connection.commit()
    connection.close()


def test_list_preserves_creation_order_when_timestamps_tie(tmp_path):
    path = str(tmp_path / "jobs.sqlite3")
    tenant_id = uuid4()
    store = JobStore(path)
    first = store.create(
        tenant_id=tenant_id, kind="dataset-export", payload={}, idempotency_key="first"
    )
    second = store.create(
        tenant_id=tenant_id, kind="dataset-export", payload={}, idempotency_key="second"
    )
    _force_creation_order(path, first.job_id, second.job_id)

    assert [record.job_id for record in store.list()] == [UUID(int=2), UUID(int=1)]
    store.close()


def test_expired_lease_recovery_preserves_creation_order_when_timestamps_tie(tmp_path):
    path = str(tmp_path / "jobs.sqlite3")
    tenant_id = uuid4()
    store = JobStore(path)
    first = store.create(
        tenant_id=tenant_id, kind="dataset-export", payload={}, idempotency_key="first"
    )
    second = store.create(
        tenant_id=tenant_id, kind="dataset-export", payload={}, idempotency_key="second"
    )
    _force_creation_order(path, first.job_id, second.job_id)
    store.claim(UUID(int=2), worker_id="worker-a", lease_seconds=60)
    store.claim(UUID(int=1), worker_id="worker-a", lease_seconds=60)

    recovered = store.requeue_expired_leases(
        now=datetime.now(timezone.utc) + timedelta(seconds=120)
    )

    assert recovered == (UUID(int=2), UUID(int=1))
    store.close()


def test_legacy_jobs_table_is_migrated_and_keeps_insertion_order(tmp_path):
    path = str(tmp_path / "jobs.sqlite3")
    tenant_id = uuid4()
    _legacy_jobs_database(
        path,
        [(UUID(int=2), tenant_id, "first"), (UUID(int=1), tenant_id, "second")],
    )

    store = JobStore(path)

    assert [record.job_id for record in store.list()] == [UUID(int=2), UUID(int=1)]
    assert store.get(UUID(int=1)).idempotency_key == "second"
    store.close()
