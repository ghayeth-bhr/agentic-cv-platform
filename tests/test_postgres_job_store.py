"""Contract tests shared by the SQLite and PostgreSQL job stores."""

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from agentic_cv.job_store import JobStore


def exercise_job_contract(store):
    tenant = uuid4()
    job = store.create(
        tenant_id=tenant,
        kind="dataset-export",
        payload={"version_id": "v1"},
        idempotency_key="export-1",
    )

    assert job.status == "queued"
    assert job.attempts == 0
    assert store.get(job.job_id) == job
    assert store.create(
        tenant_id=tenant,
        kind="dataset-export",
        payload={"version_id": "v1"},
        idempotency_key="export-1",
    ) == job
    with pytest.raises(ValueError, match="different job"):
        store.create(
            tenant_id=tenant,
            kind="dataset-export",
            payload={"version_id": "v2"},
            idempotency_key="export-1",
        )
    with pytest.raises(KeyError, match="unknown job"):
        store.get(uuid4())

    assert [record.job_id for record in store.list(status="queued")] == [job.job_id]
    assert [record.job_id for record in store.list(kind="other")] == []

    running = store.transition(job.job_id, "running")
    assert running.status == "running"
    assert running.attempts == 1
    succeeded = store.transition(job.job_id, "succeeded")
    assert succeeded.status == "succeeded"
    with pytest.raises(ValueError, match="cannot transition"):
        store.transition(job.job_id, "failed")

    claim_job = store.create(
        tenant_id=tenant,
        kind="dataset-export",
        payload={"item": 1},
        idempotency_key="claim-1",
    )
    claimed = store.claim(claim_job.job_id, worker_id="worker-a", lease_seconds=60)
    assert claimed.status == "running"
    assert claimed.lease_owner == "worker-a"
    assert claimed.attempts == 1
    heartbeated = store.heartbeat(claim_job.job_id, worker_id="worker-a", lease_seconds=120)
    assert heartbeated.lease_expires_at is not None
    assert datetime.fromisoformat(heartbeated.lease_expires_at) >= datetime.fromisoformat(
        claimed.lease_expires_at
    )
    with pytest.raises(RuntimeError, match="leased by another worker"):
        store.claim(claim_job.job_id, worker_id="worker-b", lease_seconds=60)

    queued_cancel = store.create(
        tenant_id=tenant,
        kind="label-batch",
        payload={"item": 2},
        idempotency_key="cancel-q",
    )
    cancelled = store.request_cancel(queued_cancel.job_id)
    assert cancelled.status == "cancelled"
    assert cancelled.cancel_requested is True

    running_cancel = store.create(
        tenant_id=tenant,
        kind="label-batch",
        payload={"item": 3},
        idempotency_key="cancel-r",
    )
    store.claim(running_cancel.job_id, worker_id="worker-a", lease_seconds=60)
    requested = store.request_cancel(running_cancel.job_id)
    assert requested.status == "running"
    assert requested.cancel_requested is True

    expired = store.create(
        tenant_id=tenant,
        kind="dataset-export",
        payload={"item": 4},
        idempotency_key="expire-1",
    )
    store.claim(expired.job_id, worker_id="worker-a", lease_seconds=60)

    future = datetime.now(timezone.utc) + timedelta(days=1)
    requeued_ids = set(store.requeue_expired_leases(now=future))
    assert {claim_job.job_id, running_cancel.job_id, expired.job_id} <= requeued_ids

    assert store.get(claim_job.job_id).status == "queued"
    assert store.get(claim_job.job_id).lease_owner is None
    assert store.get(expired.job_id).status == "queued"
    cancelled_after_expiry = store.get(running_cancel.job_id)
    assert cancelled_after_expiry.status == "cancelled"
    assert "cancellation requested" in (cancelled_after_expiry.error or "")


def test_sqlite_job_contract(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    try:
        exercise_job_contract(store)
    finally:
        store.close()


@pytest.mark.postgres
def test_postgres_job_contract():
    dsn = os.environ.get("AGENTIC_CV_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set AGENTIC_CV_TEST_POSTGRES_DSN to run against a live PostgreSQL")
    from agentic_cv.postgres import PostgresConfig, connect
    from agentic_cv.postgres_job_store import open_postgres_job_store

    schema = f"agentic_cv_test_{uuid4().hex[:8]}"
    store = None
    try:
        store = open_postgres_job_store(PostgresConfig(dsn=dsn, schema=schema))
        exercise_job_contract(store)
    finally:
        if store is not None:
            store.close()
        cleanup = connect(PostgresConfig(dsn=dsn))
        cleanup.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()
