from uuid import uuid4

import pytest

from agentic_cv.job_store import JobStore


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
