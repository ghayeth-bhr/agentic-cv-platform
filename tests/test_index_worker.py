from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import UUID, uuid4

from agentic_cv.index_worker import INDEX_JOB_KIND, IndexWorker
from agentic_cv.job_store import JobStore


@dataclass
class RecordingIndexer:
    calls: list[tuple[UUID, UUID]]

    def index_asset(self, *, tenant_id: UUID, asset_id: UUID):
        self.calls.append((tenant_id, asset_id))
        return SimpleNamespace(indexed_records=2, sources_indexed=1, detail="indexed")


@dataclass
class FailingIndexer:
    def index_asset(self, *, tenant_id: UUID, asset_id: UUID):
        raise RuntimeError("embedding model unavailable")


def test_index_worker_executes_queued_job_and_persists_result(tmp_path):
    jobs = JobStore(str(tmp_path / "jobs.sqlite3"))
    indexer = RecordingIndexer(calls=[])
    worker = IndexWorker(jobs=jobs, indexer=indexer)  # type: ignore[arg-type]
    tenant_id = uuid4()
    asset_id = uuid4()

    job = worker.submit(tenant_id=tenant_id, asset_id=asset_id)
    processed = worker.run_once()

    assert processed == (job.job_id,)
    assert jobs.get(job.job_id).status == "succeeded"
    assert jobs.get(job.job_id).kind == INDEX_JOB_KIND
    assert indexer.calls == [(tenant_id, asset_id)]
    jobs.close()


def test_index_worker_persists_failure_and_can_retry(tmp_path):
    jobs = JobStore(str(tmp_path / "jobs.sqlite3"))
    worker = IndexWorker(jobs=jobs, indexer=FailingIndexer())  # type: ignore[arg-type]
    job = worker.submit(tenant_id=uuid4(), asset_id=uuid4())

    worker.run_once()

    failed = jobs.get(job.job_id)
    assert failed.status == "failed"
    assert "embedding model unavailable" in (failed.error or "")

    # A failed job can be requeued by the durable store, then retried.
    jobs.transition(job.job_id, "queued")
    worker.run_once()
    assert jobs.get(job.job_id).status == "failed"
    jobs.close()


def test_index_worker_requeues_expired_lease(tmp_path):
    jobs = JobStore(str(tmp_path / "jobs.sqlite3"))
    indexer = RecordingIndexer(calls=[])
    worker = IndexWorker(jobs=jobs, indexer=indexer)  # type: ignore[arg-type]
    tenant_id = uuid4()
    asset_id = uuid4()
    job = worker.submit(tenant_id=tenant_id, asset_id=asset_id)
    jobs.claim(job.job_id, worker_id="stale-worker", lease_seconds=0.001)
    time.sleep(0.02)

    processed = worker.run_once()

    assert job.job_id in processed
    assert jobs.get(job.job_id).status == "succeeded"
    assert indexer.calls == [(tenant_id, asset_id)]
    jobs.close()
