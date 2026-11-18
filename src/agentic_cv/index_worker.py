"""Durable semantic-index jobs for the local worker."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from uuid import UUID, uuid4

from .job_store import JobRecord, JobStoreBackend
from .semantic_indexing import AssetSemanticIndexer

INDEX_JOB_KIND = "index-asset"
logger = logging.getLogger("agentic_cv.worker")


@dataclass(frozen=True, kw_only=True)
class IndexWorker:
    jobs: JobStoreBackend
    indexer: AssetSemanticIndexer

    def submit(
        self,
        *,
        tenant_id: UUID,
        asset_id: UUID,
        idempotency_key: str | None = None,
    ) -> JobRecord:
        key = idempotency_key or f"{INDEX_JOB_KIND}:{asset_id}:{uuid4()}"
        return self.jobs.create(
            tenant_id=tenant_id,
            kind=INDEX_JOB_KIND,
            payload={"asset_id": str(asset_id)},
            idempotency_key=key,
        )

    def run_once(self, *, worker_id: str = "index-worker") -> tuple[UUID, ...]:
        requeued = list(self.jobs.requeue_expired_leases(kind=INDEX_JOB_KIND))
        if requeued:
            logger.info(
                "requeued expired index leases",
                extra={"worker_id": worker_id, "count": len(requeued), "kind": INDEX_JOB_KIND},
            )
        processed: list[UUID] = list(requeued)
        seen = set(processed)
        for job in self.jobs.list(status="queued", kind=INDEX_JOB_KIND):
            if job.job_id not in seen:
                processed.append(job.job_id)
                seen.add(job.job_id)
            start = time.perf_counter()
            try:
                claimed = self.jobs.claim(job.job_id, worker_id=worker_id)
            except Exception:
                # Another worker may have claimed the job between list() and
                # claim(). Leave it to that worker rather than failing it.
                continue
            try:
                if claimed.cancel_requested:
                    self.jobs.transition(claimed.job_id, "cancelled")
                    continue
                asset_id = UUID(str(claimed.payload.get("asset_id")))
                result = self.indexer.index_asset(
                    tenant_id=claimed.tenant_id, asset_id=asset_id
                )
                self.jobs.transition(claimed.job_id, "succeeded")
            except Exception as exc:
                logger.exception(
                    "index job attempt failed",
                    extra={
                        "worker_id": worker_id,
                        "job_id": str(job.job_id),
                        "duration_ms": round((time.perf_counter() - start) * 1000, 3),
                    },
                )
                try:
                    current = self.jobs.get(job.job_id)
                    if current.status == "running":
                        self.jobs.transition(
                            job.job_id, "failed", error=str(exc)[:1000] or "indexing failed"
                        )
                except Exception:
                    # Keep the queue moving even when the job row cannot be
                    # updated (for example after a shutdown race).
                    pass
            else:
                logger.info(
                    "index job attempt finished",
                    extra={
                        "worker_id": worker_id,
                        "job_id": str(job.job_id),
                        "status": "succeeded",
                        "indexed_records": result.indexed_records,
                        "duration_ms": round((time.perf_counter() - start) * 1000, 3),
                    },
                )
        return tuple(processed)

    def run_forever(
        self,
        *,
        worker_id: str = "index-worker",
        poll_interval: float = 1.0,
        stop_event: Event | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if poll_interval < 0:
            raise ValueError("poll_interval must not be negative")
        stop_event = stop_event or Event()
        logger.info(
            "index worker loop started",
            extra={"worker_id": worker_id, "poll_interval": poll_interval},
        )
        try:
            while not stop_event.is_set():
                self.run_once(worker_id=worker_id)
                if not stop_event.is_set() and poll_interval:
                    sleep_fn(poll_interval)
        finally:
            logger.info("index worker loop stopped", extra={"worker_id": worker_id})
