"""Local polling worker for durable dataset export jobs."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from uuid import UUID

from .dataset_export_workflow import DatasetExportWorkflow
from .job_store import JobStoreBackend

logger = logging.getLogger("agentic_cv.worker")


@dataclass(frozen=True, kw_only=True)
class DatasetExportWorker:
    jobs: JobStoreBackend
    workflow: DatasetExportWorkflow

    def run_once(self, *, worker_id: str = "export-worker") -> tuple[UUID, ...]:
        """Attempt each queued export once and return the jobs inspected."""

        requeued = list(self.jobs.requeue_expired_leases(kind="dataset-export"))
        if requeued:
            logger.info(
                "requeued expired export leases",
                extra={"worker_id": worker_id, "count": len(requeued), "kind": "dataset-export"},
            )
        processed: list[UUID] = list(requeued)
        processed_set = set(processed)
        for job in self.jobs.list(status="queued", kind="dataset-export"):
            if job.job_id not in processed_set:
                processed.append(job.job_id)
            start = time.perf_counter()
            logger.info(
                "export job attempt started",
                extra={
                    "worker_id": worker_id,
                    "job_id": str(job.job_id),
                    "tenant_id": str(job.tenant_id),
                    "kind": job.kind,
                },
            )
            try:
                self.workflow.execute(
                    job.job_id, tenant_id=job.tenant_id, worker_id=worker_id
                )
            except Exception:
                duration_ms = round((time.perf_counter() - start) * 1000, 3)
                status = "unknown"
                error = None
                try:
                    current = self.jobs.get(job.job_id)
                    status = current.status
                    error = getattr(current, "error", None)
                except Exception:
                    pass
                logger.exception(
                    "export job attempt failed",
                    extra={
                        "worker_id": worker_id,
                        "job_id": str(job.job_id),
                        "status": status,
                        "duration_ms": duration_ms,
                        "error": error,
                    },
                )
                # The workflow has already persisted failure/cancellation state.
                # Continue so one bad export cannot starve the queue.
                continue
            duration_ms = round((time.perf_counter() - start) * 1000, 3)
            status = "unknown"
            try:
                status = self.jobs.get(job.job_id).status
            except Exception:
                pass
            logger.info(
                "export job attempt finished",
                extra={
                    "worker_id": worker_id,
                    "job_id": str(job.job_id),
                    "status": status,
                    "duration_ms": duration_ms,
                },
            )
        return tuple(processed)

    def run_forever(
        self,
        *,
        worker_id: str = "export-worker",
        poll_interval: float = 1.0,
        stop_event: Event | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if poll_interval < 0:
            raise ValueError("poll_interval must not be negative")
        stop_event = stop_event or Event()
        logger.info(
            "export worker loop started",
            extra={"worker_id": worker_id, "poll_interval": poll_interval},
        )
        try:
            while not stop_event.is_set():
                self.run_once(worker_id=worker_id)
                if not stop_event.is_set() and poll_interval:
                    sleep_fn(poll_interval)
        finally:
            logger.info("export worker loop stopped", extra={"worker_id": worker_id})