"""Local polling worker for durable dataset export jobs."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from time import sleep
from uuid import UUID
from collections.abc import Callable

from .dataset_export_workflow import DatasetExportWorkflow
from .job_store import JobStoreBackend


@dataclass(frozen=True, kw_only=True)
class DatasetExportWorker:
    jobs: JobStoreBackend
    workflow: DatasetExportWorkflow

    def run_once(self, *, worker_id: str = "export-worker") -> tuple[UUID, ...]:
        """Attempt each queued export once and return the jobs inspected."""

        processed: list[UUID] = list(
            self.jobs.requeue_expired_leases(kind="dataset-export")
        )
        processed_set = set(processed)
        for job in self.jobs.list(status="queued", kind="dataset-export"):
            if job.job_id not in processed_set:
                processed.append(job.job_id)
            try:
                self.workflow.execute(
                    job.job_id, tenant_id=job.tenant_id, worker_id=worker_id
                )
            except Exception:
                # The workflow has already persisted failure/cancellation state.
                # Continue so one bad export cannot starve the queue.
                continue
        return tuple(processed)

    def run_forever(
        self,
        *,
        worker_id: str = "export-worker",
        poll_interval: float = 1.0,
        stop_event: Event | None = None,
        sleep_fn: Callable[[float], None] = sleep,
    ) -> None:
        if poll_interval < 0:
            raise ValueError("poll_interval must not be negative")
        stop_event = stop_event or Event()
        while not stop_event.is_set():
            self.run_once(worker_id=worker_id)
            if not stop_event.is_set() and poll_interval:
                sleep_fn(poll_interval)