"""Local polling worker for durable dataset export jobs."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from .dataset_export_workflow import DatasetExportWorkflow
from .job_store import JobStore


@dataclass(frozen=True, kw_only=True)
class DatasetExportWorker:
    jobs: JobStore
    workflow: DatasetExportWorkflow

    def run_once(self, *, worker_id: str = "export-worker") -> tuple[UUID, ...]:
        """Attempt each queued export once and return the jobs inspected."""

        processed: list[UUID] = []
        for job in self.jobs.list(status="queued", kind="dataset-export"):
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