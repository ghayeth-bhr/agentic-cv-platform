"""Durable orchestration for dataset export jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from .contracts import EventEnvelope
from .dataset_export import DatasetExportResult, DatasetExportService, ExportFormat
from .dataset_store import DatasetVersionStore
from .job_store import JobRecord, JobStore
from .label_workflow import JobCancelled
from .outbox import Outbox


@dataclass(frozen=True, kw_only=True)
class DatasetExportWorkflowDependencies:
    jobs: JobStore
    versions: DatasetVersionStore
    exports: DatasetExportService
    outbox: Outbox


class DatasetExportWorkflow:
    def __init__(self, dependencies: DatasetExportWorkflowDependencies) -> None:
        self.dependencies = dependencies

    def submit(
        self,
        *,
        tenant_id: UUID,
        version_id: UUID,
        format: ExportFormat,
        idempotency_key: str,
    ) -> JobRecord:
        if format not in ("coco", "yolo-detection", "yolo-segmentation"):
            raise ValueError("format must be coco, yolo-detection, or yolo-segmentation")
        self.dependencies.versions.get(tenant_id=tenant_id, version_id=version_id)
        return self.dependencies.jobs.create(
            tenant_id=tenant_id,
            kind="dataset-export",
            payload={"version_id": str(version_id), "format": format},
            idempotency_key=idempotency_key,
        )

    def execute(
        self,
        job_id: UUID,
        *,
        tenant_id: UUID,
        worker_id: str = "export-worker",
        lease_seconds: float = 60,
    ) -> DatasetExportResult:
        job = self.dependencies.jobs.get(job_id)
        if job.tenant_id != tenant_id:
            raise PermissionError("job does not belong to tenant")
        if job.cancel_requested or job.status == "cancelled":
            raise JobCancelled(f"job {job_id} was cancelled before execution")
        job = self.dependencies.jobs.claim(job_id, worker_id=worker_id, lease_seconds=lease_seconds)
        try:
            current = self.dependencies.jobs.get(job_id)
            if current.cancel_requested:
                raise JobCancelled(f"job {job_id} cancellation requested")
            result = self.dependencies.exports.export(
                tenant_id=tenant_id,
                version_id=UUID(job.payload["version_id"]),
                format=job.payload["format"],
            )
            self.dependencies.jobs.transition(job_id, "succeeded")
            self.dependencies.outbox.enqueue(
                EventEnvelope(
                    event_type="DatasetExportCompleted",
                    tenant_id=tenant_id,
                    aggregate_type="job",
                    aggregate_id=job_id,
                    payload={"version_id": str(result.version_id), "format": result.format,
                             "output_dir": str(result.output_dir)},
                    idempotency_key=f"dataset-export-completed:{job_id}",
                )
            )
            return result
        except JobCancelled as exc:
            self.dependencies.jobs.transition(job_id, "cancelled", error=str(exc))
            self.dependencies.outbox.enqueue(
                EventEnvelope(
                    event_type="DatasetExportCancelled",
                    tenant_id=tenant_id,
                    aggregate_type="job",
                    aggregate_id=job_id,
                    payload={"reason": str(exc)},
                    idempotency_key=f"dataset-export-cancelled:{job_id}",
                )
            )
            raise
        except Exception as exc:
            self.dependencies.jobs.transition(job_id, "failed", error=str(exc))
            self.dependencies.outbox.enqueue(
                EventEnvelope(
                    event_type="DatasetExportFailed",
                    tenant_id=tenant_id,
                    aggregate_type="job",
                    aggregate_id=job_id,
                    payload={"error": str(exc)},
                    idempotency_key=f"dataset-export-failed:{job_id}",
                )
            )
            raise