"""Durable application workflow for batch SAM3 labeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from .annotation_store import AnnotationRevisionStore
from .batch_labeling import BatchLabelResult, LabelImage, Sam3BatchLabeler
from .contracts import EventEnvelope
from .dataset import DatasetCategory
from .job_store import JobRecord, JobStore
from .outbox import Outbox


@dataclass(frozen=True, kw_only=True)
class LabelWorkflowDependencies:
    jobs: JobStore
    annotations: AnnotationRevisionStore
    outbox: Outbox
    labeler: Sam3BatchLabeler


class DurableLabelWorkflow:
    def __init__(self, dependencies: LabelWorkflowDependencies) -> None:
        self.dependencies = dependencies

    def submit(
        self,
        *,
        tenant_id: UUID,
        asset_id: str,
        prompt: str,
        threshold: float,
        task: Literal["detection", "segmentation"],
        category_id: int,
        category_name: str,
        idempotency_key: str,
    ) -> JobRecord:
        return self.dependencies.jobs.create(
            tenant_id=tenant_id,
            kind="batch-label",
            payload={
                "asset_id": asset_id,
                "prompt": prompt,
                "threshold": threshold,
                "task": task,
                "category_id": category_id,
                "category_name": category_name,
            },
            idempotency_key=idempotency_key,
        )

    def execute(
        self,
        job_id: UUID,
        *,
        images: list[LabelImage],
        tenant_id: UUID,
        prompt: str,
        threshold: float,
        task: Literal["detection", "segmentation"],
        category: DatasetCategory,
    ) -> BatchLabelResult:
        job = self.dependencies.jobs.get(job_id)
        if job.tenant_id != tenant_id:
            raise PermissionError("job does not belong to tenant")
        job = self.dependencies.jobs.transition(job_id, "running")
        try:
            result = self.dependencies.labeler.label(
                images, prompt=prompt, threshold=threshold, category=category, task=task
            )
            for annotation in result.dataset.annotations:
                self.dependencies.annotations.save(
                    tenant_id=tenant_id,
                    annotation=annotation,
                    author_id="sam3-batch",
                )
            self.dependencies.jobs.transition(job.job_id, "succeeded")
            self.dependencies.outbox.enqueue(
                EventEnvelope(
                    event_type="LabelRunCompleted",
                    tenant_id=tenant_id,
                    aggregate_type="job",
                    aggregate_id=job_id,
                    payload={"annotation_count": len(result.dataset.annotations)},
                    idempotency_key=f"label-completed:{job_id}",
                )
            )
            return result
        except Exception as exc:
            self.dependencies.jobs.transition(job.job_id, "failed", error=str(exc))
            self.dependencies.outbox.enqueue(
                EventEnvelope(
                    event_type="LabelRunFailed",
                    tenant_id=tenant_id,
                    aggregate_type="job",
                    aggregate_id=job_id,
                    payload={"error": str(exc)},
                    idempotency_key=f"label-failed:{job_id}",
                )
            )
            raise
