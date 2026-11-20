"""Durable application workflow for batch SAM3 labeling."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Callable, Literal
from uuid import UUID

from .annotation_sets import job_set_id, live_in_set
from .annotation_store import AnnotationRevisionBackend
from .batch_labeling import BatchLabelResult, LabelClass, LabelImage, Sam3BatchLabeler
from .contracts import EventEnvelope
from .dataset import DatasetCategory
from .job_store import JobRecord, JobStoreBackend
from .outbox import OutboxBackend
from .label_monitor import LabelHealthMonitor, LabelHealthStore
from .preview import PreviewBackend

logger = logging.getLogger("agentic_cv.workflow")


class JobCancelled(RuntimeError):
    """Raised when a running label job observes a cancellation request."""


class JobPaused(RuntimeError):
    """Raised when the health monitor pauses a run to ask the user."""

    def __init__(self, alert: dict) -> None:
        super().__init__(alert.get("message", "labeling paused"))
        self.alert = alert


@dataclass(frozen=True, kw_only=True)
class LabelPlan:
    asset_ids: tuple[str, ...]
    classes: tuple[LabelClass, ...]

    @property
    def categories(self) -> tuple[DatasetCategory, ...]:
        return tuple(item.category for item in self.classes)


def label_plan(payload: dict) -> LabelPlan:
    """Read a batch-label payload: v2 (``asset_ids`` + ``classes``) or v1 (one asset, one class)."""

    asset_ids = tuple(str(item) for item in payload.get("asset_ids") or ()) or (
        str(payload["asset_id"]),
    )
    raw_classes = payload.get("classes") or [
        {
            "category_id": payload["category_id"],
            "name": payload["category_name"],
            "prompt": payload["prompt"],
            "threshold": payload["threshold"],
        }
    ]
    classes = tuple(
        LabelClass(
            prompt=str(item["prompt"]),
            threshold=float(item["threshold"]),
            category=DatasetCategory(category_id=int(item["category_id"]), name=str(item["name"])),
        )
        for item in raw_classes
    )
    return LabelPlan(asset_ids=asset_ids, classes=classes)


@dataclass(frozen=True, kw_only=True)
class LabelWorkflowDependencies:
    jobs: JobStoreBackend
    annotations: AnnotationRevisionBackend
    outbox: OutboxBackend
    labeler: Sam3BatchLabeler
    preview: PreviewBackend | None = None
    health: LabelHealthStore | None = None


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
        asset_ids: list[str] | None = None,
        classes: list[LabelClass] | None = None,
    ) -> JobRecord:
        payload = {
            "asset_id": asset_id,
            "prompt": prompt,
            "threshold": threshold,
            "task": task,
            "category_id": category_id,
            "category_name": category_name,
        }
        if asset_ids:
            payload["asset_ids"] = list(asset_ids)
        if classes:
            payload["classes"] = [
                {
                    "category_id": item.category.category_id,
                    "name": item.category.name,
                    "prompt": item.prompt,
                    "threshold": item.threshold,
                }
                for item in classes
            ]
        return self.dependencies.jobs.create(
            tenant_id=tenant_id,
            kind="batch-label",
            payload=payload,
            idempotency_key=idempotency_key,
        )

    def execute(
        self,
        job_id: UUID,
        *,
        images: list[LabelImage] | None = None,
        image_loader: Callable[[], list[LabelImage]] | None = None,
        tenant_id: UUID,
        prompt: str,
        threshold: float,
        task: Literal["detection", "segmentation"],
        category: DatasetCategory | None = None,
        classes: list[LabelClass] | None = None,
        worker_id: str = "label-worker",
        lease_seconds: float = 60,
    ) -> BatchLabelResult:
        job = self.dependencies.jobs.get(job_id)
        if job.tenant_id != tenant_id:
            raise PermissionError("job does not belong to tenant")
        if job.cancel_requested or job.status == "cancelled":
            raise JobCancelled(f"job {job_id} was cancelled before execution")
        job = self.dependencies.jobs.claim(
            job_id, worker_id=worker_id, lease_seconds=lease_seconds
        )
        started = time.perf_counter()
        logger.info(
            "label workflow started",
            extra={
                "worker_id": worker_id,
                "job_id": str(job_id),
                "tenant_id": str(tenant_id),
                "image_count": len(images) if images is not None else None,
                "task": task,
            },
        )

        set_id = job_set_id(job_id)
        monitor: LabelHealthMonitor | None = None
        skip: set[str] = set()

        def start_monitor(all_images: list[LabelImage], label_classes: list[LabelClass]) -> None:
            nonlocal monitor, skip
            health = self.dependencies.health
            if health is None:
                return
            if self.dependencies.preview is not None:
                # Resumed run: images that already published a preview event
                # were labeled (and saved) by the earlier attempt.
                skip = {
                    event.image_id
                    for event in self.dependencies.preview.list_for_job(tenant_id=tenant_id, job_id=job_id)
                }
            previous = health.read(job_id) or {}
            monitor = LabelHealthMonitor(
                total=len(all_images),
                class_names={item.category.category_id: item.category.name for item in label_classes},
                acknowledged=set(previous.get("acknowledged", [])),
            )
            if skip and previous:
                monitor.processed = min(len(skip), int(previous.get("processed", 0)))
                monitor.resumed_from = monitor.processed
                monitor.empty_images = int(previous.get("empty_images", 0))
                for entry in previous.get("classes", []):
                    for stats in monitor.classes.values():
                        if stats.name == entry.get("name"):
                            stats.detections = int(entry.get("detections", 0))
                            stats.images_with = int(entry.get("images_with", 0))
                            mean = entry.get("mean_score")
                            stats.score_sum = float(mean or 0) * stats.detections
            health.write(job_id, monitor.snapshot(status="running"))

        def save_image_annotations(image_id, dataset_image, image_annotations) -> None:
            current = self.dependencies.jobs.get(job_id)
            if current.cancel_requested:
                raise JobCancelled(f"job {job_id} cancellation requested")
            written = set()
            for annotation in image_annotations:
                self.dependencies.annotations.save(
                    tenant_id=tenant_id,
                    annotation=annotation,
                    author_id="sam3-batch",
                    expected_revision=None,
                )
                written.add(annotation.annotation_id)
            # A retried job (expired lease) may have written more annotations
            # for this image on an earlier attempt; retire the leftovers.
            stale = live_in_set(
                self.dependencies.annotations.list_latest(
                    tenant_id=tenant_id, image_ids=[image_id]
                ),
                set_id,
            )
            for revision in stale:
                if revision.annotation.annotation_id in written:
                    continue
                self.dependencies.annotations.save(
                    tenant_id=tenant_id,
                    annotation=replace(revision.annotation, deleted=True),
                    author_id="sam3-batch",
                    expected_revision=None,
                )
            if monitor is not None:
                found: dict[int, list[float]] = {category_id: [] for category_id in monitor.classes}
                for annotation in image_annotations:
                    found.setdefault(annotation.category_id, []).append(float(annotation.score or 0.0))
                monitor.record(found)
                alert = monitor.check()
                self.dependencies.health.write(  # type: ignore[union-attr]
                    job_id, monitor.snapshot(status="paused" if alert else "running", alert=alert)
                )
                if alert is not None:
                    # Publish the preview event first so a resumed run skips
                    # this image (its annotations are already saved).
                    if self.dependencies.preview is not None:
                        self.dependencies.preview.publish(
                            tenant_id=tenant_id, job_id=job_id, image_id=image_id,
                            processed=monitor.processed, total=monitor.total,
                        )
                    raise JobPaused(alert)

        def checkpoint(image_id: str, processed: int, total: int) -> None:
            current = self.dependencies.jobs.get(job_id)
            if current.cancel_requested:
                raise JobCancelled(f"job {job_id} cancellation requested")
            self.dependencies.jobs.heartbeat(
                job_id, worker_id=worker_id, lease_seconds=lease_seconds
            )
            if self.dependencies.preview is not None:
                self.dependencies.preview.publish(
                    tenant_id=tenant_id, job_id=job_id, image_id=image_id,
                    processed=processed, total=total,
                )
            logger.debug(
                "label workflow progress",
                extra={
                    "worker_id": worker_id,
                    "job_id": str(job_id),
                    "image_id": image_id,
                    "processed": processed,
                    "total": total,
                },
            )

        try:
            if images is None:
                if image_loader is None:
                    raise ValueError("images or image_loader is required")
                images = image_loader()
            if not images:
                raise ValueError("at least one label image is required")
            label_classes = classes or [
                LabelClass(prompt=prompt, threshold=threshold, category=category)  # type: ignore[arg-type]
            ]
            start_monitor(images, label_classes)
            result = self.dependencies.labeler.label(
                images, prompt=prompt, threshold=threshold, category=category, task=task,
                classes=classes,
                on_progress=checkpoint,
                on_image=save_image_annotations,
                annotation_namespace=set_id,
                keep_raw=False,
                skip_image_ids=skip,
            )
            self.dependencies.jobs.transition(job.job_id, "succeeded")
            if monitor is not None:
                self.dependencies.health.write(job_id, monitor.snapshot(status="done"))  # type: ignore[union-attr]
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
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            logger.info(
                "label workflow completed",
                extra={
                    "worker_id": worker_id,
                    "job_id": str(job_id),
                    "status": "succeeded",
                    "annotation_count": len(result.dataset.annotations),
                    "duration_ms": duration_ms,
                },
            )
            return result
        except JobPaused as exc:
            self.dependencies.jobs.transition(job.job_id, "paused", error=str(exc))
            self.dependencies.outbox.enqueue(
                EventEnvelope(
                    event_type="LabelRunPaused",
                    tenant_id=tenant_id,
                    aggregate_type="job",
                    aggregate_id=job_id,
                    payload={"alert": exc.alert},
                    idempotency_key=f"label-paused:{job_id}:{exc.alert.get('code')}",
                )
            )
            logger.info(
                "label workflow paused by the health monitor",
                extra={"worker_id": worker_id, "job_id": str(job_id), "alert": exc.alert.get("code")},
            )
            raise
        except JobCancelled as exc:
            self.dependencies.jobs.transition(job.job_id, "cancelled", error=str(exc))
            self.dependencies.outbox.enqueue(
                EventEnvelope(
                    event_type="LabelRunCancelled",
                    tenant_id=tenant_id,
                    aggregate_type="job",
                    aggregate_id=job_id,
                    payload={"reason": str(exc)},
                    idempotency_key=f"label-cancelled:{job_id}",
                )
            )
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            logger.info(
                "label workflow cancelled",
                extra={
                    "worker_id": worker_id,
                    "job_id": str(job_id),
                    "status": "cancelled",
                    "duration_ms": duration_ms,
                },
            )
            raise
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
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            logger.exception(
                "label workflow failed",
                extra={
                    "worker_id": worker_id,
                    "job_id": str(job_id),
                    "status": "failed",
                    "duration_ms": duration_ms,
                },
            )
            raise
