"""Failure-injection tests for resilience of jobs, workers, and outbox delivery."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from agentic_cv.annotation_store import AnnotationRevisionStore
from agentic_cv.batch_labeling import LabelImage, Sam3BatchLabeler
from agentic_cv.dataset import DatasetCategory, DatasetImage, DatasetVersion
from agentic_cv.contracts import EventEnvelope
from agentic_cv.dataset_export import DatasetExportService
from agentic_cv.dataset_export_workflow import DatasetExportWorkflow, DatasetExportWorkflowDependencies
from agentic_cv.dataset_export_worker import DatasetExportWorker
from agentic_cv.dataset_store import DatasetVersionStore
from agentic_cv.job_store import JobStore
from agentic_cv.label_workflow import DurableLabelWorkflow, LabelWorkflowDependencies
from agentic_cv.outbox import Outbox
from agentic_cv.outbox_dispatcher import OutboxDispatcher
from agentic_cv.preview import PreviewStore
from agentic_cv.sam3 import Sam3Prediction


class FlakyPublisher:
    def __init__(self, fail_event_ids):
        self.fail_event_ids = set(fail_event_ids)
        self.published = []

    def publish(self, event):
        if event.event_id in self.fail_event_ids:
            raise RuntimeError("broker unavailable")
        self.published.append(event.event_id)


class FailingExports:
    def __init__(self, ok_service: DatasetExportService, fail_version_id):
        self.ok_service = ok_service
        self.fail_version_id = fail_version_id

    def export(self, *, tenant_id, version_id, format):
        if version_id == self.fail_version_id:
            raise RuntimeError("injected export failure")
        return self.ok_service.export(tenant_id=tenant_id, version_id=version_id, format=format)


class FakeProvider:
    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        return [Sam3Prediction(score=0.9, box=(0, 0, 2, 2), mask=np.ones((2, 2), dtype=bool))]


class FailingAnnotationStore:
    def __init__(self, inner: AnnotationRevisionStore):
        self.inner = inner

    def save(self, **kwargs):
        raise RuntimeError("annotation store unavailable")

    def latest(self, **kwargs):
        return self.inner.latest(**kwargs)

    def close(self):
        self.inner.close()


def _expire_lease(db_path: str, job_id) -> None:
    expired = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at = ?, updated_at = ? WHERE job_id = ?",
            (expired, expired, str(job_id)),
        )
        connection.commit()


def test_outbox_dispatcher_marks_failed_and_continues_after_publish_error(tmp_path):
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    tenant_id = uuid4()
    e1 = outbox.enqueue(EventEnvelope(
        event_type="A", tenant_id=tenant_id, aggregate_type="job", aggregate_id=uuid4(),
        payload={}, idempotency_key="a",
    ))
    e2 = outbox.enqueue(EventEnvelope(
        event_type="B", tenant_id=tenant_id, aggregate_type="job", aggregate_id=uuid4(),
        payload={}, idempotency_key="b",
    ))
    publisher = FlakyPublisher(fail_event_ids={e1.event_id})
    dispatcher = OutboxDispatcher(outbox=outbox, publisher=publisher)

    processed = dispatcher.dispatch_once(retry_after_seconds=0)

    assert processed == (e1.event_id, e2.event_id)
    assert publisher.published == [e2.event_id]
    records = {r.event_id: r for r in outbox.pending(limit=10)}
    assert e1.event_id in records
    assert records[e1.event_id].attempts == 1
    assert records[e1.event_id].last_error == "broker unavailable"
    assert e2.event_id not in records
    outbox.close()


def test_export_worker_continues_queue_after_injected_export_failure(tmp_path):
    versions = DatasetVersionStore(str(tmp_path / "versions.sqlite3"))
    jobs = JobStore(str(tmp_path / "jobs.sqlite3"))
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    ok_exports = DatasetExportService(versions, tmp_path / "exports")
    tenant_id = uuid4()
    dataset = DatasetVersion(
        images=(DatasetImage(image_id="image-1", file_name="frame.jpg", width=2, height=2),),
        categories=(DatasetCategory(category_id=0, name="object"),),
        annotations=(),
    )
    bad_version = versions.create(tenant_id=tenant_id, name="bad", dataset=dataset, idempotency_key="v-bad")
    good_version = versions.create(tenant_id=tenant_id, name="good", dataset=dataset, idempotency_key="v-good")
    workflow = DatasetExportWorkflow(DatasetExportWorkflowDependencies(
        jobs=jobs,
        versions=versions,
        exports=FailingExports(ok_exports, bad_version.version_id),
        outbox=outbox,
    ))
    worker = DatasetExportWorker(jobs=jobs, workflow=workflow)
    bad_job = workflow.submit(tenant_id=tenant_id, version_id=bad_version.version_id, format="coco", idempotency_key="e-bad")
    good_job = workflow.submit(tenant_id=tenant_id, version_id=good_version.version_id, format="coco", idempotency_key="e-good")

    inspected = worker.run_once(worker_id="worker-a")

    assert set(inspected) == {bad_job.job_id, good_job.job_id}
    assert jobs.get(bad_job.job_id).status == "failed"
    assert jobs.get(good_job.job_id).status == "succeeded"
    event_types = sorted(event.event_type for event in outbox.pending(limit=10))
    assert event_types == ["DatasetExportCompleted", "DatasetExportFailed"]
    versions.close(); jobs.close(); outbox.close()


def test_export_worker_recovers_expired_lease_and_completes_job(tmp_path):
    jobs_path = str(tmp_path / "jobs.sqlite3")
    versions = DatasetVersionStore(str(tmp_path / "versions.sqlite3"))
    jobs = JobStore(jobs_path)
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    workflow = DatasetExportWorkflow(DatasetExportWorkflowDependencies(
        jobs=jobs, versions=versions,
        exports=DatasetExportService(versions, tmp_path / "exports"),
        outbox=outbox,
    ))
    worker = DatasetExportWorker(jobs=jobs, workflow=workflow)
    tenant_id = uuid4()
    dataset = DatasetVersion(
        images=(DatasetImage(image_id="image-1", file_name="frame.jpg", width=2, height=2),),
        categories=(DatasetCategory(category_id=0, name="object"),),
        annotations=(),
    )
    version = versions.create(tenant_id=tenant_id, name="v1", dataset=dataset, idempotency_key="v1")
    job = workflow.submit(tenant_id=tenant_id, version_id=version.version_id, format="coco", idempotency_key="e1")

    jobs.claim(job.job_id, worker_id="crashed-worker", lease_seconds=60)
    _expire_lease(jobs_path, job.job_id)
    inspected = worker.run_once(worker_id="recovery-worker")

    assert job.job_id in inspected
    assert jobs.get(job.job_id).status == "succeeded"
    versions.close(); jobs.close(); outbox.close()


def test_label_workflow_records_failure_when_annotation_persistence_breaks(tmp_path):
    jobs = JobStore(str(tmp_path / "jobs.sqlite3"))
    annotations = AnnotationRevisionStore(str(tmp_path / "annotations.sqlite3"))
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    preview = PreviewStore(str(tmp_path / "preview.sqlite3"))
    workflow = DurableLabelWorkflow(LabelWorkflowDependencies(
        jobs=jobs,
        annotations=FailingAnnotationStore(annotations),
        outbox=outbox,
        labeler=Sam3BatchLabeler(FakeProvider()),
        preview=preview,
    ))
    tenant_id = uuid4()
    job = workflow.submit(
        tenant_id=tenant_id, asset_id="asset-1", prompt="bear", threshold=0.5,
        task="detection", category_id=1, category_name="bear", idempotency_key="run-fail-save",
    )
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), (255, 255, 255)).save(image_path)

    with pytest.raises(RuntimeError, match="annotation store unavailable"):
        workflow.execute(
            job.job_id,
            images=[LabelImage(image_id="frame-1", path=image_path)],
            tenant_id=tenant_id,
            prompt="bear",
            threshold=0.5,
            task="detection",
            category=DatasetCategory(category_id=1, name="bear"),
        )

    assert jobs.get(job.job_id).status == "failed"
    assert outbox.pending()[0].event_type == "LabelRunFailed"
    jobs.close(); annotations.close(); outbox.close(); preview.close()
