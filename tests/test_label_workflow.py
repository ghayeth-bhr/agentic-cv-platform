from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from agentic_cv.annotation_store import AnnotationRevisionStore
from agentic_cv.batch_labeling import LabelImage, Sam3BatchLabeler
from agentic_cv.dataset import DatasetCategory
from agentic_cv.job_store import JobStore
from agentic_cv.label_workflow import DurableLabelWorkflow, LabelWorkflowDependencies
from agentic_cv.outbox import Outbox
from agentic_cv.sam3 import Sam3Prediction


class FakeProvider:
    def __init__(self, fail=False):
        self.fail = fail

    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        if self.fail:
            raise RuntimeError("SAM3 unavailable")
        return [Sam3Prediction(score=0.9, box=(0, 0, 2, 2), mask=np.ones((2, 2), dtype=bool))]


def make_workflow(tmp_path, *, fail=False):
    jobs = JobStore(str(tmp_path / "jobs.sqlite3"))
    annotations = AnnotationRevisionStore(str(tmp_path / "annotations.sqlite3"))
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    labeler = Sam3BatchLabeler(FakeProvider(fail=fail))
    workflow = DurableLabelWorkflow(LabelWorkflowDependencies(
        jobs=jobs, annotations=annotations, outbox=outbox, labeler=labeler
    ))
    return workflow, jobs, annotations, outbox


def image(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), (255, 255, 255)).save(path)
    return [LabelImage(image_id="frame-1", path=path)]


def test_successful_workflow_persists_annotations_and_event(tmp_path):
    workflow, jobs, annotations, outbox = make_workflow(tmp_path)
    tenant_id = uuid4()
    job = workflow.submit(
        tenant_id=tenant_id, asset_id="asset-1", prompt="bear", threshold=0.5,
        task="detection", category_id=1, category_name="bear", idempotency_key="run-1",
    )

    result = workflow.execute(
        job.job_id, images=image(tmp_path), tenant_id=tenant_id, prompt="bear",
        threshold=0.5, task="detection", category=DatasetCategory(category_id=1, name="bear"),
    )

    assert len(result.dataset.annotations) == 1
    assert jobs.get(job.job_id).status == "succeeded"
    assert annotations.latest(tenant_id=tenant_id, image_id="frame-1", annotation_id="frame-1:0").revision == 1
    assert outbox.pending()[0].event_type == "LabelRunCompleted"
    jobs.close(); annotations.close(); outbox.close()


def test_failed_workflow_persists_error_and_failure_event(tmp_path):
    workflow, jobs, annotations, outbox = make_workflow(tmp_path, fail=True)
    tenant_id = uuid4()
    job = workflow.submit(
        tenant_id=tenant_id, asset_id="asset-1", prompt="bear", threshold=0.5,
        task="detection", category_id=1, category_name="bear", idempotency_key="run-1",
    )

    with pytest.raises(RuntimeError, match="unavailable"):
        workflow.execute(
            job.job_id, images=image(tmp_path), tenant_id=tenant_id, prompt="bear",
            threshold=0.5, task="detection", category=DatasetCategory(category_id=1, name="bear"),
        )

    assert jobs.get(job.job_id).status == "failed"
    assert outbox.pending()[0].event_type == "LabelRunFailed"
    jobs.close(); annotations.close(); outbox.close()


def test_workflow_rejects_wrong_tenant_before_running(tmp_path):
    workflow, jobs, annotations, outbox = make_workflow(tmp_path)
    owner = uuid4()
    job = workflow.submit(
        tenant_id=owner, asset_id="asset-1", prompt="bear", threshold=0.5,
        task="detection", category_id=1, category_name="bear", idempotency_key="run-1",
    )

    with pytest.raises(PermissionError, match="tenant"):
        workflow.execute(
            job.job_id, images=image(tmp_path), tenant_id=uuid4(), prompt="bear",
            threshold=0.5, task="detection", category=DatasetCategory(category_id=1, name="bear"),
        )

    assert jobs.get(job.job_id).status == "queued"
    jobs.close(); annotations.close(); outbox.close()
