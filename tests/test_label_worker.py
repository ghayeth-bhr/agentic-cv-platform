import hashlib
import time
from uuid import uuid4

import numpy as np
from PIL import Image

from agentic_cv.annotation_store import AnnotationRevisionStore
from agentic_cv.asset_registry import AssetRegistry
from agentic_cv.batch_labeling import Sam3BatchLabeler
from agentic_cv.job_store import JobStore
from agentic_cv.label_images import LabelImageResolver
from agentic_cv.label_worker import LabelWorker
from agentic_cv.label_workflow import DurableLabelWorkflow, LabelWorkflowDependencies
from agentic_cv.outbox import Outbox
from agentic_cv.preview import PreviewStore
from agentic_cv.sam3 import Sam3Prediction


class FakeProvider:
    def __init__(self, fail=False):
        self.fail = fail

    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        if self.fail:
            raise RuntimeError("SAM3 unavailable")
        return [
            Sam3Prediction(
                score=0.9,
                box=(1, 1, 5, 5),
                mask=np.ones((6, 6), dtype=bool),
            )
        ]


def make_runtime(tmp_path, *, fail=False):
    registry = AssetRegistry(str(tmp_path / "assets.sqlite3"))
    annotations = AnnotationRevisionStore(str(tmp_path / "annotations.sqlite3"))
    jobs = JobStore(str(tmp_path / "jobs.sqlite3"))
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    preview = PreviewStore(str(tmp_path / "preview.sqlite3"))
    workflow = DurableLabelWorkflow(
        LabelWorkflowDependencies(
            jobs=jobs,
            annotations=annotations,
            outbox=outbox,
            labeler=Sam3BatchLabeler(FakeProvider(fail=fail)),
            preview=preview,
        )
    )
    resolver = LabelImageResolver(registry=registry, data_dir=tmp_path / "data")
    worker = LabelWorker(jobs=jobs, workflow=workflow, images=resolver)
    return registry, annotations, jobs, outbox, preview, workflow, worker


def add_asset_image(registry, tmp_path, tenant_id):
    asset = registry.create_asset(tenant_id=tenant_id, name="photos")
    path = tmp_path / "frame.png"
    Image.new("RGB", (6, 6), (255, 255, 255)).save(path)
    registry.add_source(
        tenant_id=tenant_id,
        asset_id=asset.asset_id,
        filename=path.name,
        media_type="image/png",
        object_uri=path.resolve().as_uri(),
        content_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return asset


def submit(workflow, tenant_id, asset_id, key="run-1"):
    return workflow.submit(
        tenant_id=tenant_id,
        asset_id=str(asset_id),
        prompt="worker",
        threshold=0.5,
        task="detection",
        category_id=0,
        category_name="worker",
        idempotency_key=key,
    )


def test_label_worker_executes_queued_job_and_persists_annotation(tmp_path):
    registry, annotations, jobs, outbox, preview, workflow, worker = make_runtime(tmp_path)
    tenant_id = uuid4()
    asset = add_asset_image(registry, tmp_path, tenant_id)
    job = submit(workflow, tenant_id, asset.asset_id)

    inspected = worker.run_once(worker_id="worker-a")

    assert inspected == (job.job_id,)
    assert jobs.get(job.job_id).status == "succeeded"
    latest = annotations.list_latest(tenant_id=tenant_id)
    assert len(latest) == 1
    assert latest[0].annotation.category_id == 0
    registry.close(); annotations.close(); jobs.close(); outbox.close(); preview.close()


def test_label_worker_requeues_expired_lease_after_crash(tmp_path):
    registry, annotations, jobs, outbox, preview, workflow, worker = make_runtime(tmp_path)
    tenant_id = uuid4()
    asset = add_asset_image(registry, tmp_path, tenant_id)
    job = submit(workflow, tenant_id, asset.asset_id)
    jobs.claim(job.job_id, worker_id="crashed-worker", lease_seconds=0.01)
    time.sleep(0.02)

    worker.run_once(worker_id="recovery-worker")

    recovered = jobs.get(job.job_id)
    assert recovered.status == "succeeded"
    assert len(annotations.list_latest(tenant_id=tenant_id)) == 1
    registry.close(); annotations.close(); jobs.close(); outbox.close(); preview.close()


def test_label_worker_leaves_pre_cancelled_job_cancelled(tmp_path):
    registry, annotations, jobs, outbox, preview, workflow, worker = make_runtime(tmp_path)
    tenant_id = uuid4()
    asset = add_asset_image(registry, tmp_path, tenant_id)
    job = submit(workflow, tenant_id, asset.asset_id)
    jobs.request_cancel(job.job_id)

    worker.run_once()

    assert jobs.get(job.job_id).status == "cancelled"
    assert annotations.list_latest(tenant_id=tenant_id) == []
    registry.close(); annotations.close(); jobs.close(); outbox.close(); preview.close()


def test_label_worker_marks_unresolvable_asset_failed(tmp_path):
    registry, annotations, jobs, outbox, preview, workflow, worker = make_runtime(tmp_path)
    tenant_id = uuid4()
    asset = registry.create_asset(tenant_id=tenant_id, name="empty")
    job = submit(workflow, tenant_id, asset.asset_id)

    worker.run_once()

    failed = jobs.get(job.job_id)
    assert failed.status == "failed"
    assert "no uploaded sources" in (failed.error or "")
    registry.close(); annotations.close(); jobs.close(); outbox.close(); preview.close()
