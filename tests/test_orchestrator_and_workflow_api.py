import io
from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from agentic_cv.annotation_store import AnnotationRevisionStore
from agentic_cv.api import (
    ApiDependencies,
    CalibrationApiDependencies,
    LabelApiDependencies,
    create_app,
)
from agentic_cv.asset_registry import AssetRegistry
from agentic_cv.batch_labeling import Sam3BatchLabeler
from agentic_cv.job_store import JobStore
from agentic_cv.label_workflow import DurableLabelWorkflow, LabelWorkflowDependencies
from agentic_cv.object_storage import LocalObjectStorage
from agentic_cv.outbox import Outbox
from agentic_cv.preview import PreviewStore
from agentic_cv.sam3 import Sam3Prediction
from agentic_cv.uploads import ResumableUploadService


class FakeProvider:
    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        score = 0.9 if (threshold is None or threshold <= 0.5) else 0.2
        return [Sam3Prediction(score=score, box=(0.0, 0.0, 1.0, 1.0), mask=np.ones((2, 2), dtype=bool))]


def make_client(tmp_path, *, include_label=False, include_calibration=False):
    registry = AssetRegistry(str(tmp_path / "assets.sqlite3"))
    storage = LocalObjectStorage(tmp_path / "objects")
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    jobs = JobStore(str(tmp_path / "jobs.sqlite3"))
    annotations = AnnotationRevisionStore(str(tmp_path / "annotations.sqlite3"))
    preview = PreviewStore(str(tmp_path / "preview.sqlite3"))
    provider = FakeProvider()
    labels = None
    if include_label:
        workflow = DurableLabelWorkflow(LabelWorkflowDependencies(
            jobs=jobs,
            annotations=annotations,
            outbox=outbox,
            labeler=Sam3BatchLabeler(provider),
            preview=preview,
        ))
        labels = LabelApiDependencies(workflow=workflow)
    calibration = CalibrationApiDependencies(provider=provider) if include_calibration else None
    app = create_app(ApiDependencies(
        registry=registry,
        uploads=ResumableUploadService(registry, storage),
        outbox=outbox,
        labels=labels,
        calibration=calibration,
    ))
    client = TestClient(app)
    return client, registry, outbox, jobs, annotations, preview


def test_orchestrate_endpoint_returns_task_spec(tmp_path):
    client, registry, outbox, jobs, annotations, preview = make_client(tmp_path)
    response = client.post(
        "/v1/orchestrate",
        json={"message": "please segment the brown bear in each frame"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] == "segmentation"
    assert body["target_objects"]
    assert body["sample_size"] == 5
    registry.close(); outbox.close(); jobs.close(); annotations.close(); preview.close()


def test_orchestrate_endpoint_accepts_explicit_fields(tmp_path):
    client, registry, outbox, jobs, annotations, preview = make_client(tmp_path)
    response = client.post(
        "/v1/orchestrate",
        json={
            "message": "label animals",
            "task_type": "detection",
            "target_objects": ["bear", "deer"],
            "visual_attributes": ["brown"],
            "sample_size": 3,
            "max_iterations": 4,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] == "detection"
    assert body["target_objects"] == ["bear", "deer"]
    assert body["visual_attributes"] == ["brown"]
    assert body["sample_size"] == 3
    assert body["max_iterations"] == 4
    registry.close(); outbox.close(); jobs.close(); annotations.close(); preview.close()


def test_label_job_api_submits_gets_and_cancels(tmp_path):
    tenant_id = uuid4()
    client, registry, outbox, jobs, annotations, preview = make_client(tmp_path, include_label=True)
    headers = {"X-Tenant-ID": str(tenant_id)}

    submitted = client.post(
        "/v1/label-jobs",
        headers=headers,
        json={
            "asset_id": "asset-1",
            "prompt": "bear",
            "threshold": 0.5,
            "task": "detection",
            "category_id": 1,
            "category_name": "bear",
            "idempotency_key": "label-1",
        },
    )
    assert submitted.status_code == 202
    job = submitted.json()
    assert job["kind"] == "batch-label"
    assert job["status"] == "queued"
    assert job["payload"]["asset_id"] == "asset-1"

    fetched = client.get(f"/v1/label-jobs/{job['job_id']}", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json()["job_id"] == job["job_id"]

    cancelled = client.post(f"/v1/label-jobs/{job['job_id']}/cancel", headers=headers)
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    registry.close(); outbox.close(); jobs.close(); annotations.close(); preview.close()


def test_label_job_api_enforces_tenant_isolation(tmp_path):
    tenant_id = uuid4()
    client, registry, outbox, jobs, annotations, preview = make_client(tmp_path, include_label=True)
    headers = {"X-Tenant-ID": str(tenant_id)}
    job = client.post(
        "/v1/label-jobs",
        headers=headers,
        json={
            "asset_id": "asset-1",
            "prompt": "bear",
            "threshold": 0.5,
            "task": "detection",
            "category_id": 1,
            "category_name": "bear",
            "idempotency_key": "label-1",
        },
    ).json()

    other = {"X-Tenant-ID": str(uuid4())}
    assert client.get(f"/v1/label-jobs/{job['job_id']}", headers=other).status_code == 404
    assert client.post(f"/v1/label-jobs/{job['job_id']}/cancel", headers=other).status_code == 404

    registry.close(); outbox.close(); jobs.close(); annotations.close(); preview.close()


def _png_bytes(color):
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format="PNG")
    return buf.getvalue()


def test_calibration_api_runs_workflow(tmp_path):
    client, registry, outbox, jobs, annotations, preview = make_client(tmp_path, include_calibration=True)
    response = client.post(
        "/v1/calibration/run",
        data={
            "prompt": "bear",
            "thresholds": "0.3,0.6,0.8",
            "minimum_agreement": "1.0",
            "max_iterations": "5",
            "validator_mode": "has_prediction",
        },
        files=[
            ("images", ("a.png", _png_bytes((255, 0, 0)), "image/png")),
            ("images", ("b.png", _png_bytes((0, 255, 0)), "image/png")),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "calibrated"
    # Every threshold is scored now; all pass, so the middle one is chosen.
    assert body["threshold"] == 0.6
    assert body["prompt"] == "bear"
    assert len(body["attempts"]) == 3
    assert all(attempt["accepted_count"] == 2 for attempt in body["attempts"])

    registry.close(); outbox.close(); jobs.close(); annotations.close(); preview.close()


def test_calibration_api_rejects_invalid_input(tmp_path):
    client, registry, outbox, jobs, annotations, preview = make_client(tmp_path, include_calibration=True)
    response = client.post(
        "/v1/calibration/run",
        data={"prompt": "bear", "thresholds": "", "validator_mode": "has_prediction"},
        files=[("images", ("a.png", _png_bytes((255, 0, 0)), "image/png"))],
    )
    assert response.status_code == 422
    registry.close(); outbox.close(); jobs.close(); annotations.close(); preview.close()
