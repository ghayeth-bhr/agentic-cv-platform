"""API contracts for gallery overlays, pagination, and live preview images."""

from __future__ import annotations

import hashlib
import io
import json
import time

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from agentic_cv.application import LocalAppConfig, create_local_app
from agentic_cv.sam3 import Sam3Prediction

TENANT_ID = "00000000-0000-0000-0000-000000000001"


class FakeSam3:
    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        width, height = image.size
        return [
            Sam3Prediction(
                score=0.9,
                box=(2, 2, width - 2, height - 2),
                mask=np.ones((height, width), dtype=bool),
            )
        ]


def headers():
    return {"X-Tenant-ID": TENANT_ID}


def upload_image(client, name="frame.png"):
    asset = client.post("/v1/assets", headers=headers(), json={"name": "Gallery source"})
    assert asset.status_code == 201, asset.text
    asset_id = asset.json()["asset_id"]
    buffer = io.BytesIO()
    Image.new("RGB", (24, 16), (30, 90, 150)).save(buffer, format="PNG")
    payload = buffer.getvalue()
    upload = client.post(
        f"/v1/assets/{asset_id}/uploads",
        headers=headers(),
        json={
            "filename": name,
            "media_type": "image/png",
            "expected_sha256": hashlib.sha256(payload).hexdigest(),
        },
    )
    assert upload.status_code == 201, upload.text
    upload_id = upload.json()["upload_id"]
    part = client.put(
        f"/v1/uploads/{upload_id}/parts/1",
        files={"part": (name, io.BytesIO(payload), "image/png")},
    )
    assert part.status_code == 200, part.text
    completed = client.post(f"/v1/uploads/{upload_id}/complete")
    assert completed.status_code == 200, completed.text
    return asset_id


def wait_for_job(client, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/v1/label-jobs/{job_id}", headers=headers())
        if response.status_code == 200 and response.json()["status"] == "succeeded":
            return response.json()
        time.sleep(0.05)
    raise AssertionError("label job did not succeed")


def test_gallery_and_live_preview_endpoints(tmp_path):
    app = create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            sam3_provider=FakeSam3(),
            start_workers=True,
            worker_poll_interval=0.05,
        )
    )
    with TestClient(app) as client:
        asset_id = upload_image(client)
        submitted = client.post(
            "/v1/label-jobs",
            headers=headers(),
            json={
                "asset_id": asset_id,
                "prompt": "worker",
                "threshold": 0.5,
                "task": "segmentation",
                "category_id": 1,
                "category_name": "worker",
                "idempotency_key": "gallery-live-1",
            },
        )
        assert submitted.status_code == 202, submitted.text
        job_id = submitted.json()["job_id"]
        wait_for_job(client, job_id)

        stream = client.get(
            f"/v1/jobs/{job_id}/preview?tenant_id={TENANT_ID}&after_event_id=0"
        )
        assert stream.status_code == 200
        assert stream.headers["content-type"].startswith("text/event-stream")
        frames = [
            json.loads(line.removeprefix("data: "))
            for line in stream.text.splitlines()
            if line.startswith("data: ")
        ]
        assert frames and frames[0]["total"] == 1
        image_id = frames[0]["image_id"]

        overlay = client.get(
            f"/v1/label-jobs/{job_id}/preview/{image_id}/image?tenant_id={TENANT_ID}"
        )
        assert overlay.status_code == 200
        assert overlay.headers["content-type"] == "image/png"
        with Image.open(io.BytesIO(overlay.content)) as rendered:
            assert rendered.size == (24, 16)

        created = client.post(
            "/v1/datasets",
            headers=headers(),
            json={
                "job_id": job_id,
                "name": "Gallery source — workers",
                "idempotency_key": "gallery-dataset-1",
            },
        )
        assert created.status_code == 201, created.text
        version_id = created.json()["version_id"]

        summary = client.get("/v1/datasets", headers=headers()).json()[0]
        assert summary["source_asset"] == "Gallery source"
        assert summary["status"] == "ready"
        assert summary["class_counts"] == [
            {"category_id": 1, "name": "worker", "count": 1, "color": "#40a0ff"}
        ]

        page = client.get(
            f"/v1/datasets/{version_id}/images?limit=1&offset=0", headers=headers()
        ).json()
        assert page["total"] == 1
        assert page["limit"] == 1
        assert page["has_more"] is False
        assert page["images"][0]["split"] in {"train", "validation", "test"}
        assert page["images"][0]["file_size"] > 0
        assert page["images"][0]["annotations"][0]["category_id"] == 1
