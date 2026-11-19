import hashlib
import io
import time
import zipfile
from threading import Event
from uuid import UUID

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from pycocotools.coco import COCO

from agentic_cv.application import LocalAppConfig, create_local_app
from agentic_cv.sam3 import Sam3Prediction


TENANT_ID = "00000000-0000-0000-0000-000000000001"


class FakeSam3:
    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        width, height = image.size
        return [
            Sam3Prediction(
                score=0.91,
                box=(2, 2, width - 2, height - 2),
                mask=np.ones((height, width), dtype=bool),
            )
        ]


class BlockingSam3(FakeSam3):
    def __init__(self):
        self.started = Event()
        self.release = Event()

    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test did not release the blocking provider")
        return super().predict(image, prompt, threshold=threshold, mask_threshold=mask_threshold)


def make_app(tmp_path, provider, *, start_workers=True):
    return create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            sam3_provider=provider,
            start_workers=start_workers,
            worker_poll_interval=0.05,
        )
    )


def headers():
    return {"X-Tenant-ID": TENANT_ID}


def upload_image(client, *, name="workers"):
    asset = client.post("/v1/assets", headers=headers(), json={"name": name})
    assert asset.status_code == 201, asset.text
    asset_id = asset.json()["asset_id"]
    image = io.BytesIO()
    Image.new("RGB", (10, 10), (30, 90, 150)).save(image, format="PNG")
    payload = image.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    upload = client.post(
        f"/v1/assets/{asset_id}/uploads",
        headers=headers(),
        json={
            "filename": "frame.png",
            "media_type": "image/png",
            "expected_sha256": digest,
        },
    )
    assert upload.status_code == 201, upload.text
    upload_id = upload.json()["upload_id"]
    part = client.put(
        f"/v1/uploads/{upload_id}/parts/1",
        files={"part": ("frame.png", io.BytesIO(payload), "image/png")},
    )
    assert part.status_code == 200, part.text
    completed = client.post(f"/v1/uploads/{upload_id}/complete")
    assert completed.status_code == 200, completed.text
    return asset_id


def wait_for(client, path, *, statuses=("succeeded",), timeout=10):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = client.get(path, headers=headers())
        last = response
        if response.status_code == 200 and response.json()["status"] in statuses:
            return response.json()
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}: {last.text if last else 'no response'}")


def submit_label_job(client, asset_id, *, key="label-1"):
    response = client.post(
        "/v1/label-jobs",
        headers=headers(),
        json={
            "asset_id": asset_id,
            "prompt": "workers wearing yellow vests",
            "threshold": 0.5,
            "task": "detection",
            "category_id": 0,
            "category_name": "worker",
            "idempotency_key": key,
        },
    )
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def test_http_label_to_coco_and_yolo_download(tmp_path):
    with TestClient(make_app(tmp_path, FakeSam3()), raise_server_exceptions=False) as client:
        asset_id = upload_image(client)
        job_id = submit_label_job(client, asset_id)
        wait_for(client, f"/v1/label-jobs/{job_id}")

        created = client.post(
            "/v1/datasets",
            headers=headers(),
            json={
                "job_id": job_id,
                "name": "workers-v1",
                "idempotency_key": "dataset-1",
            },
        )
        assert created.status_code == 201, created.text
        dataset = created.json()
        assert dataset["image_count"] == 1
        assert dataset["annotation_count"] == 1
        version_id = dataset["version_id"]

        detail = client.get(f"/v1/datasets/{version_id}", headers=headers())
        assert detail.status_code == 200
        assert detail.json()["name"] == "workers-v1"

        coco_export = client.post(
            "/v1/dataset-exports",
            headers=headers(),
            json={
                "version_id": version_id,
                "format": "coco",
                "idempotency_key": "export-coco",
            },
        )
        assert coco_export.status_code == 202, coco_export.text
        coco_job = coco_export.json()["job_id"]
        wait_for(client, f"/v1/dataset-exports/{coco_job}")
        coco_download = client.get(f"/v1/dataset-exports/{coco_job}/download", headers=headers())
        assert coco_download.status_code == 200, coco_download.text
        assert coco_download.headers["content-type"] == "application/zip"
        coco_dir = tmp_path / "coco"
        with zipfile.ZipFile(io.BytesIO(coco_download.content)) as bundle:
            bundle.extractall(coco_dir)
        coco = COCO(str(coco_dir / "annotations.json"))
        assert len(coco.imgs) == 1
        assert len(coco.anns) == 1
        image_info = next(iter(coco.imgs.values()))
        assert (coco_dir / image_info["file_name"]).is_file()

        yolo_export = client.post(
            "/v1/dataset-exports",
            headers=headers(),
            json={
                "version_id": version_id,
                "format": "yolo-detection",
                "idempotency_key": "export-yolo",
            },
        )
        assert yolo_export.status_code == 202, yolo_export.text
        yolo_job = yolo_export.json()["job_id"]
        wait_for(client, f"/v1/dataset-exports/{yolo_job}")
        yolo_download = client.get(f"/v1/dataset-exports/{yolo_job}/download", headers=headers())
        assert yolo_download.status_code == 200, yolo_download.text
        yolo_dir = tmp_path / "yolo"
        with zipfile.ZipFile(io.BytesIO(yolo_download.content)) as bundle:
            yolo_names = set(bundle.namelist())
            bundle.extractall(yolo_dir)
        assert "data.yaml" in yolo_names
        assert any(name.startswith("images/") for name in yolo_names)
        label_name = next(name for name in yolo_names if name.startswith("labels/") and name.endswith(".txt"))
        fields = (yolo_dir / label_name).read_text().split()
        assert len(fields) == 5
        assert fields[0] == "0"
        assert all(0.0 <= float(value) <= 1.0 for value in fields[1:])


def test_http_cancel_stops_running_label_job(tmp_path):
    provider = BlockingSam3()
    with TestClient(make_app(tmp_path, provider), raise_server_exceptions=False) as client:
        asset_id = upload_image(client, name="cancel")
        job_id = submit_label_job(client, asset_id, key="label-cancel")
        assert provider.started.wait(timeout=5), "worker never started labeling"
        cancelled = client.post(f"/v1/label-jobs/{job_id}/cancel", headers=headers())
        assert cancelled.status_code == 200, cancelled.text
        provider.release.set()
        result = wait_for(client, f"/v1/label-jobs/{job_id}", statuses=("cancelled",))
        assert result["status"] == "cancelled"


def test_http_restart_recovers_expired_label_lease(tmp_path):
    provider = FakeSam3()
    first = make_app(tmp_path, provider, start_workers=False)
    with TestClient(first, raise_server_exceptions=False) as client:
        asset_id = upload_image(client, name="crash")
        job_id = submit_label_job(client, asset_id, key="label-crash")
        first.state.resources.jobs.claim(
            UUID(job_id), worker_id="crashed-worker", lease_seconds=0.01
        )
    time.sleep(0.03)

    with TestClient(make_app(tmp_path, provider), raise_server_exceptions=False) as client:
        result = wait_for(client, f"/v1/label-jobs/{job_id}")
        assert result["status"] == "succeeded"
