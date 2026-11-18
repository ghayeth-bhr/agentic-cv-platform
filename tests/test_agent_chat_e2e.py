import hashlib
import io
import time
import zipfile

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from pycocotools.coco import COCO

from agentic_cv.agent import OllamaTaskPlanner, RulesTaskPlanner
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


class SlowSam3(FakeSam3):
    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        time.sleep(0.3)
        return super().predict(image, prompt, threshold=threshold, mask_threshold=mask_threshold)


class NoPredictionSam3:
    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        return []


class FailingPlannerClient:
    def plan_task(self, message):
        raise ConnectionError("ollama is unreachable")


def make_app(tmp_path, provider, *, start_workers=True, timeout=120, planner=None):
    return create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            sam3_provider=provider,
            agent_planner=planner or RulesTaskPlanner(),
            calibration_timeout_sec=timeout,
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


def create_session(client, asset_id, message="Create a segmentation dataset of workers."):
    response = client.post(
        "/v1/agent/sessions",
        headers=headers(),
        json={"asset_id": asset_id, "message": message},
    )
    assert response.status_code == 200, response.text
    return response.json()


def wait_for_session(client, session_id, *, statuses=("complete",), timeout=15):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = client.get(f"/v1/agent/sessions/{session_id}", headers=headers())
        last = response
        if response.status_code == 200 and response.json()["status"] in statuses:
            return response.json()
        time.sleep(0.05)
    raise AssertionError(
        f"timed out waiting for agent session: {last.text if last else 'no response'}"
    )


def test_agent_chat_to_downloadable_coco_dataset(tmp_path):
    with TestClient(
        make_app(tmp_path, FakeSam3()), raise_server_exceptions=False
    ) as client:
        asset_id = upload_image(client)
        session = create_session(client, asset_id)

        assert session["status"] == "awaiting_confirmation"
        assert session["planner_mode"] == "rules"
        assert session["prompt"] == "workers"
        assert session["spec"]["task_type"] == "segmentation"
        assert session["download_path"] is None

        confirmed = client.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm",
            headers=headers(),
            json={},
        )
        assert confirmed.status_code == 200, confirmed.text
        assert confirmed.json()["status"] == "labeling"
        assert confirmed.json()["threshold"] == 0.3

        completed = wait_for_session(client, session["session_id"])
        assert completed["status"] == "complete"
        assert completed["label_status"] == "succeeded"
        assert completed["export_status"] == "succeeded"
        assert completed["dataset"]["image_count"] == 1
        assert completed["dataset"]["annotation_count"] == 1
        assert completed["dataset"]["categories"][0]["name"] == "workers"
        assert completed["download_path"].endswith("/download")

        download = client.get(completed["download_path"], headers=headers())
        assert download.status_code == 200, download.text
        assert download.headers["content-type"] == "application/zip"
        target = tmp_path / "download"
        with zipfile.ZipFile(io.BytesIO(download.content)) as bundle:
            bundle.extractall(target)
        coco = COCO(str(target / "annotations.json"))
        assert len(coco.imgs) == 1
        assert len(coco.anns) == 1

        listed = client.get("/v1/agent/sessions", headers=headers())
        assert listed.status_code == 200
        assert listed.json()["sessions"][0]["session_id"] == session["session_id"]


def test_agent_reports_calibration_timeout_and_marks_session_failed(tmp_path):
    with TestClient(
        make_app(tmp_path, SlowSam3(), timeout=0.02), raise_server_exceptions=False
    ) as client:
        asset_id = upload_image(client)
        session = create_session(client, asset_id)

        confirmed = client.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm",
            headers=headers(),
            json={},
        )
        assert confirmed.status_code == 504, confirmed.text
        assert "Calibration timed out" in confirmed.json()["detail"]

        detail = client.get(
            f"/v1/agent/sessions/{session['session_id']}", headers=headers()
        )
        assert detail.status_code == 200
        body = detail.json()
        assert body["status"] == "failed"
        assert "Calibration timed out" in body["error"]


def test_agent_cancel_cancels_queued_label_job(tmp_path):
    with TestClient(
        make_app(tmp_path, FakeSam3(), start_workers=False),
        raise_server_exceptions=False,
    ) as client:
        asset_id = upload_image(client)
        session = create_session(client, asset_id)

        confirmed = client.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm",
            headers=headers(),
            json={"threshold": 0.5},
        )
        assert confirmed.status_code == 200, confirmed.text
        assert confirmed.json()["status"] == "labeling"
        assert confirmed.json()["threshold"] == 0.5

        cancelled = client.post(
            f"/v1/agent/sessions/{session['session_id']}/cancel", headers=headers()
        )
        assert cancelled.status_code == 200, cancelled.text
        body = cancelled.json()
        assert body["status"] == "cancelled"
        assert body["label_status"] == "cancelled"


def test_agent_allows_fixed_threshold_when_calibration_does_not_agree(tmp_path):
    with TestClient(
        make_app(tmp_path, NoPredictionSam3(), start_workers=False),
        raise_server_exceptions=False,
    ) as client:
        asset_id = upload_image(client)
        session = create_session(client, asset_id)

        calibrated = client.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm",
            headers=headers(),
            json={},
        )
        assert calibrated.status_code == 200, calibrated.text
        body = calibrated.json()
        assert body["status"] == "needs_attention"
        assert body["threshold"] is None
        assert body["calibration"]["status"] == "no_acceptable_threshold"
        assert "fixed threshold" in body["error"]

        manual = client.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm",
            headers=headers(),
            json={"threshold": 0.5},
        )
        assert manual.status_code == 200, manual.text
        body = manual.json()
        assert body["status"] == "labeling"
        assert body["threshold"] == 0.5
        assert body["calibration"]["status"] == "manual"


def test_agent_degrades_to_rules_when_ollama_is_down(tmp_path):
    planner = OllamaTaskPlanner(client=FailingPlannerClient())
    with TestClient(
        make_app(tmp_path, FakeSam3(), planner=planner, start_workers=False),
        raise_server_exceptions=False,
    ) as client:
        asset_id = upload_image(client)
        session = create_session(client, asset_id)

        assert session["status"] == "awaiting_confirmation"
        assert session["planner_mode"] == "rules"
        assert "local rule-based planner" in session["planner_note"]
        assert "local rule-based planner" in session["messages"][-1]["content"]
        assert session["spec"]["target_objects"] == ["workers"]
