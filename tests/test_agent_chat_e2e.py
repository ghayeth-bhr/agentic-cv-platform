import hashlib
import io
import threading
import time
import zipfile

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from pycocotools.coco import COCO

from agentic_cv.agent import OllamaTaskPlanner, RulesTaskPlanner
from agentic_cv.application import LocalAppConfig, create_local_app
from agentic_cv.calibration import CalibrationTrace
from agentic_cv.calibration_validation import (
    ValidatorSelection,
    any_prediction_selection,
)
from agentic_cv.sam3 import Sam3Prediction
from agentic_cv.vlm_validator import (
    OverlayArtifact,
    VlmCalibrationValidator,
    VlmValidator,
)


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


def make_app(
    tmp_path,
    provider,
    *,
    start_workers=True,
    timeout=120,
    planner=None,
    validator_factory=None,
):
    return create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            sam3_provider=provider,
            agent_planner=planner or RulesTaskPlanner(),
            calibration_timeout_sec=timeout,
            calibration_validator_factory=(
                validator_factory
                or (lambda tenant_id, session_id: any_prediction_selection())
            ),
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
        assert confirmed.json()["threshold"] == 0.6  # middle of the passing run (all pass)

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


class FakeVlmClient:
    def __init__(self):
        self.calls = []

    def judge(self, *, overlay_uri, prompt):
        self.calls.append((overlay_uri, prompt))
        return {
            "target_present": True,
            "fully_detected": True,
            "false_positive": False,
            "mask_tight": True,
            "confidence": 0.94,
            "reasoning": "The mask covers the worker and is tight.",
        }


def vlm_factory(tmp_path):
    client = FakeVlmClient()

    def factory(tenant_id, session_id):
        directory = (
            tmp_path
            / "data"
            / "calibration-overlays"
            / str(tenant_id)
            / str(session_id)
        )
        overlay_id = "a" * 24 + ".png"

        def build(sample, predictions):
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / overlay_id
            Image.new("RGB", (12, 10), (30, 100, 160)).save(path, format="PNG")
            return OverlayArtifact(uri=str(path), overlay_id=overlay_id)

        return ValidatorSelection(
            validator=VlmCalibrationValidator(
                VlmValidator(client), build, model_revision="fake-vlm"
            ),
            mode="vlm",
            model="fake-vlm",
        )

    return factory, client


def test_agent_uses_the_vlm_validator_and_serves_sample_overlays(tmp_path):
    factory, vlm = vlm_factory(tmp_path)
    with TestClient(
        make_app(
            tmp_path,
            FakeSam3(),
            start_workers=False,
            validator_factory=factory,
        ),
        raise_server_exceptions=False,
    ) as client:
        asset_id = upload_image(client)
        session = create_session(client, asset_id)

        confirmed = client.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm",
            headers=headers(),
            json={},
        )
        assert confirmed.status_code == 200, confirmed.text
        body = confirmed.json()
        calibration = body["calibration"]
        assert calibration["validator_mode"] == "vlm"
        assert calibration["validator_model"] == "fake-vlm"
        assert calibration["notice"] is None
        assert calibration["attempts"][0]["verdicts"][0]["accepted"] is True
        assert calibration["attempts"][0]["verdicts"][0]["confidence"] == 0.94
        assert "tight" in calibration["attempts"][0]["verdicts"][0]["reasoning"]
        overlay_id = calibration["attempts"][0]["verdicts"][0]["overlay_id"]
        assert overlay_id
        assert vlm.calls

        overlay = client.get(
            f"/v1/agent/sessions/{session['session_id']}/calibration-overlays/{overlay_id}",
            headers=headers(),
        )
        assert overlay.status_code == 200, overlay.text
        assert overlay.headers["content-type"] == "image/png"
        assert overlay.content.startswith(b"\x89PNG\r\n\x1a\n")

        missing = client.get(
            f"/v1/agent/sessions/{session['session_id']}/calibration-overlays/{'b' * 24}.png",
            headers=headers(),
        )
        assert missing.status_code == 404


def test_agent_labels_the_fallback_validator_in_chat_and_calibration(tmp_path):
    notice = (
        "Vision validation is unavailable: Ollama is not reachable. Calibration used "
        "the fallback \"at least one prediction\" check."
    )

    def factory(tenant_id, session_id):
        return any_prediction_selection(notice=notice)

    with TestClient(
        make_app(
            tmp_path,
            FakeSam3(),
            start_workers=False,
            validator_factory=factory,
        ),
        raise_server_exceptions=False,
    ) as client:
        asset_id = upload_image(client)
        session = create_session(client, asset_id)

        confirmed = client.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm",
            headers=headers(),
            json={},
        )
        assert confirmed.status_code == 200, confirmed.text
        body = confirmed.json()
        assert body["calibration"]["validator_mode"] == "any_prediction"
        assert body["calibration"]["notice"] == notice
        assert any(notice in message["content"] for message in body["messages"])


def test_agent_surfaces_a_runtime_validator_fallback_in_chat_and_calibration(tmp_path):
    class RuntimeFallbackValidator:
        def __init__(self):
            self.notice = None

        def validate_with_trace(self, sample, predictions, prompt, *, threshold=None):
            self.notice = (
                "Vision validation failed during calibration (TimeoutError: timed out). "
                "Calibration continued with the fallback \"at least one prediction\" check."
            )
            return CalibrationTrace(
                accepted=True,
                reasoning=self.notice,
            )

    validator = RuntimeFallbackValidator()

    def factory(tenant_id, session_id):
        return ValidatorSelection(
            validator=validator,
            mode="vlm",
            model="qwen3-vl:4b",
            notice_provider=lambda: validator.notice,
        )

    with TestClient(
        make_app(
            tmp_path,
            FakeSam3(),
            start_workers=False,
            validator_factory=factory,
        ),
        raise_server_exceptions=False,
    ) as client:
        asset_id = upload_image(client)
        session = create_session(client, asset_id)

        confirmed = client.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm",
            headers=headers(),
            json={},
        )
        assert confirmed.status_code == 200, confirmed.text
        body = confirmed.json()
        assert body["calibration"]["validator_mode"] == "vlm"
        assert "fallback" in body["calibration"]["notice"]
        assert any(
            "Vision validation failed during calibration" in message["content"]
            for message in body["messages"]
        )


def test_timed_out_calibration_does_not_block_the_next_run(tmp_path):
    class SlowOnceSam3(FakeSam3):
        def __init__(self):
            self.lock = threading.Lock()
            self.calls = 0

        def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
            with self.lock:
                self.calls += 1
                if self.calls == 1:
                    time.sleep(0.25)
                return super().predict(
                    image, prompt, threshold=threshold, mask_threshold=mask_threshold
                )

    provider = SlowOnceSam3()
    with TestClient(
        make_app(tmp_path, provider, timeout=0.05, start_workers=False),
        raise_server_exceptions=False,
    ) as client:
        first_asset = upload_image(client, name="first")
        first = create_session(client, first_asset)
        timed_out = client.post(
            f"/v1/agent/sessions/{first['session_id']}/confirm",
            headers=headers(),
            json={},
        )
        assert timed_out.status_code == 504, timed_out.text

        second_asset = upload_image(client, name="second")
        second = create_session(client, second_asset)
        started = time.monotonic()
        confirmed = client.post(
            f"/v1/agent/sessions/{second['session_id']}/confirm",
            headers=headers(),
            json={},
        )
        elapsed = time.monotonic() - started

        assert confirmed.status_code == 200, confirmed.text
        assert confirmed.json()["status"] == "labeling"
        assert elapsed < 0.5
        assert provider.lock.locked() is False
        assert not any(
            item.name == "agent-calibration" and item.is_alive()
            for item in threading.enumerate()
        )


def test_cancel_during_calibration_stops_the_run(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class BlockingValidator:
        def validate_with_trace(
            self, sample, predictions, prompt, *, threshold=None
        ):
            entered.set()
            release.wait(timeout=5)
            return CalibrationTrace(accepted=True, confidence=0.9, reasoning="ok")

    def factory(tenant_id, session_id):
        return ValidatorSelection(
            validator=BlockingValidator(), mode="custom", model=None
        )

    with TestClient(
        make_app(
            tmp_path,
            FakeSam3(),
            timeout=10,
            start_workers=False,
            validator_factory=factory,
        ),
        raise_server_exceptions=False,
    ) as client:
        asset_id = upload_image(client)
        session = create_session(client, asset_id)
        response_holder = {}

        def confirm():
            response_holder["response"] = client.post(
                f"/v1/agent/sessions/{session['session_id']}/confirm",
                headers=headers(),
                json={},
            )

        thread = threading.Thread(target=confirm)
        thread.start()
        assert entered.wait(timeout=5)

        cancelled = client.post(
            f"/v1/agent/sessions/{session['session_id']}/cancel", headers=headers()
        )
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["status"] == "cancelled"

        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        response = response_holder["response"]
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "cancelled"


def test_absent_target_ends_with_cant_segment_reliably(tmp_path):
    with TestClient(
        make_app(tmp_path, NoPredictionSam3(), start_workers=False),
        raise_server_exceptions=False,
    ) as client:
        asset_id = upload_image(client, name="absent")
        session = create_session(
            client, asset_id, message="Create a segmentation dataset of purple unicorns."
        )
        confirmed = client.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm",
            headers=headers(),
            json={},
        )
        assert confirmed.status_code == 200, confirmed.text
        body = confirmed.json()
        assert body["status"] == "needs_attention"
        assert "can't segment reliably" in body["messages"][-1]["content"]
        assert body["calibration"]["timings"]["calibration_sec"] >= 0


def test_validator_is_released_before_the_labeling_message(tmp_path):
    events: list[str] = []

    def factory(tenant_id, session_id):
        def release() -> bool:
            events.append("release")
            return True

        return ValidatorSelection(
            validator=lambda sample, predictions: True,
            mode="custom",
            model="release-probe",
            release=release,
        )

    with TestClient(
        make_app(tmp_path, FakeSam3(), start_workers=False, validator_factory=factory),
        raise_server_exceptions=False,
    ) as client:
        asset_id = upload_image(client)
        session = create_session(client, asset_id)
        confirmed = client.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm",
            headers=headers(),
            json={},
        )
        assert confirmed.status_code == 200, confirmed.text
        body = confirmed.json()
        assert events == ["release"]
        assert body["status"] == "labeling"
        calibration = body["calibration"]
        assert calibration["vlm_unloaded_after_calibration"] is True
        assert calibration["timings"]["vlm_unloaded"] is True
        contents = [message["content"] for message in body["messages"]]
        unload_index = next(
            index for index, text in enumerate(contents) if "Unloaded the vision validator" in text
        )
        label_index = next(
            index for index, text in enumerate(contents) if "Labeling 1 image" in text
        )
        assert unload_index < label_index
