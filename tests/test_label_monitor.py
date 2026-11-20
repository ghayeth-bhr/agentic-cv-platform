"""Phase 3: labeling health monitor, pause-and-ask, resume, shared vision pass."""

from __future__ import annotations

import hashlib
import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from agentic_cv.agent import RulesTaskPlanner
from agentic_cv.application import LocalAppConfig, create_local_app
from agentic_cv.batch_labeling import LabelClass, LabelImage, Sam3BatchLabeler
from agentic_cv.calibration_validation import any_prediction_selection
from agentic_cv.dataset import DatasetCategory
from agentic_cv.label_monitor import LabelHealthMonitor
from agentic_cv.sam3 import Sam3Prediction

from test_agent_chat_e2e import headers, wait_for_session


# --------------------------------------------------------------------------- monitor rules


def test_no_alerts_on_small_assets_or_before_enough_images():
    small = LabelHealthMonitor(total=9, class_names={1: "car"})
    for _ in range(9):
        small.record({1: []})
    assert small.check() is None

    early = LabelHealthMonitor(total=100, class_names={1: "car"})
    for _ in range(14):  # needs max(8, 15) images
        early.record({1: []})
    assert early.check() is None
    early.record({1: []})
    assert early.check()["code"] == "mostly_empty"


def test_rules_and_acknowledgement():
    monitor = LabelHealthMonitor(total=20, class_names={1: "elephant", 2: "tusk"})
    for _ in range(8):
        monitor.record({1: [0.9], 2: []})
    alert = monitor.check()
    assert alert["code"] == "class_missing:tusk"
    assert "“tusk” has not been found" in alert["message"]
    monitor.acknowledged.add(alert["code"])
    assert monitor.check() is None

    crowded = LabelHealthMonitor(total=20, class_names={1: "grass"})
    for _ in range(8):
        crowded.record({1: [0.31] * 40})
    assert crowded.check()["code"] == "crowded:grass"

    snapshot = monitor.snapshot(status="running")
    assert snapshot["processed"] == 8
    assert snapshot["classes"][0] == {"name": "elephant", "detections": 8, "images_with": 8, "mean_score": 0.9}
    assert snapshot["eta_sec"] is not None


# --------------------------------------------------------------------------- labeler


class CountingProvider:
    def __init__(self):
        self.single = 0
        self.many = 0

    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        self.single += 1
        return [Sam3Prediction(score=0.9, box=(0, 0, 2, 2), mask=np.ones((4, 4), dtype=bool))]

    def predict_many(self, image, prompts, *, mask_threshold=None):
        self.many += 1
        return [self.predict(image, prompt, threshold=threshold) for prompt, threshold in prompts]


def _images(tmp_path, count):
    images = []
    for index in range(count):
        path = tmp_path / f"f{index}.png"
        Image.new("RGB", (4, 4)).save(path)
        images.append(LabelImage(image_id=f"img-{index}", path=path))
    return images


def test_multi_class_uses_one_shared_vision_pass_per_image_and_skips_done_images(tmp_path):
    provider = CountingProvider()
    progress = []
    result = Sam3BatchLabeler(provider).label(
        _images(tmp_path, 4),
        task="detection",
        classes=[
            LabelClass(prompt="car", threshold=0.5, category=DatasetCategory(category_id=1, name="car")),
            LabelClass(prompt="bus", threshold=0.5, category=DatasetCategory(category_id=2, name="bus")),
        ],
        skip_image_ids={"img-0", "img-1"},
        on_progress=lambda image_id, done, total: progress.append((image_id, done, total)),
    )
    assert provider.many == 2  # one call per remaining image, not per class
    assert len(result.dataset.annotations) == 4
    assert progress == [("img-2", 3, 4), ("img-3", 4, 4)]


# --------------------------------------------------------------------------- agent journey


class PickySam3:
    """Finds the target only on the first image (used for calibration-free runs)."""

    def __init__(self):
        self.calls = []

    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        self.calls.append(image.getpixel((0, 0)))
        if image.getpixel((0, 0)) == (250, 250, 250):
            width, height = image.size
            return [Sam3Prediction(score=0.9, box=(1, 1, width - 1, height - 1), mask=np.ones((height, width), dtype=bool))]
        return []


def upload_frames(client, count):
    asset = client.post("/v1/assets", headers=headers(), json={"name": "Long clip"}).json()["asset_id"]
    for index in range(count):
        buffer = io.BytesIO()
        color = (250, 250, 250) if index == 0 else (10 + index, 20, 30)
        Image.new("RGB", (16, 12), color).save(buffer, format="PNG")
        payload = buffer.getvalue()
        upload = client.post(
            f"/v1/assets/{asset}/uploads", headers=headers(),
            json={"filename": f"f{index}.png", "media_type": "image/png", "expected_sha256": hashlib.sha256(payload).hexdigest()},
        ).json()["upload_id"]
        client.put(f"/v1/uploads/{upload}/parts/1", files={"part": (f"f{index}.png", io.BytesIO(payload), "image/png")})
        assert client.post(f"/v1/uploads/{upload}/complete").status_code == 200
    return asset


def make_app(tmp_path, sam3):
    return create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            sam3_provider=sam3,
            agent_planner=RulesTaskPlanner(),
            calibration_timeout_sec=30,
            calibration_validator_factory=lambda tenant_id, session_id: any_prediction_selection(),
            start_workers=True,
            worker_poll_interval=0.05,
        )
    )


def start_run(client, count=14):
    asset = upload_frames(client, count)
    session = client.post(
        "/v1/agent/sessions", headers=headers(),
        json={"asset_ids": [asset], "message": "detect forklifts"},
    ).json()
    confirmed = client.post(
        f"/v1/agent/sessions/{session['session_id']}/confirm", headers=headers(),
        json={"threshold": 0.5, "wait": True},
    ).json()
    assert confirmed["status"] == "labeling"
    return session["session_id"]


def say(client, session_id, message):
    response = client.post(
        f"/v1/agent/sessions/{session_id}/messages", headers=headers(), json={"message": message}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_monitor_pauses_asks_and_resumes_without_relabeling(tmp_path):
    sam3 = PickySam3()
    with TestClient(make_app(tmp_path, sam3), raise_server_exceptions=False) as client:
        sid = start_run(client)
        paused = wait_for_session(client, sid, statuses=("paused",))
        health = paused["health"]
        assert health["status"] == "paused"
        assert health["alert"]["code"] == "mostly_empty"
        assert health["processed"] == 8
        assert "I paused labeling to check with you" in paused["messages"][-1]["content"]
        assert paused["quick_replies"] == ["Continue labeling", "Stop"]

        unclear = say(client, sid, "hmm what does that mean")
        assert unclear["status"] == "paused"
        assert "Continue labeling" in unclear["messages"][-1]["content"]

        resumed = say(client, sid, "Continue labeling")
        assert resumed["status"] == "labeling"
        done = wait_for_session(client, sid, statuses=("complete",))
        assert done["health"]["status"] == "done"
        assert done["health"]["processed"] == 14
        assert done["health"]["acknowledged"] == ["mostly_empty"]
        assert done["dataset"]["image_count"] == 14
        assert done["dataset"]["annotation_count"] == 1
    # The resumed run skipped the 8 images already labeled.
    assert len(sam3.calls) == 14


def test_stop_cancels_a_paused_run(tmp_path):
    with TestClient(make_app(tmp_path, PickySam3()), raise_server_exceptions=False) as client:
        sid = start_run(client)
        wait_for_session(client, sid, statuses=("paused",))
        stopped = say(client, sid, "Stop")
        assert stopped["status"] == "cancelled"
        assert stopped["label_status"] == "cancelled"


def test_browser_shows_health_and_the_pause_question(tmp_path):
    playwright_api = pytest.importorskip("playwright.sync_api")
    from test_agent_ui_playwright import _serve

    expect = playwright_api.expect
    with _serve(make_app(tmp_path, PickySam3())) as base_url:
        import httpx

        with httpx.Client(base_url=base_url, timeout=30) as client:
            sid = start_run(client)
        with playwright_api.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except playwright_api.Error as exc:
                pytest.skip(f"Playwright Chromium is not installed: {exc}")
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            page.add_init_script(f"localStorage.setItem('agentic-cv-agent-session', '{sid}')")
            page.goto(base_url + "/#/agent", wait_until="domcontentloaded")
            page.locator(".nav-item[data-view='agent']").click()
            expect(page.locator("#agent-run-pill")).to_have_text("Paused — your call", timeout=30_000)
            expect(page.locator("#agent-health")).to_be_visible()
            expect(page.locator("#agent-health-alert")).to_contain_text("have no detections")
            expect(page.locator("#agent-health-metrics")).to_contain_text("8 / 14")
            chip = page.locator("#agent-quick-replies .agent-quick-reply", has_text="Continue labeling")
            expect(chip).to_be_visible()
            chip.click()
            expect(page.locator("#agent-run-pill")).to_have_text("Dataset ready", timeout=30_000)
            expect(page.locator("#agent-health-state")).to_have_text("Finished")
            expect(page.locator("#agent-health-metrics")).to_contain_text("14 / 14")
            browser.close()


def test_an_all_empty_single_class_run_raises_one_alert_not_two():
    monitor = LabelHealthMonitor(total=20, class_names={1: "forklift"})
    for _ in range(8):
        monitor.record({1: []})
    assert monitor.check()["code"] == "mostly_empty"
    monitor.acknowledged.add("mostly_empty")
    assert monitor.check() is None  # no separate "forklift is missing" alert
