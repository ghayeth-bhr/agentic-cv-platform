"""Phase 2b: background calibration, live progress, resume, threshold override."""

from __future__ import annotations

import threading
import time
from dataclasses import replace

from fastapi.testclient import TestClient

from agentic_cv.agent import RulesTaskPlanner
from agentic_cv.agent_store import AgentSessionStore
from agentic_cv.application import LocalAppConfig, create_local_app
from agentic_cv.calibration_validation import ValidatorSelection, any_prediction_selection

from test_agent_chat_e2e import FakeSam3, NoPredictionSam3, headers, upload_image, wait_for_session


def make_app(tmp_path, *, sam3=None, validator_factory=None, start_workers=True):
    return create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            sam3_provider=sam3 or FakeSam3(),
            agent_planner=RulesTaskPlanner(),
            calibration_timeout_sec=60,
            calibration_validator_factory=validator_factory
            or (lambda tenant_id, session_id: any_prediction_selection()),
            start_workers=start_workers,
            worker_poll_interval=0.05,
        )
    )


def plan(client, message="Create a segmentation dataset of workers."):
    asset_id = upload_image(client)
    response = client.post(
        "/v1/agent/sessions", headers=headers(), json={"asset_ids": [asset_id], "message": message}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_confirm_returns_immediately_and_publishes_progress(tmp_path):
    gate = threading.Event()
    seen = []

    def validator(sample, predictions):
        seen.append(len(predictions))
        if len(seen) > 1:  # hold the second threshold until the test has looked
            assert gate.wait(timeout=10)
        return bool(predictions)

    factory = lambda tenant_id, session_id: ValidatorSelection(validator=validator, mode="custom")
    with TestClient(make_app(tmp_path, validator_factory=factory), raise_server_exceptions=False) as client:
        session = plan(client)
        started = time.monotonic()
        confirmed = client.post(f"/v1/agent/sessions/{session['session_id']}/confirm", headers=headers(), json={})
        assert confirmed.status_code == 200, confirmed.text
        assert confirmed.json()["status"] == "calibrating"
        assert time.monotonic() - started < 2

        live = wait_for_session(client, session["session_id"], statuses=("calibrating",))
        deadline = time.monotonic() + 10
        while not (live.get("calibration") or {}).get("progress") and time.monotonic() < deadline:
            time.sleep(0.05)
            live = client.get(f"/v1/agent/sessions/{session['session_id']}", headers=headers()).json()
        progress = live["calibration"]["progress"]
        assert live["status"] == "calibrating"
        assert live["calibration"]["status"] == "running"
        assert progress[0]["name"] == "workers"
        assert progress[0]["threshold"] == 0.3
        gate.set()

        done = wait_for_session(client, session["session_id"], statuses=("complete",))
        assert done["calibration"]["status"] == "calibrated"


def test_interrupted_calibration_resumes_after_restart(tmp_path):
    with TestClient(make_app(tmp_path, start_workers=False), raise_server_exceptions=False) as client:
        session = plan(client)
    # Simulate a crash mid-calibration: the session is left "calibrating".
    store = AgentSessionStore(str(tmp_path / "data" / "metadata" / "agent-sessions.sqlite3"))
    from uuid import UUID

    record = store.get_any(UUID(session["session_id"]))
    store.save(
        replace(
            record,
            status="calibrating",
            spec={**record.spec, "calibration_request": {"threshold": None, "thresholds": [0.3, 0.6], "minimum_agreement": 0.6, "class_thresholds": None}},
        )
    )
    store.close()

    with TestClient(make_app(tmp_path), raise_server_exceptions=False) as client:
        done = wait_for_session(client, session["session_id"], statuses=("complete", "failed"))
        assert done["status"] == "complete", done["messages"][-1]
        assert [attempt["threshold"] for attempt in done["calibration"]["attempts"]] == [0.3, 0.6]


def test_user_thresholds_per_class_after_calibration_needs_attention(tmp_path):
    with TestClient(make_app(tmp_path, sam3=NoPredictionSam3()), raise_server_exceptions=False) as client:
        session = plan(client, "detect helmets and vests")
        failed = client.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm", headers=headers(), json={"wait": True}
        ).json()
        assert failed["status"] == "needs_attention"

        missing = client.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm",
            headers=headers(),
            json={"class_thresholds": {"helmets": 0.4}, "wait": True},
        )
        assert missing.status_code == 422

        chosen = client.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm",
            headers=headers(),
            json={"class_thresholds": {"helmets": 0.4, "vests": 0.55}, "wait": True},
        ).json()
        assert chosen["status"] == "labeling"
        assert chosen["calibration"]["status"] == "manual"
        assert chosen["calibration"]["class_thresholds"] == {"helmets": 0.4, "vests": 0.55}
        assert chosen["calibration"]["previous"]["status"] == "no_acceptable_threshold"
        assert "Using your thresholds: helmets 0.40, vests 0.55" in chosen["messages"][-1]["content"]
        wait_for_session(client, session["session_id"], statuses=("complete",))


class TwoScoreSam3:
    """A strong and a weak mask, so a threshold slider changes what is shown."""

    def __init__(self, scores=(0.91, 0.4)):
        self.scores = scores

    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        import numpy as np

        from agentic_cv.sam3 import Sam3Prediction

        width, height = image.size
        predictions = []
        for index, score in enumerate(self.scores):
            mask = np.zeros((height, width), dtype=bool)
            if index == 0:
                mask[1 : height // 2, 1 : width // 2] = True
                box = (1, 1, width // 2, height // 2)
            else:
                mask[height // 2 : height - 1, width // 2 : width - 1] = True
                box = (width // 2, height // 2, width - 1, height - 1)
            if threshold is None or score >= threshold:
                predictions.append(Sam3Prediction(score=score, box=box, mask=mask))
        return predictions


def test_calibration_preview_and_sample_images(tmp_path):
    with TestClient(make_app(tmp_path, sam3=TwoScoreSam3()), raise_server_exceptions=False) as client:
        session = plan(client)
        sid = session["session_id"]
        assert client.get(f"/v1/agent/sessions/{sid}/calibration-preview", headers=headers()).status_code == 404
        done = client.post(f"/v1/agent/sessions/{sid}/confirm", headers=headers(), json={"wait": True}).json()
        assert done["status"] == "labeling"

        preview = client.get(f"/v1/agent/sessions/{sid}/calibration-preview", headers=headers())
        assert preview.status_code == 200, preview.text
        entry = preview.json()["classes"][0]
        assert entry["name"] == "workers"
        sample = entry["samples"][0]
        # Raw predictions at the lowest threshold, with polygons for the slider.
        assert sorted(prediction["score"] for prediction in sample["predictions"]) == [0.4, 0.91]
        assert all(prediction["polygons"] for prediction in sample["predictions"])

        from urllib.parse import quote

        image = client.get(
            f"/v1/agent/sessions/{sid}/calibration-samples/{quote(sample['image_id'], safe='')}/image",
            headers=headers(),
        )
        assert image.status_code == 200
        assert image.content[:4] == b"\x89PNG"
        other = client.get(
            f"/v1/agent/sessions/{sid}/calibration-samples/not-a-sample/image", headers=headers()
        )
        assert other.status_code == 404
        foreign = client.get(
            f"/v1/agent/sessions/{sid}/calibration-preview",
            headers={"X-Tenant-ID": "00000000-0000-0000-0000-000000000009"},
        )
        assert foreign.status_code == 404


def test_browser_calibration_board_slider_and_threshold_override(tmp_path):
    import pytest

    playwright_api = pytest.importorskip("playwright.sync_api")
    from PIL import Image

    from test_agent_ui_playwright import FakeEmbeddingProvider, _serve

    expect = playwright_api.expect
    image_path = tmp_path / "yard.png"
    Image.new("RGB", (96, 64), (30, 90, 150)).save(image_path)

    def strict_validator(sample, predictions):
        # Only a single strong mask is acceptable, so nothing passes
        # (the weak mask is filtered above 0.4, the strong one above 0.91).
        return False

    app = create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            embedding_provider=FakeEmbeddingProvider(),
            sam3_provider=TwoScoreSam3(),
            agent_planner=RulesTaskPlanner(),
            calibration_timeout_sec=30,
            calibration_validator_factory=lambda tenant_id, session_id: ValidatorSelection(
                validator=strict_validator, mode="custom"
            ),
            start_workers=True,
            worker_poll_interval=0.05,
        )
    )
    with _serve(app) as base_url:
        with playwright_api.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except playwright_api.Error as exc:
                pytest.skip(f"Playwright Chromium is not installed: {exc}")
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            page.goto(base_url, wait_until="domcontentloaded")
            page.locator("#upload-toggle").click()
            page.locator("#asset-name").fill("Yard")
            page.locator("#asset-file").set_input_files(str(image_path))
            page.locator("#upload-button").click()
            expect(page.locator("#upload-status")).to_contain_text("indexed for search", timeout=30_000)

            page.locator(".nav-item[data-view='agent']").click()
            page.locator("#agent-message").fill("Segment workers")
            page.locator("#agent-plan").click()
            expect(page.locator("#agent-run-pill")).to_have_text("Awaiting confirmation", timeout=10_000)
            page.locator("#agent-confirm").click()

            expect(page.locator("#agent-run-pill")).to_have_text("Needs attention", timeout=30_000)
            expect(page.locator("#agent-calibration")).not_to_contain_text("Fallback validator")
            board = page.locator("#agent-board")
            expect(board).to_be_visible(timeout=10_000)
            expect(board.locator(".agent-board-class")).to_have_count(1)
            expect(board.locator(".agent-board-tile")).to_have_count(1)
            expect(board.locator(".agent-board-chip")).to_have_count(5)
            caption = board.locator(".agent-board-caption").first

            slider = board.locator(".agent-board-slider")
            slider.fill("0.3")
            expect(board.locator(".agent-board-value")).to_have_text("0.30")
            expect(caption).to_contain_text("2 masks")
            slider.fill("0.6")
            expect(caption).to_contain_text("1 mask")
            expect(caption).to_contain_text("✗ rejected")  # 0.60 was a tried threshold
            slider.fill("0.95")
            expect(caption).to_contain_text("0 masks")

            # Label at a user-picked threshold.
            slider.fill("0.6")
            expect(page.locator("#agent-board-actions")).to_be_visible()
            page.locator("#agent-board-apply").click()
            expect(page.locator("#agent-run-pill")).to_have_text("Dataset ready", timeout=30_000)
            expect(page.locator("#agent-messages")).to_contain_text("Using the fixed threshold 0.60")
            browser.close()


def test_shutdown_mid_calibration_leaves_the_session_resumable(tmp_path):
    started = threading.Event()

    def slow_validator(sample, predictions):
        started.set()
        time.sleep(0.2)
        return bool(predictions)

    factory = lambda tenant_id, session_id: ValidatorSelection(validator=slow_validator, mode="custom")
    with TestClient(make_app(tmp_path, validator_factory=factory, start_workers=False), raise_server_exceptions=False) as client:
        session = plan(client)
        client.post(f"/v1/agent/sessions/{session['session_id']}/confirm", headers=headers(), json={})
        assert started.wait(timeout=10)
    # The app shut down mid-calibration; the session is still resumable.
    store = AgentSessionStore(str(tmp_path / "data" / "metadata" / "agent-sessions.sqlite3"))
    from uuid import UUID

    record = store.get_any(UUID(session["session_id"]))
    store.close()
    assert record.status == "calibrating"
    with TestClient(make_app(tmp_path), raise_server_exceptions=False) as client:
        done = wait_for_session(client, session["session_id"], statuses=("complete", "failed"))
        assert done["status"] == "complete"
