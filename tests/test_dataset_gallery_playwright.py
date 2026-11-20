"""Browser journeys for the dataset gallery and the agent live view."""

from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image

from agentic_cv.agent import RulesTaskPlanner
from agentic_cv.application import LocalAppConfig, create_local_app
from agentic_cv.calibration_validation import ValidatorSelection
from agentic_cv.sam3 import Sam3Prediction
from agentic_cv.vlm_validator import OverlayArtifact, VlmCalibrationValidator, VlmValidator


class FakeEmbeddingProvider:
    def embed_image(self, image):
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_text(self, query):
        return np.array([1.0, 0.0], dtype=np.float32)


class GallerySam3:
    """Deterministic predictions with no / low / high-confidence cases."""

    def __init__(self, delay=0.0):
        self.delay = delay

    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        if self.delay:
            time.sleep(self.delay)
        red, green, _ = image.getpixel((0, 0))[:3]
        if green < 40:
            return []
        score = 0.42 if red < 100 else 0.91
        width, height = image.size
        return [
            Sam3Prediction(
                score=score,
                box=(2, 2, width - 2, height - 2),
                mask=np.ones((height, width), dtype=bool),
            )
        ]


class FakeVlmClient:
    def judge(self, *, overlay_uri, prompt):
        return {
            "target_present": True,
            "fully_detected": True,
            "false_positive": False,
            "mask_tight": True,
            "confidence": 0.95,
            "reasoning": "The sample matches the prompt.",
        }


def vlm_factory(tmp_path):
    def factory(tenant_id, session_id):
        directory = (
            tmp_path / "data" / "calibration-overlays" / str(tenant_id) / str(session_id)
        )
        overlay_id = "gallery-overlay.png"

        def build(sample, predictions):
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / overlay_id
            Image.new("RGB", (32, 24), (40, 120, 180)).save(path, format="PNG")
            return OverlayArtifact(uri=str(path), overlay_id=overlay_id)

        return ValidatorSelection(
            validator=VlmCalibrationValidator(
                VlmValidator(FakeVlmClient()), build, model_revision="fake-vlm"
            ),
            mode="vlm",
            model="fake-vlm",
        )

    return factory


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _serve(app):
    import uvicorn

    port = _free_port()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        if not thread.is_alive():
            raise RuntimeError("uvicorn server thread stopped before startup")
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("uvicorn server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _make_images(tmp_path, count):
    colors = [(20, 20, 20), (30, 90, 150), (220, 200, 80)]
    paths = []
    for index in range(count):
        path = tmp_path / f"frame-{index + 1}.png"
        image = Image.new("RGB", (96, 64), colors[index % len(colors)])
        image.putpixel((1, 1), (index, index, index))
        image.save(path)
        paths.append(str(path))
    return paths


def _build_app(tmp_path, provider):
    return create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            embedding_provider=FakeEmbeddingProvider(),
            sam3_provider=provider,
            agent_planner=RulesTaskPlanner(),
            calibration_validator_factory=vlm_factory(tmp_path),
            calibration_timeout_sec=30,
            start_workers=True,
            worker_poll_interval=0.05,
        )
    )


def _upload_and_run(page, image_paths, expect):
    page.locator("#upload-toggle").click()
    page.locator("#asset-name").fill("Gallery scene")
    page.locator("#asset-file").set_input_files(image_paths)
    page.locator("#upload-button").click()
    expect(page.locator("#upload-status")).to_contain_text(
        "indexed for search", timeout=60_000
    )

    page.locator(".nav-item[data-view='agent']").click()
    page.locator("#agent-message").fill(
        "Create a segmentation dataset of workers in high-visibility vests."
    )
    page.locator("#agent-plan").click()
    expect(page.locator("#agent-run-pill")).to_have_text(
        "Awaiting confirmation", timeout=15_000
    )
    page.locator("#agent-confirm").click()
    page.wait_for_function(
        """
        () => {
          const text = document.querySelector('#agent-run-pill')?.textContent || '';
          return text === 'Dataset ready' || text === 'Needs attention';
        }
        """,
        timeout=120_000,
    )
    if (page.locator("#agent-run-pill").text_content() == "Needs attention"):
        page.locator("#agent-retry").click()
    expect(page.locator("#agent-run-pill")).to_have_text(
        "Dataset ready", timeout=120_000
    )


def test_gallery_navigation_filters_toggles_and_deep_link(tmp_path):
    playwright_api = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright_api.sync_playwright
    expect = playwright_api.expect
    Error = playwright_api.Error

    image_paths = _make_images(tmp_path, 9)
    with _serve(_build_app(tmp_path, GallerySam3())) as base_url:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except Error as exc:
                pytest.skip(f"Playwright Chromium is not installed: {exc}")
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(base_url, wait_until="domcontentloaded")
            _upload_and_run(page, image_paths, expect)

            page.locator(".nav-item[data-view='datasets']").click()
            card = page.locator(".dataset-card").first
            expect(card).to_be_visible(timeout=15_000)
            version_id = card.get_attribute("data-version-id")
            assert version_id

            card.locator("[data-open-gallery]").click()
            expect(page.locator("#dataset-gallery")).to_be_visible()
            expect(page.locator("#gallery-counter")).to_have_text("image 1 of 9")
            assert page.evaluate("location.hash") == f"#/datasets/{version_id}"

            page.keyboard.press("ArrowRight")
            expect(page.locator("#gallery-counter")).to_have_text("image 2 of 9")
            page.locator("#gallery-next").click()
            expect(page.locator("#gallery-counter")).to_have_text("image 3 of 9")
            page.keyboard.press("ArrowLeft")
            expect(page.locator("#gallery-counter")).to_have_text("image 2 of 9")

            overlay = page.locator("#gallery-overlay")
            expect(overlay).to_have_attribute("data-boxes", "true")
            page.locator("#gallery-toggle-boxes").uncheck()
            expect(overlay).to_have_attribute("data-boxes", "false")
            page.locator("#gallery-toggle-masks").uncheck()
            expect(overlay).to_have_attribute("data-masks", "false")
            page.locator("#gallery-toggle-labels").uncheck()
            expect(overlay).to_have_attribute("data-labels", "false")

            page.locator("#gallery-filter-lowconfidence").check()
            expect(page.locator("#gallery-counter")).to_have_text("image 1 of 3")
            page.locator("#gallery-reset-filters").click()
            expect(page.locator("#gallery-counter")).to_have_text("image 1 of 9")

            page.locator("#gallery-filter-unannotated").check()
            expect(page.locator("#gallery-counter")).to_have_text("image 1 of 3")
            page.locator("#gallery-reset-filters").click()

            page.locator("#gallery-filter-class").select_option(index=1)
            expect(page.locator("#gallery-counter")).to_have_text("image 1 of 6")
            page.locator("#gallery-reset-filters").click()

            page.locator("#gallery-filter-split").select_option("train")
            train_count = page.locator("#gallery-counter").text_content()
            assert train_count.startswith("image 1 of ")
            assert train_count != "image 1 of 9"
            page.locator("#gallery-reset-filters").click()

            page.goto(f"{base_url}/#/datasets/{version_id}", wait_until="domcontentloaded")
            expect(page.locator("#dataset-gallery")).to_be_visible()
            expect(page.locator("#gallery-counter")).to_have_text("image 1 of 9")

            page.locator("#gallery-refine").click()
            expect(page.locator(".dataset-card [data-review-panel='true']")).to_be_visible()
            browser.close()


def test_live_grid_fills_during_run_and_hands_off_to_gallery(tmp_path):
    playwright_api = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright_api.sync_playwright
    expect = playwright_api.expect
    Error = playwright_api.Error

    image_paths = _make_images(tmp_path, 4)
    with _serve(_build_app(tmp_path, GallerySam3(delay=0.6))) as base_url:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except Error as exc:
                pytest.skip(f"Playwright Chromium is not installed: {exc}")
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(base_url, wait_until="domcontentloaded")
            _upload_and_run(page, image_paths, expect)

            expect(page.locator("#agent-live")).to_be_visible()
            expect(page.locator("#agent-live-badge")).to_have_text("Live")
            expect(page.locator("#agent-live-grid figure")).to_have_count(4)
            expect(page.locator("#agent-live-count")).to_have_text("4 / 4 images")
            expect(page.locator("#agent-open-gallery")).to_be_visible()

            page.locator("#agent-open-gallery").click()
            expect(page.locator("#dataset-gallery")).to_be_visible()
            expect(page.locator("#gallery-counter")).to_have_text("image 1 of 4")
            browser.close()
