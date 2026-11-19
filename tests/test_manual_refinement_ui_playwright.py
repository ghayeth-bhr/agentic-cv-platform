"""Browser-level manual refinement: click, accept, rebuild, and export."""

from __future__ import annotations

import socket
import threading
import time
import zipfile
from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image
from pycocotools.coco import COCO

from agentic_cv.agent import RulesTaskPlanner
from agentic_cv.application import LocalAppConfig, create_local_app
from agentic_cv.calibration_validation import any_prediction_selection
from agentic_cv.sam3 import Sam3Prediction


class FakeEmbeddingProvider:
    def embed_image(self, image):
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_text(self, query):
        return np.array([1.0, 0.0], dtype=np.float32)


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


class FakeRefiner:
    def refine_with_prompt(self, image, points, prompt):
        width, height = image.size
        return [
            Sam3Prediction(
                score=0.88,
                box=(8, 8, width - 8, height - 8),
                mask=np.ones((height, width), dtype=bool),
            )
        ]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _serve(app):
    import uvicorn

    port = _free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="on",
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


def _uuid_pattern():
    import re

    return re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        re.IGNORECASE,
    )


def test_manual_refinement_click_accept_rebuild_and_export_in_browser(tmp_path):
    playwright_api = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright_api.sync_playwright
    expect = playwright_api.expect
    Error = playwright_api.Error

    image_path = tmp_path / "workers.png"
    Image.new("RGB", (96, 64), (30, 90, 150)).save(image_path)
    app = create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            embedding_provider=FakeEmbeddingProvider(),
            sam3_provider=FakeSam3(),
            refinement_provider=FakeRefiner(),
            agent_planner=RulesTaskPlanner(),
            calibration_validator_factory=(
                lambda tenant_id, session_id: any_prediction_selection()
            ),
            calibration_timeout_sec=30,
            start_workers=True,
            worker_poll_interval=0.05,
        )
    )

    with _serve(app) as base_url:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except Error as exc:
                pytest.skip(f"Playwright Chromium is not installed: {exc}")
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(base_url, wait_until="domcontentloaded")

            page.locator("#upload-toggle").click()
            page.locator("#asset-name").fill("Site workers")
            page.locator("#asset-file").set_input_files(str(image_path))
            page.locator("#upload-button").click()
            expect(page.locator("#upload-status")).to_contain_text(
                "indexed for search", timeout=30_000
            )

            page.locator(".nav-item[data-view='agent']").click()
            page.locator("#agent-message").fill(
                "Create a segmentation dataset of workers wearing yellow vests."
            )
            page.locator("#agent-plan").click()
            expect(page.locator("#agent-run-pill")).to_have_text(
                "Awaiting confirmation", timeout=10_000
            )
            page.locator("#agent-confirm").click()
            expect(page.locator("#agent-run-pill")).to_have_text(
                "Dataset ready", timeout=30_000
            )

            page.locator(".nav-item[data-view='datasets']").click()
            expect(page.locator(".dataset-card")).to_have_count(1)
            card = page.locator(".dataset-card").first
            card.get_by_role("button", name="Review labels").click()
            expect(card.locator(".review-panel")).to_be_visible()
            expect(card.locator(".review-image-item")).to_have_count(1)
            card.locator(".review-image-item").first.click()
            expect(card.locator(".review-image-caption")).to_contain_text(
                "existing label", timeout=15_000
            )
            expect(card.locator("[data-review-status='true']")).to_contain_text(
                "Click the object", timeout=15_000
            )

            canvas = card.locator(".review-canvas")
            before = canvas.evaluate(
                """(el) => {
                  const ctx = el.getContext('2d');
                  const data = ctx.getImageData(0, 0, el.width, el.height).data;
                  let green = 0;
                  for (let i = 0; i < data.length; i += 4) {
                    if (data[i + 1] > 120 && data[i + 2] < 110) green += 1;
                  }
                  return green;
                }"""
            )
            assert before == 0
            canvas.click(position={"x": 48, "y": 32})
            expect(card.locator("[data-review-status='true']")).to_contain_text(
                "Mask preview ready", timeout=15_000
            )
            expect(card.get_by_role("button", name="Save corrected mask")).to_be_enabled()
            after = canvas.evaluate(
                """(el) => {
                  const ctx = el.getContext('2d');
                  const data = ctx.getImageData(0, 0, el.width, el.height).data;
                  let green = 0;
                  for (let i = 0; i < data.length; i += 4) {
                    if (data[i + 1] > 120 && data[i + 2] < 110) green += 1;
                  }
                  return green;
                }"""
            )
            assert after > 0, "the accepted mask overlay did not render"

            card.get_by_role("button", name="Save corrected mask").click()
            expect(page.locator("#datasets-status")).to_contain_text(
                "Correction saved as revision", timeout=20_000
            )

            refined = page.locator(".dataset-card").filter(has_text="refined").last
            expect(refined).to_be_visible()
            with page.expect_download(timeout=20_000) as download_info:
                refined.get_by_role("button", name="Export", exact=True).click()
            download = download_info.value
            archive = tmp_path / "refined.zip"
            download.save_as(archive)
            with zipfile.ZipFile(archive) as bundle:
                assert "annotations.json" in bundle.namelist()
                extracted = tmp_path / "refined-dataset"
                bundle.extractall(extracted)
            coco = COCO(str(extracted / "annotations.json"))
            assert len(coco.imgs) == 1
            assert len(coco.anns) == 1
            annotation = next(iter(coco.anns.values()))
            assert annotation["bbox"] == [8, 8, 80, 48]

            body_text = page.locator("body").inner_text()
            assert _uuid_pattern().search(body_text) is None, body_text
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth + 1"
            )
            browser.close()
