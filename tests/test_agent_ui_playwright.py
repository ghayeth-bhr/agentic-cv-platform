"""Browser-level agent chat: plan, calibrate, label, package, and download."""

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


def test_agent_chat_download_and_session_restore_in_browser(tmp_path):
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
            agent_planner=RulesTaskPlanner(),
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

            # Upload one asset through the shipped UI and wait for indexing.
            page.locator("#upload-toggle").click()
            page.locator("#asset-name").fill("Site workers")
            page.locator("#asset-file").set_input_files(str(image_path))
            page.locator("#upload-button").click()
            expect(page.locator("#upload-status")).to_contain_text(
                "indexed for search", timeout=30_000
            )

            # Open the Agent, plan a task, and confirm the extracted spec.
            page.locator(".nav-item[data-view='agent']").click()
            expect(page.locator("#view-title")).to_have_text("Agent")
            expect(page.locator("#agent-asset")).to_have_value(
                page.locator("#agent-asset option").first.get_attribute("value")
            )
            page.locator("#agent-message").fill(
                "Create a segmentation dataset of workers wearing yellow vests."
            )
            page.locator("#agent-plan").click()

            expect(page.locator("#agent-run-pill")).to_have_text(
                "Awaiting confirmation", timeout=10_000
            )
            expect(page.locator("#agent-spec")).to_contain_text("segmentation")
            expect(page.locator("#agent-spec")).to_contain_text(
                "workers wearing yellow vests"
            )
            expect(page.locator("#agent-confirm")).to_be_visible()

            # Confirm: calibrate, label in the workers, build, and export.
            page.locator("#agent-confirm").click()
            expect(page.locator("#agent-run-pill")).to_have_text(
                "Dataset ready", timeout=30_000
            )
            expect(page.locator("#agent-messages")).to_contain_text("Calibrated at")
            expect(page.locator("#agent-download")).to_be_visible()
            expect(page.locator("#agent-progress")).to_contain_text(
                "COCO archive ready to download"
            )

            with page.expect_download(timeout=15_000) as download_info:
                page.locator("#agent-download").click()
            download = download_info.value
            assert download.suggested_filename.endswith(".zip")
            archive = tmp_path / "agent-download.zip"
            download.save_as(archive)
            with zipfile.ZipFile(archive) as bundle:
                assert "annotations.json" in bundle.namelist()
                assert any(
                    name.startswith("images/") for name in bundle.namelist()
                )
                extracted = tmp_path / "agent-dataset"
                bundle.extractall(extracted)
            coco = COCO(str(extracted / "annotations.json"))
            assert len(coco.imgs) == 1
            assert len(coco.anns) == 1

            # Reloading restores the finished session from local storage.
            page.reload(wait_until="domcontentloaded")
            page.locator(".nav-item[data-view='agent']").click()
            expect(page.locator("#agent-run-pill")).to_have_text(
                "Dataset ready", timeout=20_000
            )
            expect(page.locator("#agent-download")).to_be_visible()

            body_text = page.locator("body").inner_text()
            assert _uuid_pattern().search(body_text) is None, body_text
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth + 1"
            )
            browser.close()
