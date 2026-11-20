"""Browser-level assets experience: upload, auto-index, search, and sub-assets.

The test runs against a real uvicorn server on localhost with a deterministic
fake embedding provider, then drives the shipped UI in headless Chromium. It is
skipped when the Playwright browser binary is not installed so the default test
suite still runs on machines without one.
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image

from agentic_cv.application import LocalAppConfig, create_local_app


class FakeEmbeddingProvider:
    """Small deterministic image/text embedding used only by this UI test."""

    def embed_image(self, image):
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        vector = np.array([pixels[0] - pixels[1], pixels[2] - pixels[1]], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return np.array([1.0, 0.0], dtype=np.float32)
        return vector / norm

    def embed_text(self, query):
        lowered = query.lower()
        if "blue" in lowered:
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.array([1.0, 0.0], dtype=np.float32)


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


@pytest.fixture
def asset_images(tmp_path):
    red = tmp_path / "red-site.png"
    blue = tmp_path / "blue-site.bmp"
    Image.new("RGB", (96, 64), (255, 0, 0)).save(red)
    # BMP is uncompressed, so a modest image crosses the 5 MB chunk boundary
    # and exercises a second upload part without needing a huge test fixture.
    Image.new("RGB", (1600, 1200), (0, 0, 255)).save(blue, format="BMP")
    return red, blue


def _uuid_pattern():
    import re

    return re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        re.IGNORECASE,
    )


def test_assets_upload_index_search_and_subasset_in_browser(tmp_path, asset_images):
    playwright_api = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright_api.sync_playwright
    expect = playwright_api.expect
    Error = playwright_api.Error

    app = create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            embedding_provider=FakeEmbeddingProvider(),
            start_workers=True,
            worker_poll_interval=0.05,
        )
    )
    red, blue = asset_images

    with _serve(app) as base_url:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except Error as exc:
                pytest.skip(f"Playwright Chromium is not installed: {exc}")
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            upload_parts = []

            def record_request(request):
                if "/parts/" in request.url:
                    upload_parts.append(request.url)

            page.on("request", record_request)
            page.goto(base_url, wait_until="domcontentloaded")

            # The workspace loads with Assets selected and no UUID controls.
            expect(page.locator("#view-title")).to_have_text("Assets")
            expect(page.locator("#tenant-id")).to_have_count(0)
            expect(page.locator("body")).not_to_contain_text("X-Tenant-ID")

            # Upload two files into one asset.
            page.locator("#upload-toggle").click()
            page.locator("#asset-name").fill("Site photos")
            page.locator("#asset-file").set_input_files([str(red), str(blue)])
            expect(page.locator("#asset-file-info")).to_contain_text("red-site.png")
            expect(page.locator("#asset-file-info")).to_contain_text("blue-site.bmp")
            page.locator("#upload-button").click()

            expect(page.locator("#upload-status")).to_contain_text(
                "indexed for search", timeout=30_000
            )
            assert any(url.endswith("/parts/2") for url in upload_parts), upload_parts
            expect(page.locator(".asset-card")).to_have_count(1)
            expect(page.locator(".asset-card")).to_contain_text("Site photos")

            # The newly uploaded asset is selected and can be searched in place.
            expect(page.locator("#asset-detail-content")).to_contain_text("Site photos")
            expect(page.locator("#asset-detail-content")).to_contain_text("2 sources")
            page.locator("#asset-detail-content button", has_text="Search this asset").click()
            expect(page.locator("#search-scope")).to_have_value("selected")

            page.locator("#search-query").fill("red")
            page.locator("#search-button").click()
            expect(page.locator("#search-status")).to_contain_text("result", timeout=20_000)
            expect(page.locator(".result-card")).to_have_count(2)
            expect(page.locator(".result-card").first).to_contain_text("Match 1")
            expect(page.locator("#subasset-button")).to_be_enabled()

            # Save the result as a durable sub-asset and inspect its lineage.
            page.locator("#subasset-button").click()
            expect(page.locator("#search-status")).to_contain_text(
                "Saved", timeout=20_000
            )
            expect(page.locator(".asset-card")).to_have_count(2)
            expect(page.locator(".asset-card", has_text="Sub-asset: red")).to_have_count(1)
            expect(page.locator("#asset-detail-content")).to_contain_text("Sub-asset lineage")
            expect(page.locator("#asset-detail-content")).to_contain_text(
                "Parent assets: Site photos"
            )
            expect(page.locator("#asset-detail-content")).to_contain_text(
                "Created from search: “red”"
            )

            # The small library area is reachable and does not expose UUIDs.
            page.locator(".nav-item[data-view='jobs']").click()
            expect(page.locator("#view-title")).to_have_text("Jobs")
            expect(page.locator(".job-card").first).to_contain_text("Search indexing")
            page.locator(".nav-item[data-view='datasets']").click()
            expect(page.locator("#view-title")).to_have_text("Datasets")
            expect(page.locator("#datasets-list")).to_contain_text("No dataset versions yet")
            page.locator(".nav-item[data-view='assets']").click()
            expect(page.locator(".asset-card")).to_have_count(2)

            # No UUID is ever rendered into the page text.
            body_text = page.locator("body").inner_text()
            assert _uuid_pattern().search(body_text) is None, body_text
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth + 1"
            )

            browser.close()


def test_upload_status_waits_for_slow_indexing(tmp_path, asset_images):
    """Indexing a video can take minutes; the upload status must still end on the result.

    The UI used to give up after 60 polls (about 30 s) and leave
    "Queuing search indexing…" on screen forever. Here the index job reports
    "running" for 70 polls before the real answer comes through.
    """
    playwright_api = pytest.importorskip("playwright.sync_api")
    expect = playwright_api.expect
    app = create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            embedding_provider=FakeEmbeddingProvider(),
            start_workers=True,
            worker_poll_interval=0.05,
        )
    )
    red, _ = asset_images
    polls = {"count": 0}

    def slow_index_job(route):
        polls["count"] += 1
        if polls["count"] <= 70:
            response = route.fetch()
            body = response.json()
            body["status"] = "running"
            route.fulfill(response=response, json=body)
        else:
            route.continue_()

    with _serve(app) as base_url:
        with playwright_api.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except playwright_api.Error as exc:
                pytest.skip(f"Playwright Chromium is not installed: {exc}")
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.clock.install()
            page.route("**/v1/index-jobs/*", slow_index_job)
            page.goto(base_url, wait_until="domcontentloaded")
            page.locator("#upload-toggle").click()
            page.locator("#asset-name").fill("Long video")
            page.locator("#asset-file").set_input_files([str(red)])
            page.locator("#upload-button").click()

            expect(page.locator("#upload-status")).to_contain_text("Indexing Long video for search", timeout=15_000)
            # Move the page clock forward so the 70 slow polls take seconds, not minutes.
            for _ in range(400):
                if polls["count"] > 70:
                    break
                page.clock.fast_forward(2000)
                page.wait_for_timeout(25)
            assert polls["count"] > 70, polls
            expect(page.locator("#upload-status")).to_contain_text("indexed for search", timeout=15_000)
            browser.close()
