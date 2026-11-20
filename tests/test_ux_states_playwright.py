"""Browser UX states: loading, empty, and recoverable errors.

These tests drive the real UI against a real uvicorn server, but hold or fail
individual HTTP responses so the transient states are deterministic. The
default suite still runs without Playwright by skipping when Chromium is absent.
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest

from agentic_cv.application import LocalAppConfig, create_local_app


class FakeEmbeddingProvider:
    def embed_image(self, image):
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        vector = np.array([pixels[0] - pixels[1], pixels[2] - pixels[1]], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return np.array([1.0, 0.0], dtype=np.float32)
        return vector / norm

    def embed_text(self, query):
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


def _app(tmp_path):
    return create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            embedding_provider=FakeEmbeddingProvider(),
            start_workers=True,
            worker_poll_interval=0.05,
        )
    )


def _uuid_pattern():
    import re

    return re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        re.IGNORECASE,
    )


def _launch(playwright):
    Error = pytest.importorskip("playwright.sync_api").Error
    try:
        return playwright.chromium.launch()
    except Error as exc:
        pytest.skip(f"Playwright Chromium is not installed: {exc}")


def test_loading_then_empty_states_across_views(tmp_path):
    playwright_api = pytest.importorskip("playwright.sync_api")
    expect = playwright_api.expect

    app = _app(tmp_path)
    with _serve(app) as base_url:
        with playwright_api.sync_playwright() as playwright:
            browser = _launch(playwright)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            held = []

            def hold(route):
                held.append(route)

            page.route("**/v1/assets", hold)
            try:
                page.goto(base_url, wait_until="domcontentloaded")

                # The gallery paints skeletons while the library request is open.
                expect(page.locator("#asset-gallery .skeleton-card")).to_have_count(
                    3, timeout=10_000
                )
                expect(page.locator("#asset-gallery")).to_have_attribute(
                    "aria-busy", "true"
                )
                expect(page.locator("#assets-meta")).to_have_text("Loading assets…")

                # The Agent view shows its own loading state rather than a blank picker.
                page.locator(".nav-item[data-view='agent']").click()
                expect(page.locator("#agent-assets")).to_contain_text("Loading assets…")
                expect(page.locator("#agent-plan")).to_be_disabled()

                page.locator(".nav-item[data-view='assets']").click()
                expect(page.locator("#asset-gallery .skeleton-card")).to_have_count(3)

                # Release every held request; both views settle on a real empty state.
                for route in held:
                    route.continue_()
                held.clear()
                page.unroute("**/v1/assets")

                expect(page.locator("#asset-gallery .state-panel-empty")).to_contain_text(
                    "No assets yet.", timeout=10_000
                )
                expect(page.locator("#asset-gallery")).to_have_attribute(
                    "aria-busy", "false"
                )
                expect(page.locator("#asset-gallery .state-panel-empty button")).to_have_text(
                    "Upload media"
                )
                page.locator("#asset-gallery .state-panel-empty button").click()
                expect(page.locator("#upload-panel")).to_be_visible()
                page.locator("#upload-close").click()
                expect(page.locator("#upload-panel")).to_be_hidden()

                page.locator(".nav-item[data-view='agent']").click()
                expect(page.locator("#agent-plan")).to_be_disabled()
                expect(page.locator("#agent-assets")).to_contain_text(
                    "No assets yet — upload media first"
                )
                expect(page.locator("#agent-status")).to_contain_text(
                    "No assets yet. Upload media in Assets"
                )

                page.locator(".nav-item[data-view='datasets']").click()
                expect(page.locator("#datasets-list .state-panel-empty")).to_contain_text(
                    "No dataset versions yet."
                )
                expect(page.locator("#datasets-list")).to_have_attribute(
                    "aria-busy", "false"
                )
                page.locator("#datasets-list .state-panel-empty button").click()
                expect(page.locator("#view-title")).to_have_text("Agent")

                page.locator(".nav-item[data-view='jobs']").click()
                expect(page.locator("#jobs-list .state-panel-empty")).to_contain_text(
                    "No background jobs yet."
                )
                page.locator("#jobs-list .state-panel-empty button").click()
                expect(page.locator("#view-title")).to_have_text("Assets")

                assert _uuid_pattern().search(page.locator("body").inner_text()) is None
            finally:
                for route in held:
                    route.abort()
                page.unroute("**/v1/assets")
                browser.close()


def test_backend_errors_are_recoverable_and_retryable(tmp_path):
    playwright_api = pytest.importorskip("playwright.sync_api")
    expect = playwright_api.expect

    app = _app(tmp_path)
    with _serve(app) as base_url:
        with playwright_api.sync_playwright() as playwright:
            browser = _launch(playwright)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})

            def fail_assets(route):
                route.fulfill(
                    status=500,
                    content_type="application/json",
                    body='{"detail": "asset store is offline"}',
                )

            page.route("**/v1/assets", fail_assets)
            try:
                page.goto(base_url, wait_until="domcontentloaded")
                expect(page.locator("#asset-gallery .state-panel-error")).to_contain_text(
                    "Could not load assets.", timeout=10_000
                )
                expect(page.locator("#asset-gallery .state-panel-error")).to_contain_text(
                    "asset store is offline"
                )
                expect(page.locator("#asset-gallery")).to_have_attribute(
                    "aria-busy", "false"
                )
                expect(page.locator("#assets-meta")).to_have_text("Could not load assets")
                assert _uuid_pattern().search(page.locator("body").inner_text()) is None

                page.unroute("**/v1/assets")
                page.locator("#asset-gallery .state-panel-error button").click()
                expect(page.locator("#asset-gallery .state-panel-empty")).to_contain_text(
                    "No assets yet.", timeout=10_000
                )

                def fail_datasets(route):
                    route.fulfill(
                        status=503,
                        content_type="application/json",
                        body='{"detail": "dataset store is unavailable"}',
                    )

                page.route("**/v1/datasets", fail_datasets)
                page.locator(".nav-item[data-view='datasets']").click()
                expect(page.locator("#datasets-list .state-panel-error")).to_contain_text(
                    "Could not load dataset versions.", timeout=10_000
                )
                expect(page.locator("#datasets-list .state-panel-error")).to_contain_text(
                    "dataset store is unavailable"
                )
                page.unroute("**/v1/datasets")
                page.locator("#datasets-list .state-panel-error button").click()
                expect(page.locator("#datasets-list .state-panel-empty")).to_contain_text(
                    "No dataset versions yet.", timeout=10_000
                )

                def fail_jobs(route):
                    route.fulfill(
                        status=500,
                        content_type="application/json",
                        body='{"detail": "job store is offline"}',
                    )

                page.route("**/v1/jobs", fail_jobs)
                page.locator(".nav-item[data-view='jobs']").click()
                expect(page.locator("#jobs-list .state-panel-error")).to_contain_text(
                    "Could not load background jobs.", timeout=10_000
                )
                expect(page.locator("#jobs-list .state-panel-error")).to_contain_text(
                    "job store is offline"
                )
                page.unroute("**/v1/jobs")
                page.locator("#jobs-list .state-panel-error button").click()
                expect(page.locator("#jobs-list .state-panel-empty")).to_contain_text(
                    "No background jobs yet.", timeout=10_000
                )
            finally:
                browser.close()
