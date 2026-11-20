"""Phase 5: multi-asset selection (search + Agent), thumbnail cache, original names."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from agentic_cv.agent import RulesTaskPlanner
from agentic_cv.application import LocalAppConfig, create_local_app
from agentic_cv.calibration_validation import any_prediction_selection

from test_agent_chat_e2e import FakeSam3, headers, wait_for_session
from test_assets_ui_playwright import FakeEmbeddingProvider, _serve
from test_search_selection import COLORS, upload_asset


def make_app(tmp_path, start_workers=True):
    return create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            embedding_provider=FakeEmbeddingProvider(),
            sam3_provider=FakeSam3(),
            agent_planner=RulesTaskPlanner(),
            calibration_validator_factory=lambda tenant_id, session_id: any_prediction_selection(),
            start_workers=start_workers,
            worker_poll_interval=0.05,
        )
    )


def test_thumbnails_are_generated_once_per_source(tmp_path):
    with TestClient(make_app(tmp_path, start_workers=False)) as client:
        asset_id = upload_asset(client, "Clips", ["red"])
        first = client.get(f"/v1/assets/{asset_id}/thumbnail", headers=headers())
        assert first.status_code == 200
        cached = list((tmp_path / "data" / "derived" / "thumbnails").glob("*.jpg"))
        assert len(cached) == 1
        cached[0].write_bytes(b"cached-bytes")  # prove the second response comes from the cache
        second = client.get(f"/v1/assets/{asset_id}/thumbnail", headers=headers())
        assert second.content == b"cached-bytes"
        assert second.headers["etag"]


def test_dataset_and_review_images_carry_the_original_file_name(tmp_path):
    with TestClient(make_app(tmp_path), raise_server_exceptions=False) as client:
        asset_id = upload_asset(client, "Clips", ["red", "orange"])
        session = client.post(
            "/v1/agent/sessions", headers=headers(),
            json={"asset_ids": [asset_id], "message": "detect red things"},
        ).json()
        client.post(f"/v1/agent/sessions/{session['session_id']}/confirm", headers=headers(), json={"threshold": 0.5, "wait": True})
        done = wait_for_session(client, session["session_id"])
        version = done["dataset_version_id"]
        images = client.get(f"/v1/datasets/{version}/images", headers=headers()).json()["images"]
        review = client.get(f"/v1/datasets/{version}/review", headers=headers()).json()["images"]
        assert sorted(image["source_name"] for image in images) == ["orange.png", "red.png"]
        assert sorted(image["source_name"] for image in review) == ["orange.png", "red.png"]
        assert all(image["file_name"].startswith("images/") for image in images)  # export names unchanged


def test_browser_tick_assets_search_them_and_use_them_in_the_agent(tmp_path):
    playwright_api = pytest.importorskip("playwright.sync_api")
    expect = playwright_api.expect
    paths = []
    for color in ("red", "orange", "magenta"):
        path = tmp_path / f"{color}.png"
        Image.new("RGB", (96, 64), COLORS[color]).save(path)
        paths.append(path)

    with _serve(make_app(tmp_path)) as base_url:
        with playwright_api.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except playwright_api.Error as exc:
                pytest.skip(f"Playwright Chromium is not installed: {exc}")
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            page.goto(base_url, wait_until="domcontentloaded")
            for index, path in enumerate(paths):
                page.locator("#upload-toggle").click()
                page.locator("#asset-name").fill(f"Site {index + 1}")
                page.locator("#asset-file").set_input_files(str(path))
                page.locator("#upload-button").click()
                expect(page.locator("#upload-status")).to_contain_text("indexed for search", timeout=30_000)
            expect(page.locator(".asset-card")).to_have_count(3)

            # Tick two assets: the selection bar and the search scope follow.
            page.locator(".asset-card-wrap", has_text="Site 1").locator(".asset-pick").check()
            page.locator(".asset-card-wrap", has_text="Site 3").locator(".asset-pick").check()
            expect(page.locator("#assets-selection-count")).to_have_text("2 selected")
            page.locator("#assets-search-selected").click()
            expect(page.locator("#search-scope")).to_have_value("chosen")
            expect(page.locator("#search-scope-chip")).to_have_text("2 assets")
            page.locator("#search-query").fill("red")
            page.locator("#search-limit").fill("10")
            page.locator("#search-button").click()
            expect(page.locator("#search-results .result-card")).to_have_count(2, timeout=20_000)
            expect(page.locator("#search-results")).not_to_contain_text("Site 2")

            # Use them in the Agent: a new task on exactly those two assets.
            page.locator("#assets-use-in-agent").click()
            expect(page.locator("#view-title")).to_have_text("Agent")
            checked = page.locator("#agent-assets .agent-asset-option input:checked")
            expect(checked).to_have_count(2)
            expect(page.locator("#agent-assets-count")).to_have_text("2 selected")
            page.locator("#agent-message").fill("Detect red signs, export YOLO")
            page.locator("#agent-plan").click()
            expect(page.locator("#agent-run-pill")).to_have_text("Awaiting confirmation", timeout=15_000)
            expect(page.locator("#agent-spec")).to_contain_text("Site 1, Site 3")
            browser.close()
