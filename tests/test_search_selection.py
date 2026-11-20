"""Save only the search results the user selected as a new asset."""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from agentic_cv.application import LocalAppConfig, create_local_app

from test_assets_ui_playwright import FakeEmbeddingProvider, _serve

TENANT = {"X-Tenant-ID": "00000000-0000-0000-0000-000000000001"}
OTHER = {"X-Tenant-ID": "00000000-0000-0000-0000-000000000002"}
COLORS = {"red": (255, 0, 0), "orange": (255, 90, 0), "magenta": (230, 0, 160)}


def make_app(tmp_path, start_workers=False):
    return create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            embedding_provider=FakeEmbeddingProvider(),
            start_workers=start_workers,
            worker_poll_interval=0.05,
        )
    )


def upload_asset(client, name, colors, headers=TENANT):
    asset_id = client.post("/v1/assets", headers=headers, json={"name": name}).json()["asset_id"]
    for color in colors:
        buffer = io.BytesIO()
        Image.new("RGB", (32, 24), COLORS[color]).save(buffer, format="PNG")
        payload = buffer.getvalue()
        upload = client.post(
            f"/v1/assets/{asset_id}/uploads",
            headers=headers,
            json={
                "filename": f"{color}.png",
                "media_type": "image/png",
                "expected_sha256": hashlib.sha256(payload).hexdigest(),
            },
        ).json()["upload_id"]
        client.put(f"/v1/uploads/{upload}/parts/1", files={"part": (f"{color}.png", io.BytesIO(payload), "image/png")})
        assert client.post(f"/v1/uploads/{upload}/complete").status_code == 200
    assert client.post(f"/v1/assets/{asset_id}/index", headers=headers).status_code == 200
    return asset_id


def test_save_only_selected_results_as_a_named_asset(tmp_path):
    with TestClient(make_app(tmp_path)) as client:
        upload_asset(client, "Clips", ["red", "orange", "magenta"])
        found = client.post("/v1/search", headers=TENANT, json={"query": "red", "limit": 3}).json()["results"]
        assert len(found) == 3
        keep = [found[0]["record_id"], found[2]["record_id"]]

        saved = client.post(
            "/v1/sub-assets/from-search",
            headers=TENANT,
            json={"query": "red", "record_ids": keep + [keep[0]], "name": "Only the good ones"},
        )
        assert saved.status_code == 201, saved.text
        body = saved.json()
        assert body["result_ids"] == keep  # selection order, duplicates dropped
        assert body["source_count"] == 2
        assert body["asset_name"] == "Only the good ones"

        assets = client.get("/v1/assets", headers=TENANT).json()["assets"]
        created = next(item for item in assets if item["name"] == "Only the good ones")
        assert len(created["sources"]) == 2


def test_selection_is_validated(tmp_path):
    with TestClient(make_app(tmp_path)) as client:
        upload_asset(client, "Clips", ["red", "orange"])
        upload_asset(client, "Someone else's", ["magenta"], headers=OTHER)
        theirs = client.post("/v1/search", headers=OTHER, json={"query": "red"}).json()["results"]
        assert len(theirs) == 1

        unknown = client.post(
            "/v1/sub-assets/from-search", headers=TENANT,
            json={"query": "red", "record_ids": ["no-such-record"]},
        )
        foreign = client.post(
            "/v1/sub-assets/from-search", headers=TENANT,
            json={"query": "red", "record_ids": [theirs[0]["record_id"]]},
        )
        empty = client.post(
            "/v1/sub-assets/from-search", headers=TENANT, json={"query": "red", "record_ids": []},
        )
        assert unknown.status_code == 404
        assert foreign.status_code == 404
        assert empty.status_code == 422


def test_search_is_scoped_to_the_tenant_and_honours_min_score(tmp_path):
    with TestClient(make_app(tmp_path)) as client:
        upload_asset(client, "Clips", ["red", "orange", "magenta"])
        upload_asset(client, "Someone else's", ["red"], headers=OTHER)

        mine = client.post("/v1/search", headers=TENANT, json={"query": "red", "limit": 50}).json()["results"]
        assert len(mine) == 3
        scores = sorted(item["score"] for item in mine)
        cutoff = (scores[0] + scores[1]) / 2
        strong = client.post(
            "/v1/search", headers=TENANT, json={"query": "red", "limit": 50, "min_score": cutoff}
        ).json()["results"]
        assert len(strong) == 2
        assert all(item["score"] >= cutoff for item in strong)


def test_browser_select_results_then_save_selection(tmp_path):
    playwright_api = pytest.importorskip("playwright.sync_api")
    expect = playwright_api.expect
    images = []
    for color in ("red", "orange", "magenta"):
        path = tmp_path / f"{color}.png"
        Image.new("RGB", (96, 64), COLORS[color]).save(path)
        images.append(str(path))

    with _serve(make_app(tmp_path, start_workers=True)) as base_url:
        with playwright_api.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except playwright_api.Error as exc:
                pytest.skip(f"Playwright Chromium is not installed: {exc}")
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(base_url, wait_until="domcontentloaded")
            page.locator("#upload-toggle").click()
            page.locator("#asset-name").fill("Wildlife")
            page.locator("#asset-file").set_input_files(images)
            page.locator("#upload-button").click()
            expect(page.locator("#upload-status")).to_contain_text("indexed for search", timeout=30_000)

            page.locator("#search-query").fill("red")
            page.locator("#search-limit").fill("3")
            page.locator("#search-button").click()
            cards = page.locator("#search-results .result-card")
            expect(cards).to_have_count(3, timeout=20_000)
            expect(page.locator("#search-selected-count")).to_have_text("3 of 3 selected")
            expect(page.locator("#subasset-button")).to_have_text("Save 3 selected as asset")

            # Exclude the last match, like an antelope frame in an elephant search.
            cards.nth(2).click()
            expect(cards.nth(2)).to_have_attribute("aria-pressed", "false")
            expect(page.locator("#search-selected-count")).to_have_text("2 of 3 selected")

            page.locator("#search-select-none").click()
            expect(page.locator("#subasset-button")).to_be_disabled()
            page.locator("#search-select-all").click()
            cards.nth(2).click()

            page.locator("#subasset-name").fill("Curated wildlife")
            page.locator("#subasset-button").click()
            expect(page.locator("#search-status")).to_contain_text("with 2 matched images", timeout=20_000)
            expect(page.locator(".asset-card", has_text="Curated wildlife")).to_have_count(1)
            expect(page.locator("#asset-detail-content")).to_contain_text("2 sources")
            browser.close()
