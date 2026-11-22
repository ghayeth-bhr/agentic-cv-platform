"""Browser journey: import a COCO ZIP from the Datasets view and land in the gallery."""

from __future__ import annotations

import io
import json
import zipfile

import pytest
from PIL import Image

from agentic_cv.application import LocalAppConfig, create_local_app

from test_dataset_gallery_playwright import _serve


def _coco_zip(path):
    def png(color):
        buffer = io.BytesIO()
        Image.new("RGB", (80, 60), color).save(buffer, format="PNG")
        return buffer.getvalue()

    coco = {
        "images": [
            {"id": 1, "file_name": "one.png", "width": 80, "height": 60},
            {"id": 2, "file_name": "two.png", "width": 80, "height": 60},
        ],
        "categories": [{"id": 5, "name": "taxi"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [10, 10, 30, 20],
             "segmentation": [[10, 10, 40, 10, 40, 30, 10, 30]], "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 5, "bbox": [20, 15, 25, 25], "iscrowd": 0},
        ],
    }
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("annotations.json", json.dumps(coco))
        bundle.writestr("images/one.png", png((230, 190, 30)))
        bundle.writestr("images/two.png", png((40, 120, 200)))
    return path


def test_import_a_coco_zip_and_review_it_in_the_gallery(tmp_path):
    playwright_api = pytest.importorskip("playwright.sync_api")
    expect = playwright_api.expect
    archive = _coco_zip(tmp_path / "taxis.zip")
    app = create_local_app(LocalAppConfig(data_dir=tmp_path / "data", start_workers=False))

    with _serve(app) as base_url, playwright_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except playwright_api.Error as exc:
            pytest.skip(f"Playwright Chromium is not installed: {exc}")
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(base_url, wait_until="domcontentloaded")
        page.locator(".nav-item[data-view='datasets']").click()

        panel = page.locator("#dataset-import-panel")
        expect(panel).to_be_hidden()
        page.locator("#dataset-import-toggle").click()
        expect(panel).to_be_visible()

        page.locator("#dataset-import-button").click()
        expect(page.locator("#dataset-import-status")).to_have_text("Choose a ZIP file first.")

        page.locator("#dataset-import-name").fill("Open Images taxis")
        page.locator("#dataset-import-file").set_input_files(str(archive))
        page.locator("#dataset-import-button").click()

        expect(page.locator("#datasets-status")).to_have_text(
            "Imported 2 images and 2 objects.", timeout=15_000
        )
        expect(panel).to_be_hidden()
        expect(page.locator("#dataset-gallery")).to_be_visible(timeout=15_000)
        expect(page.locator("#gallery-subtitle")).to_contain_text("Imported COCO dataset")
        expect(page.locator("#gallery-counter")).to_contain_text("of 2")
        expect(page).to_have_url(f"{base_url}/#/datasets/" + page.url.rsplit("/", 1)[-1])

        page.goto(f"{base_url}/#/datasets", wait_until="domcontentloaded")
        expect(page.locator(".dataset-card")).to_have_count(1, timeout=15_000)
        expect(page.locator(".dataset-card").first).to_contain_text("Open Images taxis")
        browser.close()
