"""Phase 4c: review in the dataset gallery — approve, reject, bulk, delete, click-to-add, build approved."""

from __future__ import annotations

import pytest

from test_dataset_gallery_playwright import GallerySam3, _build_app, _make_images, _serve, _upload_and_run


def test_review_flow_in_the_gallery(tmp_path):
    playwright_api = pytest.importorskip("playwright.sync_api")
    expect = playwright_api.expect
    image_paths = _make_images(tmp_path, 6)
    with _serve(_build_app(tmp_path, GallerySam3())) as base_url:
        with playwright_api.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except playwright_api.Error as exc:
                pytest.skip(f"Playwright Chromium is not installed: {exc}")
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            page.goto(base_url, wait_until="domcontentloaded")
            _upload_and_run(page, image_paths, expect)

            page.locator(".nav-item[data-view='datasets']").click()
            card = page.locator(".dataset-card").first
            expect(card).to_be_visible(timeout=15_000)
            card.locator("[data-open-gallery]").click()
            counts = page.locator("#gallery-review-counts")
            expect(counts).to_have_text("6 pending · 0 approved · 0 rejected", timeout=15_000)
            expect(page.locator("#gallery-counter")).to_have_text("image 1 of 6")

            # Approve with the button: the image is marked and the gallery moves on.
            page.locator("#gallery-approve").click()
            expect(counts).to_have_text("5 pending · 1 approved · 0 rejected")
            expect(page.locator("#gallery-counter")).to_have_text("image 2 of 6")
            expect(page.locator(".gallery-thumb-wrap").nth(0)).to_have_attribute("data-status", "approved")

            # Keyboard: R rejects, A approves.
            page.locator("body").press("r")
            expect(counts).to_have_text("4 pending · 1 approved · 1 rejected")
            expect(page.locator("#gallery-counter")).to_have_text("image 3 of 6")
            page.locator("body").press("a")
            expect(counts).to_have_text("3 pending · 2 approved · 1 rejected")

            # Status filter.
            page.locator("#gallery-filter-status").select_option("rejected")
            expect(page.locator("#gallery-counter")).to_have_text("image 1 of 1")
            page.locator("#gallery-filter-status").select_option("pending")
            expect(page.locator("#gallery-counter")).to_have_text("image 1 of 3")

            # Bulk: select two pending thumbnails and reject them.
            picks = page.locator(".gallery-thumb-pick")
            picks.nth(0).check()
            picks.nth(1).check()
            expect(page.locator("#gallery-reject-selected")).to_have_text("Reject 2 selected")
            page.locator("#gallery-reject-selected").click()
            expect(counts).to_have_text("1 pending · 2 approved · 3 rejected")
            page.locator("#gallery-filter-status").select_option("all")

            # Delete the annotation of an approved image, then add one by clicking.
            page.locator(".gallery-thumb").nth(2).click()
            rows = page.locator("#gallery-annotations .gallery-annotation")
            before = rows.count()
            if before:
                page.locator("#gallery-annotations .gallery-annotation-delete").first.click()
                expect(page.locator("#dataset-gallery-status")).to_contain_text("Annotation deleted")
                expect(rows).to_have_count(before - 1)

            # Find an image where SAM 3 returns a mask, then click to add an object.
            page.locator("#gallery-filter-status").select_option("all")
            page.locator(".gallery-thumb").nth(1).click()
            existing = rows.count()
            page.locator("#gallery-click-mode").click()
            expect(page.locator("#gallery-click-status")).to_contain_text("Click the object", timeout=10_000)
            page.locator("#gallery-overlay").click(position={"x": 40, "y": 40})
            expect(page.locator("#gallery-click-save")).to_be_enabled(timeout=10_000)
            page.locator("#gallery-click-save").click()
            expect(page.locator("#dataset-gallery-status")).to_contain_text("Object added", timeout=10_000)
            expect(rows).to_have_count(existing + 1)
            page.locator("#gallery-click-done").click()

            # Approve the rest and build the approved dataset.
            page.locator("#gallery-approve-pending").click()
            expect(counts).to_have_text("0 pending · 3 approved · 3 rejected")
            page.locator("#gallery-build-approved").click()
            expect(page.locator("#gallery-counter")).to_have_text("image 1 of 3", timeout=15_000)
            expect(counts).to_have_text("0 pending · 3 approved · 0 rejected")
            browser.close()
