"""Dark theme switch: follows the system until used, then remembers the choice."""

from __future__ import annotations

import pytest

from agentic_cv.application import LocalAppConfig, create_local_app

from test_assets_ui_playwright import FakeEmbeddingProvider, _serve

LIGHT_CANVAS = "rgb(243, 245, 249)"
DARK_CANVAS = "rgb(18, 22, 29)"


def body_background(page):
    return page.evaluate("() => getComputedStyle(document.body).backgroundColor")


def test_theme_switch_follows_system_then_remembers_the_choice(tmp_path):
    playwright_api = pytest.importorskip("playwright.sync_api")
    expect = playwright_api.expect
    app = create_local_app(
        LocalAppConfig(data_dir=tmp_path / "data", embedding_provider=FakeEmbeddingProvider(), start_workers=False)
    )
    with _serve(app) as base_url:
        with playwright_api.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except playwright_api.Error as exc:
                pytest.skip(f"Playwright Chromium is not installed: {exc}")

            # A light system: the switch is off and the canvas is light.
            light = browser.new_context(color_scheme="light")
            page = light.new_page()
            page.goto(base_url, wait_until="domcontentloaded")
            toggle = page.locator("#theme-toggle")
            expect(toggle).to_have_attribute("aria-checked", "false")
            assert body_background(page) == LIGHT_CANVAS

            # Turn dark on: colours change and the choice survives a reload.
            toggle.click()
            expect(toggle).to_have_attribute("aria-checked", "true")
            expect(page.locator("html")).to_have_attribute("data-theme", "dark")
            page.wait_for_function(f"() => getComputedStyle(document.body).backgroundColor === '{DARK_CANVAS}'")
            page.reload(wait_until="domcontentloaded")
            expect(page.locator("html")).to_have_attribute("data-theme", "dark")
            expect(toggle).to_have_attribute("aria-checked", "true")
            assert body_background(page) == DARK_CANVAS
            light.close()

            # A dark system: on by default; choosing light wins over the system.
            dark = browser.new_context(color_scheme="dark")
            page = dark.new_page()
            page.goto(base_url, wait_until="domcontentloaded")
            toggle = page.locator("#theme-toggle")
            expect(toggle).to_have_attribute("aria-checked", "true")
            assert body_background(page) == DARK_CANVAS
            toggle.click()
            expect(toggle).to_have_attribute("aria-checked", "false")
            page.wait_for_function(f"() => getComputedStyle(document.body).backgroundColor === '{LIGHT_CANVAS}'")
            page.reload(wait_until="domcontentloaded")
            assert body_background(page) == LIGHT_CANVAS
            dark.close()
            browser.close()
