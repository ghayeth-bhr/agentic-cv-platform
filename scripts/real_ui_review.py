"""Real models: open the elephants dataset, review, and add an object with one click."""

import sys
import time
from pathlib import Path

repo = Path(sys.argv[1])
data = Path(sys.argv[2])
out = Path(sys.argv[3])
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "tests"))

from playwright.sync_api import expect, sync_playwright  # noqa: E402

from agentic_cv.application import LocalAppConfig, create_local_app  # noqa: E402
from test_agent_ui_playwright import _serve  # noqa: E402

app = create_local_app(LocalAppConfig.from_environment(data))
with _serve(app) as url, sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1500, "height": 1150})
    page.goto(url + "/#/datasets")
    card = page.locator(".dataset-card").first
    expect(card).to_be_visible(timeout=30_000)
    card.locator("[data-open-gallery]").click()
    expect(page.locator("#gallery-review-counts")).to_contain_text("pending", timeout=30_000)
    print("counts", page.locator("#gallery-review-counts").inner_text(), flush=True)
    # Find the mother-and-calf frame: the one whose caption mentions the last source.
    thumbs = page.locator(".gallery-thumb")
    best = None
    for index in range(thumbs.count()):
        thumbs.nth(index).click()
        page.wait_for_timeout(300)
        rows = page.locator("#gallery-annotations .gallery-annotation").count()
        print("image", index, page.locator("#gallery-caption").inner_text(), rows, "annotations", flush=True)
    target = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    thumbs.nth(target).click()
    page.wait_for_timeout(500)
    page.locator(".gallery-layout").screenshot(path=str(out / "review-before.png"))
    before = page.locator("#gallery-annotations .gallery-annotation").count()
    page.locator("#gallery-click-mode").click()
    expect(page.locator("#gallery-click-status")).to_contain_text("Click the object", timeout=60_000)
    canvas = page.locator("#gallery-overlay")
    box = canvas.bounding_box()
    fx, fy = float(sys.argv[5]), float(sys.argv[6])  # fractions of the image
    started = time.perf_counter()
    page.mouse.click(box["x"] + box["width"] * fx, box["y"] + box["height"] * fy)
    expect(page.locator("#gallery-click-save")).to_be_enabled(timeout=120_000)
    print("first click (incl. tracker load)", round(time.perf_counter() - started, 2), "s", flush=True)
    page.locator(".gallery-layout").screenshot(path=str(out / "review-click.png"))
    started = time.perf_counter()
    page.locator("#gallery-click-save").click()
    expect(page.locator("#dataset-gallery-status")).to_contain_text("Object added", timeout=60_000)
    print("saved in", round(time.perf_counter() - started, 2), "s; annotations", before, "->",
          page.locator("#gallery-annotations .gallery-annotation").count(), flush=True)
    page.locator("#gallery-click-done").click()
    page.locator("#gallery-approve").click()
    page.wait_for_timeout(800)
    print("counts", page.locator("#gallery-review-counts").inner_text(), flush=True)
    browser.close()
