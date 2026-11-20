"""Drive the real app (real SAM 3 / VLM / LLM / SigLIP) through the browser and screenshot the board.

Not collected by pytest. Uses the 9 local demo frames under .agentic-cv-demo/
and writes board-live.png, board-final.png, and a fresh data dir into <out>.

    python scripts/real_ui_calibration_board.py <repo> <out dir>
"""

import glob
import sys
import time
from pathlib import Path

repo = Path(sys.argv[1])
out = Path(sys.argv[2])
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "tests"))

from playwright.sync_api import expect, sync_playwright  # noqa: E402

from agentic_cv.application import LocalAppConfig, create_local_app  # noqa: E402
from test_agent_ui_playwright import _serve  # noqa: E402

images = sorted(glob.glob(str(repo / ".agentic-cv-demo/dataset-images/00000000-0000-0000-0000-000000000001/cb7c4b2d-8f33-4e35-b66c-b7b75b8d953f/images/*.jpg")))
app = create_local_app(LocalAppConfig.from_environment(out / "data"))
t0 = time.time()
with _serve(app) as url, sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1500, "height": 1300})
    page.goto(url)
    page.locator("#upload-toggle").click()
    page.locator("#asset-name").fill("Savanna clips")
    page.locator("#asset-file").set_input_files(images)
    page.locator("#upload-button").click()
    expect(page.locator("#upload-status")).to_contain_text("indexed for search", timeout=300_000)
    print("indexed", round(time.time() - t0), flush=True)

    page.locator(".nav-item[data-view='agent']").click()
    page.locator("#agent-message").fill("Segment the elephants and their tusks, export YOLO")
    page.locator("#agent-plan").click()
    expect(page.locator("#agent-run-pill")).to_have_text("Awaiting confirmation", timeout=120_000)
    print("planned", round(time.time() - t0), page.locator("#agent-spec").inner_text().replace("\n", " | "), flush=True)
    page.locator("#agent-confirm").click()
    expect(page.locator("#agent-run-pill")).to_have_text("Calibrating", timeout=10_000)
    print("confirm returned", round(time.time() - t0), flush=True)
    expect(page.locator("#agent-board")).to_be_visible(timeout=300_000)
    expect(page.locator("#agent-board .agent-board-chip").first).to_be_visible(timeout=600_000)
    page.locator(".agent-run").screenshot(path=str(out / "board-live.png"))
    print("live screenshot", round(time.time() - t0), flush=True)
    expect(page.locator("#agent-run-pill")).to_have_text("Dataset ready", timeout=1_200_000)
    print("done", round(time.time() - t0), flush=True)
    page.locator("#agent-board").screenshot(path=str(out / "board-final.png"))
    page.locator("#agent-board .agent-board-slider").first.fill("0.85")
    page.locator("#agent-board .agent-board-class").first.screenshot(path=str(out / "board-elephants-085.png"))
    print(page.locator("#agent-messages").inner_text()[-600:], flush=True)
    browser.close()
