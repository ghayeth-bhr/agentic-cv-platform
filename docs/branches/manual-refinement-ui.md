# P4 — Manual refinement UI

Branch: `feature/manual-refinement-ui` (stacked on the pushed P3 `feature/agent-chat`).

## Goal

Make a bad label correctable from the browser without exposing UUIDs or asking the
user to call an API. A correction must be durable, must not mutate an immutable
dataset version, and must be visible in the next export.

## What changed

### Backend

- `click_refinement.py`
  - Refinement sessions now carry optional tenant, dataset version, category
    prompt, and category ID.
  - Providers can implement `refine_with_prompt(...)`; the existing two-argument
    `refine(...)` providers still work.
  - `Sam3ClickRefinementProvider` prompts SAM 3 with the dataset category and
    ranks the returned masks using positive/negative point containment.
- `batch_labeling.py`
  - Exposed `mask_polygons(...)` for reuse by accepted manual revisions.
- `dataset_builder.py`
  - Added `rebuild_from_version(...)`: creates a new immutable version from the
    latest annotation revisions and copies the source images.
- `api.py`
  - `GET /v1/datasets/{version_id}/images`
  - `GET /v1/datasets/{version_id}/images/{image_id}/file`
  - `POST /v1/datasets/{version_id}/rebuild`
  - `WS /v1/refinement/{version_id}/{image_id}?tenant_id=...`
  - `POST /v1/refinements/{session_id}/accept`
  - The existing generic refinement websocket remains available when an
    `image_loader` is supplied, so older adapters and tests still work.
- `application.py`
  - Loads dataset images into `RefinementTarget` values, wires the default
    `Sam3ClickRefinementProvider`, and exposes the dataset image root to the API.
- `pyproject.toml` / `uv.lock`
  - Added `websockets>=13,<16`; without a websocket implementation uvicorn
    rejects the browser upgrade request.

### Browser

- Dataset cards now have a **Review labels** action.
- The review panel lists images and existing label counts, loads the image with
  the required tenant header (fetch + blob), draws existing masks, opens the
  refinement websocket, and supports positive/negative click points.
- The previewed mask is drawn on a canvas overlay. **Save corrected mask**:
  1. saves the chosen prediction as an annotation revision,
  2. rebuilds a new dataset version,
  3. reloads the library and reports that the updated version is ready.
- No tenant input, UUID, or annotation ID is rendered.

## Acceptance evidence

Commands run from `agentic-cv-platform` with the base `.venv` (no GPU, no live
services; fake providers):

```text
.\.venv\Scripts\python -m pytest tests/test_manual_refinement_api.py -q
1 passed, 1 warning in 0.78s

.\.venv\Scripts\python -m pytest tests/test_manual_refinement_ui_playwright.py -q
1 passed in 7.22s

.\.venv\Scripts\python -m pytest tests/test_assets_ui_playwright.py tests/test_agent_ui_playwright.py tests/test_manual_refinement_ui_playwright.py -q
3 passed in 14.74s

.\.venv\Scripts\python -m pytest -q
239 passed, 10 skipped, 1 warning in 29.92s
```

The HTTP test proves:

```text
dataset images list + image bytes
→ websocket initial snapshot (category + prompt)
→ positive click snapshot
→ accept (annotation revision)
→ rebuild (new immutable version)
→ COCO export contains the corrected bbox
```

The Playwright test proves the same through the shipped UI:

```text
Agent run complete
→ Datasets → Review labels
→ select image
→ click canvas (mask overlay turns green)
→ Save corrected mask
→ "Correction saved as revision ..."
→ refined dataset card
→ Export COCO → pycocotools sees bbox [8, 8, 80, 48]
→ no UUID in visible text
→ no horizontal overflow
```

## Limitations

- Click refinement is **click-guided mask selection**, not point-conditioned
  segmentation. SAM 3 is prompted with the dataset category; clicks rank the
  candidate masks. If the model returns no candidate around the clicked object,
  the UI reports that no mask was found rather than inventing one.
- Accepting a correction replaces one existing annotation for the image/category,
  or creates the first annotation when the image has none. A multi-label image
  with several annotations needs a future annotation picker.
- Corrections create a new dataset version; old versions remain immutable.
- The browser test uses a fake refiner. A real-model correction run on the
  owner's photo set is still part of the P5/P6 measurement work.
- The review panel downloads COCO only. YOLO remains available through
  `/v1/dataset-exports`.

## P5 boundary

P5 is `feature/gpu-resource-manager`: measure the baseline first, then add model
serialization/lazy unload/batching, and prove the full journey on the 6 GB laptop
without OOM. P4 does not claim those performance properties.
