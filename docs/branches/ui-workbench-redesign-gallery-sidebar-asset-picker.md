# Agentic CV Platform: UI Workbench Redesign (Gallery, Sidebar, Asset Picker)

Redesigns the local web workbench for practical, gallery-first workflows inspired by dataset/gallery UIs (e.g. Roboflow-style asset browsing): sidebar navigation, visual asset gallery with representative thumbnails, asset picker for semantic search, and improved upload/result UX. Includes runtime bug fixes required for smooth local use.

## What this branch provides

- **Navigation & layout (`src/agentic_cv/web/index.html`, `app.css`, `app.js`)**
  - Persistent sidebar with section navigation: **Gallery**, **Upload**, **Predict**, **Search**, **Exports**, **Preview** (replaces single long page).
  - Dark, technical workspace styling with responsive behavior and accessible focus states.
- **Asset gallery with representative media**
  - Gallery view lists tenant assets as visual cards with generated thumbnails (not just metadata IDs/hashes), source/type info, and quick actions (copy ID).
  - Upload view accepts **image/video** sources and shows the completed upload as a visual result card; gallery refreshes after upload.
- **Improved feature integration**
  - **Semantic search** scopes via selectable asset picker (available tenant assets) instead of manual comma-separated asset IDs. Search results remain gallery-style cards.
  - Predict, dataset export, and label-preview event flows preserved with clearer sectioned UX.
- **Backend support (`asset_registry.py`, `postgres_asset_registry.py`, `api.py`)**
  - `GET /v1/assets` — list tenant assets with sources and `thumbnail_url`.
  - `GET /v1/assets/{asset_id}/thumbnail` — on-the-fly representative thumbnail (image resize or first video frame via OpenCV); accepts `X-Tenant-ID` header or `tenant_id` query param.
  - Registry `list_assets(...)` / `get_source(...)` added to SQLite and Postgres implementations and protocol.
- **Runtime bug fixes**
  - Upload UI sends multipart `part` field (fixes 422 on `PUT /v1/uploads/{id}/parts/1`).
  - Upload completion resolves relative paths before `file://` URI conversion (fixes 409 "relative path can't be expressed as a file URI" with relative `--data-dir`).
  - Added `/health` alias (and `/api-info` entry) for generic probes.

## Validation

- `python -m compileall -q src/agentic_cv tests` → OK.
- Targeted: `pytest -k "asset or upload or api"` → **33 passed, 3 skipped, 146 deselected**.
- Full default suite (no services): **172 passed, 10 skipped, 1 warning**.
- `git diff --check` → clean.
- Manual/in-process checks: asset list + thumbnail generation (image) OK; multipart upload/complete with relative data dir OK.

## Notes / limitations

- Semantic search requires embedding dependencies and indexed content (e.g. `torch`/embedding stack); if unavailable, the UI surfaces the API error.
- Thumbnails generated on-the-fly (no persistent thumbnail cache). Video thumbnails use first frame (OpenCV). Prediction endpoint remains image-oriented.
- Local MVP defaults unchanged (SQLite + local storage, no live external services required).

---
