# feature/assets-experience (P2)

## Scope

Make the product usable from the browser without asking the owner to paste UUIDs:

1. Replace the old Gallery/Upload/Quick test/Search/Exports/Label jobs workbench with one local workspace and two primary sections, **Assets** and **Agent**, plus a small **Datasets / Jobs** area.
2. Support multi-file uploads with browser-side SHA-256 hashing and 5 MB chunked PUTs into the existing resumable-upload API.
3. Queue durable search-index jobs after upload instead of relying on a synchronous endpoint.
4. Show asset detail (sources, media type, dates) and search inside one asset or across all assets.
5. Show sub-asset lineage (query, parent asset names, matched-image count) and materialize results as a real gallery asset.
6. Cover upload → auto-index → in-asset search → sub-asset creation through a real browser and real HTTP server.

The Agent section is present as a clearly labelled next-phase placeholder. It does not pretend to chat yet.

## What changed

- `web/index.html`, `web/app.js`, `web/app.css`: replaced the UUID-driven views with a sidebar containing Assets, Agent, Datasets, and Jobs. The fixed local workspace tenant is a JS constant, never rendered. All asset/search/sub-asset/dataset/job labels are human-readable; errors are stripped of UUIDs before display.
- `web/app.js`: multi-file selection, SHA-256 hashing via Web Crypto, chunked uploads, index-job polling, asset filtering, asset detail, in-asset search scope, sub-asset save, dataset COCO export, and job list.
- `index_worker.py`: new durable `index-asset` worker. It requeues expired leases, claims queued jobs, calls `AssetSemanticIndexer`, persists `succeeded`/`failed`, and logs durations.
- `semantic_api.py`: added `POST /v1/assets/{id}/index-jobs`, `GET /v1/index-jobs/{id}`, and `GET /v1/sub-assets`. Sub-asset responses now include parent asset names, matched-image count, and creation time.
- `sub_assets.py` / `postgres_sub_assets.py`: sub-asset records can be linked to their materialized gallery asset, then looked up by asset or listed for a tenant. SQLite migrates old tables in place; PostgreSQL migration 7 adds the column and partial unique index.
- `api.py`: added `GET /v1/jobs` for tenant-scoped background-job lists.
- `application.py`: starts an index worker alongside label/export workers and wires the index worker into the semantic API.
- `rbac.py`: reader/writer gain `jobs:read`.
- `pyproject.toml` / `uv.lock`: added `playwright` to the dev group. Browser binaries are installed separately and are not required for the default test suite.

## Validation (real output)

```text
node --check src/agentic_cv/web/app.js
.venv/Scripts/python -m compileall -q src tests
git diff --check
uv lock --check

.venv/Scripts/python -m pytest -q
222 passed, 10 skipped, 1 warning in 15.17s
```

The 10 skips are the existing opt-in PostgreSQL tests. The browser test also skips with a clear message if Chromium is not installed.

The P2 gate test is `tests/test_assets_ui_playwright.py`:

- starts a real uvicorn server on localhost with `FakeEmbeddingProvider`
- opens the shipped UI in headless Chromium
- asserts there is no tenant input and no `X-Tenant-ID` text
- selects two PNGs, fills an asset name, uploads both into one asset
- waits for the durable auto-index job to report `indexed for search`
- opens asset detail, switches search scope to the selected asset, and searches for `red`
- saves the results as a sub-asset and asserts the new card plus lineage (`Parent assets: Site photos`, `Created from search: “red”`)
- asserts the rendered page text contains no UUID

Backend coverage added:

- `tests/test_index_worker.py`: success, persisted failure, and expired-lease requeue.
- `tests/test_sub_assets.py`: asset linking, tenant-scoped lookup/listing, and old-table migration.
- `tests/test_semantic_api.py`: index-job submit/read and cross-tenant 404.
- `tests/test_application.py`: tenant-scoped `/v1/jobs` list.

## Limitations

- The browser still cannot create a label job or build a dataset. Datasets are listed and exported; the Agent chat (P3) is the intended entry point for labeling.
- The browser test uses a fake embedder and fake image content. Real SigLIP indexing, SAM 3 labeling, and the owner's 6 GB laptop journey are still owner-machine gates.
- Video uploads are indexed through the existing frame extractor, but the detail panel shows the video source thumbnail rather than a frame browser.
- The upload UI hashes whole files in browser memory before chunking. Very large videos will use proportionally more browser memory.
- Removing the old Quick test view removes the browser smoke path for SAM 3; the `/v1/assets/{id}/predict` API remains available for P4 refinement work.
- `agentic-cv worker` still runs only the export worker, not label or index workers.
- The Playwright package is installed by `uv sync`, but Chromium is not. Run `.venv\Scripts\python -m playwright install chromium` to execute the browser gate.

## Next boundary

P3 `feature/agent-chat`: add a real Agent chat in the existing shell, turn a natural-language task into a task spec, confirm it, run calibration, label the whole selected asset, build a dataset, and surface download links. P2's asset selection, upload, index, search, and sub-asset state are the inputs for that flow.
