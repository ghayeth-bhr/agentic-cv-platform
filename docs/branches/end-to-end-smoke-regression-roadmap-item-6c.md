# Agentic CV Platform: End-to-End Smoke Regression (Roadmap Item 6c)

Adds a permanent in-process end-to-end smoke regression test covering the core local workflow chain: upload → search/label → VLM calibration → export. Runs with default `pytest` using SQLite/local stores and fake providers (no live services).

## What this branch provides

- `tests/test_e2e_smoke_regression.py`: `test_end_to_end_smoke_upload_search_label_vlm_calibration_export`
  - **Upload:** creates an asset and completes a resumable upload via `AssetRegistry` + `LocalObjectStorage` + `ResumableUploadService`, verifying source registration and content SHA-256.
  - **Search:** indexes the frame in `SemanticSearchService`/`LocalVectorIndex` (fake embedding provider) and verifies semantic query retrieval for the asset.
  - **Label:** submits and executes `DurableLabelWorkflow` (batch labeling, fake SAM3 provider) to produce persisted annotation revisions/events and a labeled `DatasetVersion`.
  - **VLM calibration:** runs `Sam3Calibrator` with `VlmCalibrationValidator`/`VlmValidator` backed by a fake VLM client to verify the VLM-judged calibration path and selected threshold.
  - **Export:** persists the labeled dataset version and executes `DatasetExportWorkflow` (COCO), verifying a succeeded export job and written export artifacts.

## Validation

- Focused: `tests/test_e2e_smoke_regression.py` → **1 passed**.
- `python -m compileall -q src/agentic_cv tests` → OK.
- Full default suite (no services): **161 passed, 10 skipped, 1 warning**.
- `git diff --check` → clean.

## Limitations

- In-process/component-composed smoke test (not a multi-process/container deployment test); uses fake SAM3/VLM/embedding providers and local stores so it remains deterministic and service-free by default.
- Exercises the durable workflow APIs/logic and core chain; it does not require a live Ollama VLM, real SAM3 weights, or external Postgres/broker services.

## Roadmap status

- **Item 6 (PostgreSQL/pgvector): complete** (prior slices).
- **Item 6b (orchestrator + calibration/label workflow API wiring): complete** (prior slice).
- **Item 6c (end-to-end smoke regression): delivered in this slice.**
- **Next: Frontend/gallery integration** (per agreed roadmap order, after 6c).

---
