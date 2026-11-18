# Agentic CV Platform: Frontend Workbench (Roadmap Item 7)

Modern, polished local-first frontend for the Agentic CV Platform, built as a static workbench served by the FastAPI application (`GET /`, assets at `/static`). Designed as a precision "light table" for CV/ML workflows — clear structure, IBM Plex typography, restrained ink-and-cyan palette, and responsive panels — following the `frontend-design` skill principles.

## What this branch provides

- Rebuilt `src/agentic_cv/web/index.html`, `app.css`, `app.js` as a cohesive single-page workbench (no bundler) covering the platform's user-facing flows:
  - **Runtime & tenant context:** live runtime status via `/api-info`, tenant-scoped requests (`X-Tenant-ID`).
  - **Ingest asset:** create asset, start resumable upload, PUT part, complete upload with SHA-256 hashing and result details (`/v1/assets`, `/v1/assets/{id}/uploads`, `/v1/uploads/{id}/parts/1`, `/v1/uploads/{id}/complete`).
  - **Segment with SAM3:** prompt-guided prediction (`/v1/predict`) with image preview, bounding-box overlay on canvas, per-detection score/box summary, and raw JSON response.
  - **Semantic search:** query the indexed frames with optional asset scoping/limit (`/v1/search`) and create a sub-asset from the current search (`/v1/sub-assets/from-search`).
  - **Dataset export:** queue, inspect, and cancel export jobs for a dataset version (`/v1/dataset-exports`, `/v1/dataset-exports/{job_id}`, `/v1/dataset-exports/{job_id}/cancel`).
- Accessible, responsive layout (sticky masthead, workbench grid collapsing to single column), explicit loading/error/success states, and client-side request handling with no external runtime dependencies beyond loaded fonts.

## Validation

- `python -m compileall -q src/agentic_cv tests` → OK.
- Full default suite (no services): **154 passed, 10 skipped, 1 warning**.
- `git diff --check` → clean.

## Limitations

- Static HTML/CSS/JS only; no build step or component framework. Depends on the browser's Web Crypto API for SHA-256 during upload.
- SAM3 prediction visualization overlays bounding boxes; full mask raster overlay is not rendered (raw mask data remains in the response).
- Uses the APIs exposed by the composed local app; flows that require pre-existing data (e.g. indexed vectors for search, dataset versions for export) depend on those resources being present.
- No dedicated frontend test harness added in this slice.

## Roadmap status

- **Item 6 (Move metadata and vectors to PostgreSQL and pgvector): complete** (prior slices).
- **Item 7 (frontend): in progress/delivered as this workbench slice** — a modern, polished UI built with `frontend-design` guidance on top of the existing platform APIs.

---
