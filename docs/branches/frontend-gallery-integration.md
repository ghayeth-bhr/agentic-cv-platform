# Agentic CV Platform: Frontend Gallery Integration

Integrates gallery-style browsing into the static frontend workbench (`/`, `/static`): a durable preview-event gallery for jobs and a gallery presentation for semantic search results. Frontend-only slice (no backend API changes).

## What this branch provides

- **Preview gallery panel** (`web/index.html`, `app.css`, `app.js`): loads and replays `GET /v1/jobs/{job_id}/preview` SSE event streams for the current tenant into a contact-sheet grid. Supports job ID entry, optional `after_event_id` cursor, "use export job ID" convenience, and clear. Renders per-image event cards (image ID, status pill, processed/total progress, created time, event ID) with aggregate summary (events, processed/total, latest status, job). Handles empty/error responses gracefully.
- **Search results as gallery**: semantic search results render as a responsive gallery grid of result cards (record ID, asset, score, score bar) instead of a plain list, integrating browsing into the same workbench.
- **Layout/styling:** workbench grid extended with a full-width gallery area; gallery/search cards styled to match the existing technical/editorial workbench aesthetic (rule-based, mono metadata, status/progress indicators).

## Validation

- `python -m compileall -q src/agentic_cv tests` → OK.
- Full default suite (no services): **161 passed, 10 skipped, 1 warning**.
- `git diff --check` → clean.
- No automated frontend tests in repo; changes are static assets only and do not affect Python/API behavior.

## Limitations

- Preview gallery depends on jobs that emit durable preview events (e.g. batch/label workflow previews) for the selected tenant; export jobs typically do not emit preview events.
- No backend image-serving/thumbnails or asset/job listing APIs were added; gallery renders event/search metadata available from existing endpoints.
- Frontend is a static workbench served by the app; no separate build step or frontend test harness added.

## Roadmap status

- **Item 6 (PostgreSQL/pgvector): complete** (prior slices).
- **Item 6b (orchestrator + calibration/label workflow API wiring): complete** (prior slice).
- **Item 6c (end-to-end smoke regression): complete** (prior slice).
- **Frontend/gallery integration: delivered in this slice.**

---
