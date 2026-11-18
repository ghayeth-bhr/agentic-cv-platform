# Agentic CV Platform: PostgreSQL Preview and Sub-Asset Stores

Fifth store port for roadmap item 6. Durable batch-preview events and semantic sub-asset records can now run on PostgreSQL behind the same contracts the preview SSE API and semantic/sub-asset API already use. SQLite stays the default; PostgreSQL is opt-in through `AGENTIC_CV_POSTGRES_DSN`.

## What this branch provides

- `PreviewBackend` protocol in `src/agentic_cv/preview.py` and `SubAssetBackend` protocol in `src/agentic_cv/sub_assets.py`; existing SQLite stores remain the defaults.
- `src/agentic_cv/postgres_preview.py`: `PostgresPreviewStore` + `open_postgres_preview_store(config)` — tenant/job-scoped preview events with monotonic `BIGSERIAL` IDs, ordered replay via `after_event_id`, and the same progress validation semantics.
- `src/agentic_cv/postgres_sub_assets.py`: `PostgresSubAssetStore` + `open_postgres_sub_asset_store(config)` — query/parent/result-order persistence, unique-result validation, and tenant access enforcement; `parent_asset_ids`/`result_ids` stored as PostgreSQL `TEXT[]` (matching the schema) and returned as tuples.
- Composition-root selection in `application.py`: `AGENTIC_CV_POSTGRES_DSN` selects PostgreSQL for preview and sub-assets alongside the previously ported stores; API dependency types (`preview_api`, `semantic_api`) now accept backend protocols.
- Migration **v4** in `src/agentic_cv/postgres.py`: creates `preview_events` (with `(tenant_id, job_id, event_id)` index) for PostgreSQL deployments — table was missing from v1 and is added additively.

## Validation

- Shared contract tests (`tests/test_postgres_preview_subassets.py`) run against SQLite by default and PostgreSQL when `AGENTIC_CV_TEST_POSTGRES_DSN` is set:
  - preview: ordered/incremental replay, tenant scoping, progress/`after_event_id` validation, unknown-event `KeyError`
  - sub-assets: query/parent/result-order persistence, tenant access enforcement, empty/duplicate/blank-query rejection
- Focused (default): `tests/test_postgres_preview_subassets.py tests/test_preview.py tests/test_sub_assets.py tests/test_semantic_api.py tests/test_application.py` → **15 passed, 2 skipped, 1 warning**.
- Live PostgreSQL 16.16 + pgvector (`cv-pg-test`, `127.0.0.1:5433/cvtest`): `tests/test_postgres_preview_subassets.py tests/test_postgres_job_store.py tests/test_postgres_outbox.py tests/test_postgres.py tests/test_postgres_asset_registry.py` → **21 passed**.
- `python -m compileall -q src/agentic_cv tests` → OK.
- Full default suite (no services): **150 passed, 6 skipped, 1 warning**.
- `git diff --check` → clean.

## Limitations

- Mixed-backend deployment remains deliberate: assets, outbox, jobs, preview, and sub-assets can be in PostgreSQL while annotations, dataset versions, and the vector index/search remain SQLite/local.
- One connection per store, serialized by a lock; no pool yet.
- Preview events are append-only (no retention/compaction policy in this slice).

## Next boundary

Slice 6 — port annotation and dataset-version stores, then pgvector-backed vector index/search and wiring. This branch is still within roadmap item 6 (Postgres/pgvector migration), **not** item 7 (frontend). Frontend work will start after item 6’s remaining Postgres/pgvector work and will use the `frontend-design` skill as requested.

---
