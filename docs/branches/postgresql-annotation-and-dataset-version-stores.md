# Agentic CV Platform: PostgreSQL Annotation and Dataset-Version Stores

Sixth store port for roadmap item 6. Immutable annotation revisions and tenant-scoped dataset versions can now run on PostgreSQL behind the same contracts label workflows, export services, and APIs already use. SQLite stays the default; PostgreSQL is opt-in through `AGENTIC_CV_POSTGRES_DSN`.

## What this branch provides

- `AnnotationRevisionBackend` protocol in `src/agentic_cv/annotation_store.py` and `DatasetVersionBackend` protocol in `src/agentic_cv/dataset_store.py`; existing SQLite stores remain the defaults.
- `src/agentic_cv/postgres_annotation_store.py`: `PostgresAnnotationRevisionStore` + `open_postgres_annotation_store(config)` — optimistic-concurrency saves (`expected_revision`, `RevisionConflictError` on stale/race), immutable history, latest lookup, and validation; annotations stored as `JSONB`.
- `src/agentic_cv/postgres_dataset_store.py`: `PostgresDatasetVersionStore` + `open_postgres_dataset_version_store(config)` — tenant-scoped create/get/list with idempotency-key dedupe and conflict detection, dataset payloads stored as `JSONB`, ordering by `(created_at, sequence)`.
- Composition-root/worker wiring: `application.py` and `worker_runtime.py` select PostgreSQL dataset versions when `AGENTIC_CV_POSTGRES_DSN` is set (alongside assets, outbox, jobs, preview, sub-assets). Consumer types updated to backends in `dataset_export.py`, `dataset_export_workflow.py`, `worker_runtime.py`, and `label_workflow.py` (annotation/preview protocol types).
- Migration **v5** in `src/agentic_cv/postgres.py`: additively adds `dataset_versions.sequence` + per-schema sequence/default/backfill for deterministic listing (v1 lacked the column).

## Validation

- Shared contract tests (`tests/test_postgres_annotation_dataset_stores.py`) run against SQLite by default and PostgreSQL when `AGENTIC_CV_TEST_POSTGRES_DSN` is set:
  - annotations: save/history/latest, stale `RevisionConflictError`, unknown-annotation `KeyError`, author/expected-revision validation
  - dataset versions: create/get/list ordering, idempotent replay, conflicting-key rejection, blank name/key validation, cross-tenant `KeyError`
- Focused (default): `tests/test_postgres_annotation_dataset_stores.py tests/test_annotation_store.py tests/test_dataset_store.py tests/test_dataset_export.py tests/test_dataset_export_api.py tests/test_dataset_export_workflow.py tests/test_dataset_export_worker.py tests/test_label_workflow.py tests/test_application.py` → **27 passed, 2 skipped, 1 warning**.
- Live PostgreSQL 16.16 + pgvector (`cv-pg-test`, `127.0.0.1:5433/cvtest`): `tests/test_postgres_annotation_dataset_stores.py tests/test_postgres_preview_subassets.py tests/test_postgres_job_store.py tests/test_postgres_outbox.py tests/test_postgres.py tests/test_postgres_asset_registry.py` → **25 passed**.
- `python -m compileall -q src/agentic_cv tests` → OK.
- Full default suite (no services): **152 passed, 8 skipped, 1 warning**.
- `git diff --check` → clean.

## Limitations

- Mixed-backend deployment remains deliberate: metadata stores can be in PostgreSQL while the semantic vector index/search is still the local in-memory/SQLite-era implementation. No cross-store transaction spans both.
- Annotation store is not yet composed in `application.py` (no current app endpoint constructs it); the PG backend/factory is provided for label-workflow parity and future wiring.
- One connection per store, serialized by a lock; no pool yet.

## Next boundary

Slice 7 — **pgvector-backed vector index/search and wiring** to complete roadmap item 6 (PostgreSQL/pgvector migration). This branch is still within item 6, **not** item 7 (frontend). Frontend work will start after item 6 is complete and will use the `frontend-design` skill as requested.

---
