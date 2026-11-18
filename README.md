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

# Agentic CV Platform: PostgreSQL/pgvector Vector Index

Final store port for roadmap item 6. Semantic vector search can now run on PostgreSQL/pgvector behind the same index contract the semantic search service and API already use. SQLite/local in-memory indexing remains the default; PostgreSQL is opt-in through `AGENTIC_CV_POSTGRES_DSN`.

## What this branch provides

- `VectorIndex` protocol in `src/agentic_cv/semantic_search.py`; `LocalVectorIndex` (exact cosine search over normalized vectors, in-memory) remains the default implementation and `SemanticSearchService` now depends on the protocol.
- `src/agentic_cv/postgres_pgvector.py`: `PgVectorIndex` + `open_postgres_vector_index(config)` — durable exact cosine search using pgvector (`embedding <=> query`, score reported as `1 - distance`), with the same semantics as local: normalization, dimension consistency, asset filtering, `(-score, record_id)` ordering, limit validation, upsert/remove, and persistence across reopen.
- Composition-root wiring in `application.py`: when `AGENTIC_CV_POSTGRES_DSN` is set the app uses `PgVectorIndex`; otherwise `LocalVectorIndex`. The index is owned/closed by `LocalAppResources`, and the semantic API uses the selected index.
- Migration **v6** in `src/agentic_cv/postgres.py`: `vector_records.asset_id` changed from `UUID` to `TEXT` (additive `ALTER ... TYPE TEXT USING asset_id::text`) to match string domain asset IDs.

## Validation

- `tests/test_postgres_pgvector.py`:
  - shared index contract (upsert/search ranking and scores, asset filter, limit, remove, dimension/zero/limit validation) against `LocalVectorIndex` by default and `PgVectorIndex` when `AGENTIC_CV_TEST_POSTGRES_DSN` is set
  - local `SemanticSearchService` integration preserved
  - PostgreSQL persistence/reopen test (vectors and dimension survive reconnect)
- Focused (default): `tests/test_postgres_pgvector.py tests/test_semantic_search.py tests/test_semantic_api.py tests/test_application.py` → **13 passed, 2 skipped, 1 warning**.
- Live PostgreSQL 16.16 + pgvector (`cv-pg-test`, `127.0.0.1:5433/cvtest`): `tests/test_postgres_pgvector.py tests/test_postgres_annotation_dataset_stores.py tests/test_postgres_preview_subassets.py tests/test_postgres_job_store.py tests/test_postgres_outbox.py tests/test_postgres.py tests/test_postgres_asset_registry.py` → **29 passed**.
- `python -m compileall -q src/agentic_cv tests` → OK.
- Full default suite (no services): **154 passed, 10 skipped, 1 warning**.
- `git diff --check` → clean.

## Limitations

- pgvector search is exact (`ORDER BY <=>`) with no ANN index (e.g. HNSW/IVFFlat) yet; suitable for current local/small-scale use.
- The embedding column remains untyped `VECTOR` as defined in v1; per-dimension typing/index tuning can be added later.
- One connection per index, serialized by a lock; no pool yet.
- Mixed-backend boundaries from earlier slices remain: each store/index is independently selectable, but no cross-store transaction spans them.

## Roadmap status

- **Item 6 (Move metadata and vectors to PostgreSQL and pgvector): complete in this migration series** — assets, outbox, jobs, preview, sub-assets, annotations, dataset versions, and vector index are ported behind opt-in PostgreSQL/pgvector backends with shared contract tests; SQLite/local remains the default.
- **Next: Item 7 (frontend)** — begin the frontend slice and apply the `frontend-design` skill for a modern, polished UI.

---

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

# Agentic CV Platform: PostgreSQL Job Store

Fourth store port for roadmap item 6. Durable job state (leases, cancellation, retry counters, and recovery) can now run on PostgreSQL behind the same contract workers, workflows, and the export API already use. SQLite stays the default; PostgreSQL is opt-in through `AGENTIC_CV_POSTGRES_DSN`.

## What this branch provides

- `JobStoreBackend` protocol covering `create`, `get`, `list`, `transition`, `claim`, `heartbeat`, `requeue_expired_leases`, `request_cancel`, and `close`.
- `PostgresJobStore` with parity semantics:
  - idempotent `create` keyed by tenant + idempotency key, rejecting mismatched kind/payload
  - status transition guard (`ALLOWED_TRANSITIONS`) with attempt counting on `running`
  - leases: `claim`/`heartbeat` enforce ownership and expiry; queued claims increment attempts
  - recovery: `requeue_expired_leases` returns expired running jobs in `(created_at, sequence)` order, requeuing or cancelling when cancellation was requested
  - cancellation: queued jobs cancel immediately, running jobs mark `cancel_requested` and cancel on lease expiry
- `open_postgres_job_store(config)` — connects with dict rows, applies migrations, returns the store.
- Composition-root and worker wiring now select PostgreSQL for jobs (and outbox, as before) when `AGENTIC_CV_POSTGRES_DSN` is set; export worker runtime reads the same environment signal.
- Migration **v3** adds durable job ordering for PostgreSQL: `jobs.sequence BIGINT` with a per-schema sequence/default, backfilled for any pre-existing rows.

## Validation

- Shared contract test (`exercise_job_contract`) runs against SQLite by default and PostgreSQL when `AGENTIC_CV_TEST_POSTGRES_DSN` is set: create/idempotency, unknown-job `KeyError`, list filters, transitions/invalid transitions, claim/heartbeat/lease ownership, queued and running cancellation, and expired-lease recovery (requeue vs cancel-on-expiry).
- Default suite, no services: **148 passed, 4 skipped, 1 warning**.
- Live PostgreSQL 16.16 + pgvector: **17 passed** across foundation, asset registry, outbox, and job store contracts.
- One adaptation fix found by the live run: psycopg returns UUID columns as `uuid.UUID` objects, so `requeue_expired_leases` now normalizes IDs with `UUID(str(...))`.

## Limitations

- Mixed-backend deployment remains deliberate during migration: assets, outbox, and jobs can be in PostgreSQL while preview, sub-assets, annotations, dataset versions, and vectors remain SQLite. No cross-store transaction spans both.
- Dataset version and vector stores are still local; export/label workflows that touch those stores still rely on their current backends.
- One connection per store, serialized by a lock; no pool yet.

## Next boundary

Slice 5 — port preview and sub-asset stores, then annotations/dataset versions, then pgvector-backed vector search and wiring. This branch is still within roadmap item 6 (Postgres/pgvector migration), not item 7 (frontend).

---

# Agentic CV Platform: PostgreSQL Outbox

Second store port for roadmap item 6. The durable integration outbox can now run on PostgreSQL behind the same contract the APIs, dispatcher, and export workflow already use. SQLite stays the default; PostgreSQL is opt-in through `AGENTIC_CV_POSTGRES_DSN`.

## What this branch provides

- `OutboxBackend` protocol covering `enqueue`, `get`, `pending`, `mark_published`, `mark_failed`, and `close`.
- `PostgresOutbox` with the same idempotency rules: lookup by event ID first, then by tenant-scoped idempotency key; payloads stored as `JSONB` and returned as dictionaries.
- `open_postgres_outbox(config)` — connects with dict rows, applies migrations, returns the outbox.
- Composition-root selection mirrors the asset registry: DSN set → PostgreSQL, unset → SQLite.
- Migration **v2** adds a database sequence for delivery ordering:
  - `CREATE SEQUENCE outbox_event_sequence`
  - `outbox_events.sequence DEFAULT nextval(...)`
  - `setval` seeded from the current maximum so existing rows keep their order.

  Concurrent producers now get distinct sequence values without a `MAX(...) + 1` race.

## Validation

- Shared contract test (`exercise_outbox_contract`) runs against SQLite in the default suite and against PostgreSQL when `AGENTIC_CV_TEST_POSTGRES_DSN` is set: enqueue, sequence assignment, event-ID and idempotency-key deduplication, pending → publish, retry scheduling (`retry_after_seconds=0` retries now, `60` defers), unknown-event `KeyError`, limit validation.
- Tied-timestamp ordering is asserted on both backends with descending UUIDs, proving the sequence decides order rather than the UUID.
- Default suite, no services: **147 passed, 3 skipped, 1 warning** (two consecutive runs).
- Live PostgreSQL 16.16 + pgvector: **15 passed** across foundation, asset registry, and outbox contracts.
- One adaptation fix found by the live run: `psycopg` requires `Jsonb(...)` for dict parameters; the outbox now wraps payloads explicitly.

## Limitations

- Mixed-backend deployment: assets and outbox can be in PostgreSQL while jobs, preview, sub-assets, annotations, dataset versions, and vectors remain SQLite. No cross-store transaction spans both.
- The outbox sequence is per schema, which matches the single-database deployment model.
- One connection per outbox, serialized by a lock; no pool yet.
- The dispatcher is unchanged and backend-agnostic because it consumes the protocol.

## Next boundary

Slice 4 — port the job store (leases, cancellation, retry counters) with the same protocol/factory pattern, then the remaining stores, then pgvector search.

---

# Agentic CV Platform: PostgreSQL Asset Registry

First store port for roadmap item 6. The asset registry — assets, uploads, and sources — can now run on PostgreSQL behind the same contract the upload service already uses. SQLite remains the default; the switch is the opt-in `AGENTIC_CV_POSTGRES_DSN`.

## What this branch provides

- `AssetRegistryBackend` protocol: the five-method boundary shared by both registries.
- `PostgresAssetRegistry` implementing `create_asset`, `start_upload`, `get_upload`, and `complete_upload` with identical records, validation, tenant checks, and upload-deduplication semantics.
- `open_postgres_asset_registry(config)` — connects with dict rows, applies migrations, and returns the registry.
- Composition-root selection: a configured DSN builds the PostgreSQL registry; otherwise `AssetRegistry` on SQLite is unchanged.
- `LocalAppConfig.postgres`, read from the environment once at startup.

## Configuration

```text
export AGENTIC_CV_POSTGRES_DSN=postgresql://postgres:postgres@127.0.0.1:5433/cv
export AGENTIC_CV_POSTGRES_SCHEMA=public   # optional
uv sync --extra postgres
agentic-cv serve --data-dir .agentic-cv   # assets/uploads now in PostgreSQL
```

Leaving the DSN unset keeps the original local behaviour exactly.

## Shared contract test

`tests/test_postgres_asset_registry.py::exercise_asset_lifecycle` runs against both backends: asset creation and validation, unknown-asset rejection, cross-tenant rejection, upload start/get, completion, idempotent re-completion, and duplicate-digest deduplication. The SQLite half runs in the default suite; the PostgreSQL half runs when `AGENTIC_CV_TEST_POSTGRES_DSN` is set.

## Validation

- Default suite, no services and no driver required: **145 passed, 2 skipped, 1 warning** (two consecutive runs).
- Live PostgreSQL 16.16 + pgvector (`pgvector/pgvector:pg16`): **12 passed**, run twice, including migrations and the registry contract.
- The schema is dropped between live tests; the `vector` extension is verified to stay in `public`.

## Two defects found and fixed during live validation

- Installing the `vector` extension inside a per-tenant schema made it dangle when that schema was dropped, breaking the next `CREATE EXTENSION IF NOT EXISTS`. Extension bootstrap now runs with `SET search_path TO public` and `WITH SCHEMA public`.
- The migration runner indexed rows positionally, which broke under `dict_row`. It now accepts either tuple or mapping rows.

## Limitations

- Only the asset registry is ported. Jobs, outbox, preview, sub-assets, annotations, dataset versions, and vectors still use SQLite, so pointing at PostgreSQL yields a **mixed-backend** deployment. That is deliberate for a staged migration, but it means no cross-store transaction spans both backends.
- One connection per registry, serialized by a lock; no pool yet, so `SET search_path` is per-connection rather than per-checkout.
- No SQLite→PostgreSQL data migration or dual-write.
- `psycopg` is optional; a default install cannot use this path.

## Next boundary

Slice 3 — port the next store (outbox or jobs) with the same protocol-plus-factory pattern, keeping SQLite and its tests untouched. Then the remaining stores, vector search on pgvector, and a Qdrant boundary if requested.

---

# Agentic CV Platform: Deterministic Durable Ordering

This branch removes the remaining UUID tie-breaks from durable SQLite listings. It is hygiene, found while validating the PostgreSQL slice: a full-suite run failed `test_job_store.py::test_expired_leases_are_requeued_or_cancelled` once and passed on rerun.

## The defect

`JobStore.list`, `JobStore.requeue_expired_leases`, and `DatasetVersionStore.list` all ordered rows with `ORDER BY created_at, <id>`. `created_at` is written from the wall clock, whose granularity on Windows is coarse enough that consecutive inserts routinely share a timestamp. The tie was then resolved by a random UUID, so listing order and lease-recovery order varied between runs.

This is the same class of bug fixed for the outbox in `fix/outbox-ordering`, in the two stores that still had it.

## The fix

- Both tables gain an internal monotonic `sequence` column, assigned as `MAX(sequence) + 1` inside the existing write lock.
- Queries order by `created_at, sequence`, so tied timestamps keep insertion order.
- Existing database files are migrated on open: the column is added and backfilled from SQLite `rowid`.
- The column stays internal — `JobRecord` and `DatasetVersionRecord` keep their existing fields, so callers and serialized output are unchanged.
- `DatasetVersionStore.create` now uses an explicit column list instead of positional `VALUES`, which the added column requires.

## Validation

Deterministic regression tests, each shown to fail against the previous implementation and pass after:

- `test_list_preserves_creation_order_when_timestamps_tie` (jobs) — two jobs forced to one timestamp with UUIDs in reverse insertion order.
- `test_expired_lease_recovery_preserves_creation_order_when_timestamps_tie` — same forcing, both leased, then recovered.
- `test_legacy_jobs_table_is_migrated_and_keeps_insertion_order` — old-schema file opens, backfills, and lists in insertion order.
- `test_list_preserves_creation_order_when_timestamps_tie` (dataset versions).
- `test_legacy_dataset_table_is_migrated_and_keeps_insertion_order`.

Against the previous code: **5 failed, 9 passed**. With the fix: **14 passed**. Full suite: **142 passed, 1 skipped, 1 warning** — five consecutive runs, no failure.

## Limitations

- `sequence` is per table and per database file. It provides a total order for the local single-writer design, not across replicas or shards.
- Backfill uses `rowid`; rows deleted from a table are not renumbered.
- Ordering remains primarily by `created_at`; the sequence only decides ties.

## Next boundary

Back to roadmap item 6, slice 2: port the first SQLite store behind its existing interface to the PostgreSQL foundation from `feature/postgres-pgvector-foundation`.

---

# Agentic CV Platform: PostgreSQL + pgvector Foundation

This branch opens roadmap item 6: an opt-in PostgreSQL backend. It delivers the foundation slice only — configuration, a lazily-loaded driver boundary, and a versioned migration system with the full v1 platform schema (including a pgvector column). SQLite remains the default and no caller has been changed yet.

## Why this slice exists

The platform's durable state is SQLite. Production deployments need a shared database and an approximate-nearest-neighbour vector index, but the migration must not break local mode or force `psycopg` and a live server onto the default test run. This branch establishes the boundary; later slices port each store behind the interfaces it already exposes.

## What this branch provides

- `PostgresConfig` with `dsn` and a validated `schema` identifier.
- Opt-in configuration: `AGENTIC_CV_POSTGRES_DSN` (unset means SQLite, unchanged) and optional `AGENTIC_CV_POSTGRES_SCHEMA` (default `public`).
- `connect()` — lazy `psycopg` import with an actionable install message; sets the schema search path for the session.
- Ordered, unique, transactional migrations tracked in `<schema>.schema_migrations`.
- Idempotent application: re-running applies nothing.
- Schema-scoped DDL (`CREATE SCHEMA` + `SET search_path TO <schema>, public`), so a non-public schema does not silently write into `public`.
- v1 schema mirroring the SQLite stores: `assets`, `uploads`, `sources`, `jobs`, `outbox_events`, `annotation_revisions`, `sub_assets`, `dataset_versions`, `vector_records`.
- `payload`/`annotation`/`dataset` as `JSONB`, identifiers as `UUID`, timestamps as `TIMESTAMPTZ`, arrays as native `TEXT[]`, and `embedding VECTOR` via the `vector` extension.
- Indexes for tenant lookups, job status, vector asset filtering, and a partial pending-outbox index that preserves the `(occurred_at, sequence)` delivery order.

## Setup

```text
uv sync --extra postgres
```

`psycopg` is an optional extra, never a hard dependency. A local pgvector instance:

```text
docker run -d --name cv-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=cv -p 5433:5432 pgvector/pgvector:pg16
set AGENTIC_CV_POSTGRES_DSN=postgresql://postgres:postgres@127.0.0.1:5433/cv
```

(On Git Bash use `export`.) Nothing consumes the DSN inside the application yet.

## Validation

- `tests/test_postgres.py` — 9 unit tests with a recording fake connection: opt-in config, schema validation, migration ordering/uniqueness, first-run application, idempotent re-run, rollback on failure, version sorting, duplicate rejection, unsafe schema rejection.
- 1 opt-in integration test, skipped unless `AGENTIC_CV_TEST_POSTGRES_DSN` is set.
- Live run performed against `pgvector/pgvector:pg16` (PostgreSQL 16.16) on `127.0.0.1:5433`: **10 passed**, including the second `apply_migrations` returning `()` and the `embedding` column resolving to the `vector` type.
- Default suite (no live services, no `psycopg` requirement): **137 passed, 1 skipped**.

## Limitations

- No store adapter is ported yet; every caller still uses SQLite. The env var alone changes nothing at runtime.
- `embedding VECTOR` has no fixed dimension, so no HNSW/IVFFlat index is created yet. Pinning the dimension belongs to the vector-search slice (SigLIP2 default is 1152).
- No connection pool, retry policy, migration CLI, or SQLite→Postgres data migration.
- `SET search_path` is session-scoped; pooled connections must scope it per checkout when pooling is added.

## Next boundary

Slice 2 — port the first store behind its existing interface with the same opt-in pattern, keeping the SQLite implementation and tests untouched. Then remaining stores, then vector search, then a Qdrant-compatible boundary if the owner wants it.

---

# Agentic CV Platform: Deterministic Outbox Ordering

This branch fixes a nondeterministic delivery order in the durable local outbox. It is a hygiene slice, not a feature: no roadmap item is reordered.

## The defect

`Outbox.pending()` ordered work with `ORDER BY occurred_at, event_id`. Events enqueued in the same clock tick share an identical `occurred_at`, so the tie was broken by `event_id` — a random UUID. Delivery order (and `tests/test_outbox_dispatcher.py::test_dispatcher_records_failure_and_continues`, which asserts insertion order) could therefore flip between runs.

## The fix

- An explicit monotonic `sequence` column, assigned inside the existing write lock as `MAX(sequence) + 1`.
- Pending reads now use `ORDER BY occurred_at, sequence`, so equal timestamps preserve insertion order.
- `OutboxRecord.sequence` is exposed for inspection; existing fields, payloads, and public methods are unchanged.
- Tables created before this branch are migrated on open: the column is added and backfilled from SQLite `rowid`, which is insertion order for this append-only table.

## Validation

- `tests/test_outbox.py::test_pending_uses_insertion_order_when_timestamps_tie` — three events with one shared timestamp and deliberately descending UUIDs. Verified to fail against the previous implementation (`[1, 2, 3]` returned as `[3, 2, 1]`) and pass after the fix.
- `tests/test_outbox.py::test_legacy_outbox_without_sequence_column_is_migrated` — old-schema database opens, backfills, and keeps the legacy event ahead of a newly enqueued one.
- Focused suites (`test_outbox.py`, `test_outbox_dispatcher.py`) green; the previously flaky dispatcher test passed 10 consecutive runs.
- Full suite: **128 passed, 1 warning** (Starlette/httpx deprecation).

## Limitations

- `sequence` is per outbox database. It gives a total order for this single-writer, single-file local design, not across shards or Postgres-backed implementations.
- Migration backfills from `rowid`; if rows were ever deleted from the table, historical ordering follows the surviving rowids.

## Next boundary

Roadmap item 6 — PostgreSQL/pgvector behind the existing store and vector-index boundaries, starting with schema and migrations. That work ships on its own stacked branch.

---

# Agentic CV Platform: VLM Validation Adapter

This branch adds structured VLM validation for SAM 3 calibration. It validates the response schema, applies an explicit acceptance rubric, caps calls per run, and caches judgments by overlay, prompt, and model revision.

## What this branch provides

- Typed target-presence, completeness, false-positive, and mask-tightness judgments.
- Confidence and reasoning validation.
- Explicit acceptance criteria for calibration.
- Maximum VLM call budgets with a dedicated exception.
- Deterministic cache keys including model revision.
- Client/provider injection with a local Ollama implementation.

## Local Ollama

The adapter targets Ollama at `http://localhost:11434` and defaults to `qwen3-vl:4b`.
Start Ollama and pull the model before wiring it into calibration:

```text
ollama pull qwen3-vl:4b
```

`render_prediction_overlay` writes the source image, SAM3 masks, and score-labeled
boxes to a local PNG. `OllamaVlmClient` sends that artifact as base64 to `/api/chat`
and requests the structured validation schema. Its timeout, context size,
temperature, and keep-alive are configurable; tests use a mocked HTTP boundary
and do not require Ollama or GPU access.

`OllamaPromptEnhancer` uses the same local model to produce one to three concise
SAM3 noun phrases. On Ollama/Qwen responses where structured JSON is returned in
`message.thinking`, the adapter accepts that field when `message.content` is empty.

## Interactive refinement

When configured with an image loader and `ClickRefinementService`, the API exposes
`/v1/refinement/{image_id}` as a WebSocket. It sends an initial snapshot, accepts
revisioned `click` messages with positive or negative points, and returns updated
SAM3 predictions and masks.

`JobStore` persists worker lease ownership, heartbeat expiry, and cancellation
intent. Claiming and heartbeat operations reject stale or competing workers;
`DurableLabelWorkflow` claims a lease, renews it at image checkpoints, observes
cancellation, and emits a dedicated cancellation event.

API authentication is opt-in through `StaticApiKeyAuthorizer`. When configured,
asset and upload write operations require `X-API-Key`, bind the key to
`X-Tenant-ID`, and enforce the `writer` role. Semantic-search and preview reads
use the `reader` role when their optional authorizer is configured. The default
local APIs remain unauthenticated until an authorizer is supplied.

Production OIDC/JWT validation is available through `OidcJwtAuthorizer`. The
`serve` command enables it when `AGENTIC_CV_OIDC_ISSUER`,
`AGENTIC_CV_OIDC_AUDIENCE`, and `AGENTIC_CV_OIDC_JWKS_URL` are set. Tokens must
include `exp`, `iat`, `sub`, a UUID tenant claim, and the configured roles claim.

`RolePolicy` centralizes resource permissions: readers can search, inspect
previews, and read exports; writers can mutate assets, uploads, sub-assets, and
exports; admins receive the wildcard permission. API routes enforce these
resource-action permissions rather than trusting a coarse role label.

Cloud-compatible uploads use `S3ObjectStorage`. Set `AGENTIC_CV_S3_BUCKET` to
select it, optionally set `AGENTIC_CV_S3_ENDPOINT_URL` for MinIO and `AWS_REGION`
for AWS. When the bucket variable is absent, the local filesystem adapter remains
the default.

`OutboxDispatcher` delivers pending integration events through an injected
publisher, marks successful deliveries, and records retryable failures without
blocking later events in the same polling pass.

`RabbitMqPublisher` is the concrete broker publisher. It declares a durable topic
exchange and publishes persistent JSON event envelopes; set
`AGENTIC_CV_RABBITMQ_HOST` and the optional RabbitMQ connection variables to
activate it in the composition root. `RabbitMqConsumer` declares durable queues,
binds routing keys, limits in-flight messages with QoS, acknowledges successful
handlers, and requeues failures.

`split_dataset` creates deterministic, image-disjoint train, validation, and test
`DatasetVersion` objects from configurable ratios and a seed. Annotations remain
with their source images, so each partition can be passed directly to the existing
COCO or YOLO exporters.

`DatasetVersionStore` persists validated dataset snapshots in tenant-scoped SQLite
storage with reopen-safe reads, ordered listing, and idempotent creation.

`DatasetExportService` materializes persisted versions into tenant/version-scoped
COCO or YOLO directories and writes a manifest for each export.

`DatasetExportWorkflow` wraps those exports in durable leased jobs with idempotent
submission, cancellation handling, and completion/failure events.

`DatasetExportWorker.run_once()` polls queued export jobs and processes each through
the lease-aware workflow; one failed job does not prevent the remaining queue from
being inspected. `run_forever()` adds a stoppable polling loop for local worker
processes, and each polling pass recovers expired dataset-export leases before
claiming queued work.

When configured, the API exposes authenticated `POST /v1/dataset-exports` for job
submission, `GET /v1/dataset-exports/{job_id}` for tenant-scoped status reads, and
`POST /v1/dataset-exports/{job_id}/cancel` for cancellation requests.

## Local development

```text
uv run pytest -q
```

The validator tests use a fake VLM client. No multimodal model, GPU, network call, or cloud service is required.

## Run the assembled local MVP

Start the unified API with:

```text
uv run agentic-cv serve --data-dir .agentic-cv
```

It assembles the core asset/upload API, semantic-search routes, preview SSE,
dataset-export jobs, and lazy SAM3 service under one FastAPI process. The health
checks are available at `/health/live` and `/health/ready`; model weights are
loaded only when their inference or embedding endpoints are used.

The root URL serves the local browser dashboard. It provides asset upload, SAM3
prediction, semantic search, and export-job controls. JSON runtime discovery is
available at `/api-info`, and interactive API documentation remains at `/docs`.

For a containerized local stack, use:

```text
docker compose -f docker-compose.local.yml up --build
```

This starts the API, export worker, RabbitMQ, and MinIO services with persistent
local volumes. The host-based `agentic-cv serve` and `agentic-cv worker` commands
remain the lighter development path.

## Branch boundary

This branch builds on `feature/persistent-vector-index`. It provides a local
Ollama client, overlay renderer, prompt enhancer, calibration bridge, and optional
click-refinement WebSocket, plus opt-in tenant authorization, but does not yet
implement hosted deployment configuration, health-based orchestration, or worker
autoscaling.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
