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
