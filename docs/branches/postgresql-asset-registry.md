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
