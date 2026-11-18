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
