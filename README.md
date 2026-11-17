# Agentic CV Platform: Sub-Asset Persistence

This branch turns semantic-search results into durable logical sub-assets. A sub-asset stores its tenant, parent asset IDs, original query, and ordered result IDs without duplicating image bytes.

## What this branch provides

- SQLite persistence for logical sub-assets.
- Query snapshots for reproducible search-result collections.
- Ordered result IDs and deduplicated parent asset IDs.
- Tenant-scoped reads and access checks.
- Validation for empty searches and duplicate result IDs.
- Restart persistence through the local metadata database.

## Local development

```text
uv run pytest -q
```

The tests use temporary SQLite files and synthetic search results. No embedding model, vector database, GPU, or cloud service is required.

## Branch boundary

This branch builds on `feature/huggingface-embedding-provider`. It persists sub-asset metadata but does not yet expose search/sub-asset HTTP endpoints, persist vectors, or implement collaborative edits and lifecycle deletion policies.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
