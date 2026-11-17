# Agentic CV Platform: Semantic Search API

This branch exposes semantic search and logical sub-asset creation over HTTP. It composes the provider-neutral search service with durable sub-asset metadata and deliberately omits local filesystem paths from responses.

## Endpoints

- `POST /v1/search` searches by natural-language query and optional asset IDs.
- `POST /v1/sub-assets/from-search` repeats the search and persists the ordered result set for a tenant.
- Search responses contain record IDs, asset IDs, and scores only.
- Sub-assets retain the original query and parent asset IDs without duplicating media.

## Local development

```text
uv run pytest -q
```

The API tests use a fake search service and temporary SQLite sub-asset storage. No embedding model, vector database, GPU, or cloud service is required.

## Branch boundary

This branch builds on `feature/click-refinement-sessions`. It does not yet add authentication middleware, persistent vector storage, pagination cursors, search indexing jobs, or sub-asset deletion policies.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
