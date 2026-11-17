# Agentic CV Platform: Persistent Vector Index

This branch adds a durable SQLite vector index for local semantic search. It preserves the existing exact cosine-search contract while storing normalized float32 embeddings across process restarts.

## What this branch provides

- Persistent vector records keyed by image record ID.
- Idempotent upsert and explicit removal.
- Dimension consistency checks across the index.
- Exact cosine ranking with asset filtering and bounded limits.
- Reopen-safe metadata and embedding storage.
- A replaceable local implementation before pgvector/Qdrant scale-out.

## Local development

```text
uv run pytest -q
```

The vector tests use tiny synthetic embeddings and temporary SQLite files. No embedding model, GPU, vector database, or cloud service is required.

## Branch boundary

This branch builds on `feature/preview-sse-api`. It provides durable local vectors but does not yet switch the semantic API to this index by default, add approximate nearest-neighbor search, or add model/checkpoint migration metadata.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
