# Agentic CV Platform: Semantic Search Index

This branch adds the local semantic-search boundary for image and video-frame records. It normalizes provider embeddings, performs exact cosine search, supports asset filters, and keeps the storage/index implementation replaceable by pgvector or Qdrant later.

## What this branch provides

- Provider-neutral image and text embedding interface.
- Idempotent vector upserts by record ID.
- Dimension and zero-vector validation.
- Exact cosine ranking with deterministic tie breaking.
- Asset-scoped search filters and bounded result limits.
- Image loading through the same local frame-artifact paths produced by the media worker.

## Local development

```text
uv run pytest -q
```

The tests use a deterministic fake embedding provider and tiny local images. No CLIP/SigLIP checkpoint, GPU, vector database, or cloud service is required.

## Branch boundary

This branch builds on `feature/video-frame-extraction`. The index is in-memory for reduced local mode; it does not yet persist embeddings, expose search over HTTP, create sub-assets from results, or add a real CLIP/SigLIP provider.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
