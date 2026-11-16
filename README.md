# Agentic CV Platform: Resumable Local Uploads

This branch establishes the local asset-ingestion foundation. It connects a tenant-scoped SQLite registry to a filesystem object-storage abstraction, allowing uploads to resume after interrupted chunk transfers before a source is registered in an asset.

## What this branch provides

- Asset, upload, and source metadata with SQLite foreign-key enforcement.
- Tenant ownership checks before an upload can begin.
- Out-of-order resumable parts with atomic replacement of individual chunks.
- Contiguous-part validation and streaming SHA-256 verification at completion.
- Atomic assembly into an object path before source metadata is committed.
- Duplicate-source prevention using `(asset_id, content_sha256)`.

## Local development

Run all tests:

```text
uv run pytest -q
```

The upload tests use temporary SQLite databases and temporary object roots. No cloud credentials or external services are required.

## Storage model

Incomplete parts live below `.uploads/<upload_id>/`. Completed objects live below `objects/` using generated tenant and asset prefixes. Metadata stores object URIs and content hashes rather than loading media into the database.

## Branch boundary

This is a local reduced-mode implementation. The storage interface is intentionally small so it can later be backed by S3/MinIO multipart uploads. It does not yet extract video frames, index embeddings, or publish durable workflow events.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
