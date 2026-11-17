# Agentic CV Platform: Local Asset Upload API

This branch connects the local asset registry, resumable object storage, and durable event outbox through a FastAPI API. It is the first usable HTTP flow from asset creation to a completed source and `SourceReady` event.

## Endpoints

- `POST /v1/assets` creates an asset for the `X-Tenant-ID` UUID.
- `POST /v1/assets/{asset_id}/uploads` starts a checksum-aware upload session.
- `PUT /v1/uploads/{upload_id}/parts/{part_number}` stores a resumable part.
- `POST /v1/uploads/{upload_id}/complete` assembles the object, registers the source, and enqueues `SourceReady`.

## Local development

```text
uv run pytest -q
```

The API tests use temporary SQLite metadata, local object storage, and a local outbox. No cloud credentials, broker, GPU, or model weights are required.

## Branch boundary

This branch builds on `feature/local-event-outbox`. Tenant identity is represented by a local header for now; production must replace it with authenticated identity and authorization middleware. The API does not yet expose signed URLs, upload cancellation, rate limits, video extraction, or workflow dispatch.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
