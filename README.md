# Agentic CV Platform: Preview SSE API

This branch exposes durable batch progress as replayable Server-Sent Events. Clients can reconnect with `after_event_id` and receive only the tenant-authorized preview events they have not seen.

## Endpoint

`GET /v1/jobs/{job_id}/preview?after_event_id=0`

The endpoint requires `X-Tenant-ID`, returns `text/event-stream`, and emits standard SSE `id`, `event`, and JSON `data` fields. It replays the current durable backlog and closes; a later live-polling or broker-backed implementation can preserve this client contract.

## Local development

```text
uv run pytest -q
```

The SSE tests use temporary SQLite preview storage. No browser, GPU, broker, or cloud service is required.

## Branch boundary

This branch builds on `feature/durable-preview-events`. It provides replayable SSE delivery but does not yet hold open live subscriptions, stream overlay artifacts, authenticate users, or provide a gallery UI.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
