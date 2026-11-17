# Agentic CV Platform: Durable Preview Events

This branch adds durable per-image progress events to the batch labeling workflow. Preview consumers can incrementally read tenant-scoped progress without coupling the UI directly to a worker process.

## What this branch provides

- SQLite-backed preview events with ordered IDs.
- Per-image processed/total progress and status.
- Incremental reads using `after_event_id`.
- Tenant and job filtering.
- Optional workflow callback integration after each processed image.
- Existing durable job, annotation, and outbox behavior remains unchanged.

## Local development

```text
uv run pytest -q
```

The preview tests use temporary SQLite files and fake SAM 3 providers. No model weights, GPU, broker, WebSocket server, or cloud service is required.

## Branch boundary

This branch builds on `feature/durable-label-workflow`. It persists progress but does not yet expose SSE/WebSocket streaming, preview image artifacts, cancellation, worker leases, or a browser gallery.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
