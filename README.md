# Agentic CV Platform: Local Job State

This branch adds the durable local control-plane state needed to resume work after a process restart. Jobs are stored in SQLite and can be safely created again by an idempotency key without duplicating work.

## What this branch provides

- Tenant-scoped job records with JSON payloads.
- `queued`, `running`, `succeeded`, `failed`, and `cancelled` states.
- Explicit allowed transitions that reject invalid state changes.
- Attempt counting when work enters `running`.
- Failure details and retry transition from `failed` back to `queued`.
- Idempotent creation for repeated API or worker requests.

## Local development

Run the complete test suite:

```text
uv run pytest -q
```

The job-store tests use temporary SQLite files and reopen the database to verify persistence. No broker, workflow server, GPU, or cloud service is required.

## Branch boundary

This branch depends on `feature/sam3-inference-service`. It provides persistence primitives, not a worker loop, lease expiry, event outbox, or distributed queue. Those behaviors should be added behind this store rather than encoded in the HTTP service.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
