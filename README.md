# Agentic CV Platform: Local Event Outbox

This branch adds a durable SQLite outbox for integration events. Producers persist an `EventEnvelope` before a future broker relay publishes it, preventing transient delivery failures from silently losing asset, upload, or workflow events.

## What this branch provides

- Durable event envelopes with tenant and aggregate identity.
- Idempotent insertion by event ID or tenant-scoped idempotency key.
- Ordered pending-event reads with a bounded batch size.
- Publish completion timestamps that remove events from pending delivery.
- Retry attempt counts, last-error details, and retry scheduling metadata.
- SQLite synchronization suitable for the local single-process mode.

## Local development

Run the complete test suite:

```text
uv run pytest -q
```

The outbox tests use temporary SQLite files. No Kafka, Redpanda, broker, GPU, or cloud service is required.

## Branch boundary

This branch builds on `feature/local-job-state`. It persists and exposes delivery state but does not yet run a relay worker, claim leases between processes, or publish to a production broker. The next messaging slice should add a leased relay and dead-letter policy while keeping this outbox as the local source of truth.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
