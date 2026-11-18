# Agentic CV Platform: PostgreSQL Outbox

Second store port for roadmap item 6. The durable integration outbox can now run on PostgreSQL behind the same contract the APIs, dispatcher, and export workflow already use. SQLite stays the default; PostgreSQL is opt-in through `AGENTIC_CV_POSTGRES_DSN`.

## What this branch provides

- `OutboxBackend` protocol covering `enqueue`, `get`, `pending`, `mark_published`, `mark_failed`, and `close`.
- `PostgresOutbox` with the same idempotency rules: lookup by event ID first, then by tenant-scoped idempotency key; payloads stored as `JSONB` and returned as dictionaries.
- `open_postgres_outbox(config)` — connects with dict rows, applies migrations, returns the outbox.
- Composition-root selection mirrors the asset registry: DSN set → PostgreSQL, unset → SQLite.
- Migration **v2** adds a database sequence for delivery ordering:
  - `CREATE SEQUENCE outbox_event_sequence`
  - `outbox_events.sequence DEFAULT nextval(...)`
  - `setval` seeded from the current maximum so existing rows keep their order.

  Concurrent producers now get distinct sequence values without a `MAX(...) + 1` race.

## Validation

- Shared contract test (`exercise_outbox_contract`) runs against SQLite in the default suite and against PostgreSQL when `AGENTIC_CV_TEST_POSTGRES_DSN` is set: enqueue, sequence assignment, event-ID and idempotency-key deduplication, pending → publish, retry scheduling (`retry_after_seconds=0` retries now, `60` defers), unknown-event `KeyError`, limit validation.
- Tied-timestamp ordering is asserted on both backends with descending UUIDs, proving the sequence decides order rather than the UUID.
- Default suite, no services: **147 passed, 3 skipped, 1 warning** (two consecutive runs).
- Live PostgreSQL 16.16 + pgvector: **15 passed** across foundation, asset registry, and outbox contracts.
- One adaptation fix found by the live run: `psycopg` requires `Jsonb(...)` for dict parameters; the outbox now wraps payloads explicitly.

## Limitations

- Mixed-backend deployment: assets and outbox can be in PostgreSQL while jobs, preview, sub-assets, annotations, dataset versions, and vectors remain SQLite. No cross-store transaction spans both.
- The outbox sequence is per schema, which matches the single-database deployment model.
- One connection per outbox, serialized by a lock; no pool yet.
- The dispatcher is unchanged and backend-agnostic because it consumes the protocol.

## Next boundary

Slice 4 — port the job store (leases, cancellation, retry counters) with the same protocol/factory pattern, then the remaining stores, then pgvector search.

---
