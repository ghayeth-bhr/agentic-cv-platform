# Agentic CV Platform: PostgreSQL Job Store

Fourth store port for roadmap item 6. Durable job state (leases, cancellation, retry counters, and recovery) can now run on PostgreSQL behind the same contract workers, workflows, and the export API already use. SQLite stays the default; PostgreSQL is opt-in through `AGENTIC_CV_POSTGRES_DSN`.

## What this branch provides

- `JobStoreBackend` protocol covering `create`, `get`, `list`, `transition`, `claim`, `heartbeat`, `requeue_expired_leases`, `request_cancel`, and `close`.
- `PostgresJobStore` with parity semantics:
  - idempotent `create` keyed by tenant + idempotency key, rejecting mismatched kind/payload
  - status transition guard (`ALLOWED_TRANSITIONS`) with attempt counting on `running`
  - leases: `claim`/`heartbeat` enforce ownership and expiry; queued claims increment attempts
  - recovery: `requeue_expired_leases` returns expired running jobs in `(created_at, sequence)` order, requeuing or cancelling when cancellation was requested
  - cancellation: queued jobs cancel immediately, running jobs mark `cancel_requested` and cancel on lease expiry
- `open_postgres_job_store(config)` — connects with dict rows, applies migrations, returns the store.
- Composition-root and worker wiring now select PostgreSQL for jobs (and outbox, as before) when `AGENTIC_CV_POSTGRES_DSN` is set; export worker runtime reads the same environment signal.
- Migration **v3** adds durable job ordering for PostgreSQL: `jobs.sequence BIGINT` with a per-schema sequence/default, backfilled for any pre-existing rows.

## Validation

- Shared contract test (`exercise_job_contract`) runs against SQLite by default and PostgreSQL when `AGENTIC_CV_TEST_POSTGRES_DSN` is set: create/idempotency, unknown-job `KeyError`, list filters, transitions/invalid transitions, claim/heartbeat/lease ownership, queued and running cancellation, and expired-lease recovery (requeue vs cancel-on-expiry).
- Default suite, no services: **148 passed, 4 skipped, 1 warning**.
- Live PostgreSQL 16.16 + pgvector: **17 passed** across foundation, asset registry, outbox, and job store contracts.
- One adaptation fix found by the live run: psycopg returns UUID columns as `uuid.UUID` objects, so `requeue_expired_leases` now normalizes IDs with `UUID(str(...))`.

## Limitations

- Mixed-backend deployment remains deliberate during migration: assets, outbox, and jobs can be in PostgreSQL while preview, sub-assets, annotations, dataset versions, and vectors remain SQLite. No cross-store transaction spans both.
- Dataset version and vector stores are still local; export/label workflows that touch those stores still rely on their current backends.
- One connection per store, serialized by a lock; no pool yet.

## Next boundary

Slice 5 — port preview and sub-asset stores, then annotations/dataset versions, then pgvector-backed vector search and wiring. This branch is still within roadmap item 6 (Postgres/pgvector migration), not item 7 (frontend).

---
