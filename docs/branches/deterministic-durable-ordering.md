# Agentic CV Platform: Deterministic Durable Ordering

This branch removes the remaining UUID tie-breaks from durable SQLite listings. It is hygiene, found while validating the PostgreSQL slice: a full-suite run failed `test_job_store.py::test_expired_leases_are_requeued_or_cancelled` once and passed on rerun.

## The defect

`JobStore.list`, `JobStore.requeue_expired_leases`, and `DatasetVersionStore.list` all ordered rows with `ORDER BY created_at, <id>`. `created_at` is written from the wall clock, whose granularity on Windows is coarse enough that consecutive inserts routinely share a timestamp. The tie was then resolved by a random UUID, so listing order and lease-recovery order varied between runs.

This is the same class of bug fixed for the outbox in `fix/outbox-ordering`, in the two stores that still had it.

## The fix

- Both tables gain an internal monotonic `sequence` column, assigned as `MAX(sequence) + 1` inside the existing write lock.
- Queries order by `created_at, sequence`, so tied timestamps keep insertion order.
- Existing database files are migrated on open: the column is added and backfilled from SQLite `rowid`.
- The column stays internal — `JobRecord` and `DatasetVersionRecord` keep their existing fields, so callers and serialized output are unchanged.
- `DatasetVersionStore.create` now uses an explicit column list instead of positional `VALUES`, which the added column requires.

## Validation

Deterministic regression tests, each shown to fail against the previous implementation and pass after:

- `test_list_preserves_creation_order_when_timestamps_tie` (jobs) — two jobs forced to one timestamp with UUIDs in reverse insertion order.
- `test_expired_lease_recovery_preserves_creation_order_when_timestamps_tie` — same forcing, both leased, then recovered.
- `test_legacy_jobs_table_is_migrated_and_keeps_insertion_order` — old-schema file opens, backfills, and lists in insertion order.
- `test_list_preserves_creation_order_when_timestamps_tie` (dataset versions).
- `test_legacy_dataset_table_is_migrated_and_keeps_insertion_order`.

Against the previous code: **5 failed, 9 passed**. With the fix: **14 passed**. Full suite: **142 passed, 1 skipped, 1 warning** — five consecutive runs, no failure.

## Limitations

- `sequence` is per table and per database file. It provides a total order for the local single-writer design, not across replicas or shards.
- Backfill uses `rowid`; rows deleted from a table are not renumbered.
- Ordering remains primarily by `created_at`; the sequence only decides ties.

## Next boundary

Back to roadmap item 6, slice 2: port the first SQLite store behind its existing interface to the PostgreSQL foundation from `feature/postgres-pgvector-foundation`.

---
