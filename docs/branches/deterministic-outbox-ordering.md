# Agentic CV Platform: Deterministic Outbox Ordering

This branch fixes a nondeterministic delivery order in the durable local outbox. It is a hygiene slice, not a feature: no roadmap item is reordered.

## The defect

`Outbox.pending()` ordered work with `ORDER BY occurred_at, event_id`. Events enqueued in the same clock tick share an identical `occurred_at`, so the tie was broken by `event_id` — a random UUID. Delivery order (and `tests/test_outbox_dispatcher.py::test_dispatcher_records_failure_and_continues`, which asserts insertion order) could therefore flip between runs.

## The fix

- An explicit monotonic `sequence` column, assigned inside the existing write lock as `MAX(sequence) + 1`.
- Pending reads now use `ORDER BY occurred_at, sequence`, so equal timestamps preserve insertion order.
- `OutboxRecord.sequence` is exposed for inspection; existing fields, payloads, and public methods are unchanged.
- Tables created before this branch are migrated on open: the column is added and backfilled from SQLite `rowid`, which is insertion order for this append-only table.

## Validation

- `tests/test_outbox.py::test_pending_uses_insertion_order_when_timestamps_tie` — three events with one shared timestamp and deliberately descending UUIDs. Verified to fail against the previous implementation (`[1, 2, 3]` returned as `[3, 2, 1]`) and pass after the fix.
- `tests/test_outbox.py::test_legacy_outbox_without_sequence_column_is_migrated` — old-schema database opens, backfills, and keeps the legacy event ahead of a newly enqueued one.
- Focused suites (`test_outbox.py`, `test_outbox_dispatcher.py`) green; the previously flaky dispatcher test passed 10 consecutive runs.
- Full suite: **128 passed, 1 warning** (Starlette/httpx deprecation).

## Limitations

- `sequence` is per outbox database. It gives a total order for this single-writer, single-file local design, not across shards or Postgres-backed implementations.
- Migration backfills from `rowid`; if rows were ever deleted from the table, historical ordering follows the surviving rowids.

## Next boundary

Roadmap item 6 — PostgreSQL/pgvector behind the existing store and vector-index boundaries, starting with schema and migrations. That work ships on its own stacked branch.

---
