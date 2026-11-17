# Agentic CV Platform: Durable Label Workflow

This branch connects batch SAM 3 labeling to durable job state, annotation revisions, and the event outbox. A run now has explicit success/failure state and emits a completion event after annotations are persisted.

## What this branch provides

- Idempotent batch-label job submission.
- Tenant ownership checks before execution.
- `queued` to `running` to `succeeded`/`failed` lifecycle.
- Annotation revision persistence for successful predictions.
- `LabelRunCompleted` and `LabelRunFailed` outbox events.
- Failure details retained on the job record.

## Local development

```text
uv run pytest -q
```

The workflow tests use a fake SAM 3 provider and temporary SQLite stores. No gated model weights, GPU, broker, or cloud service is required.

## Branch boundary

This branch builds on `feature/batch-sam3-labeling`. It does not yet provide worker leases, cancellation, progress events, raw prediction artifact storage, or a distributed workflow engine.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
