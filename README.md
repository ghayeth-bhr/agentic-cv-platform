# Agentic CV Platform: Foundation Contracts

This branch is the schema foundation for the fresh platform implementation. It defines the versioned contracts shared by the API, workers, agents, artifact storage, and event transport.

## Contracts in this branch

- `TaskSpec` validates detection and segmentation intent, target objects, visual attributes, calibration sample size, and iteration limits.
- `ArtifactManifest` identifies immutable external bytes by tenant, asset, content hash, model revision, and storage URI.
- `EventEnvelope` standardizes event type, aggregate identity, correlation, causation, idempotency, payload, and schema version metadata.

All contracts are frozen dataclasses with deterministic dictionary/JSON serialization. Invalid task types, empty targets, unsupported schema versions, and malformed SHA-256 values fail early.

## Local development

```text
uv run pytest -q
```

The tests are CPU-only and require no model weights, database server, object store, or cloud credentials.

## Branch boundary

This branch contains contracts only. It does not persist records, run inference, or expose HTTP endpoints. Later branches should consume these types rather than inventing parallel payload formats.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
