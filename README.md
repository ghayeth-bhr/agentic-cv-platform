# Agentic CV Platform: Click Refinement Sessions

This branch adds the interactive point-click refinement state needed by semi-manual labeling. Positive and negative points are validated against the image, sent to an injected SAM 3 click provider, and returned with a revisioned prediction snapshot.

## What this branch provides

- Interactive refinement sessions scoped to an image.
- Positive and negative point validation.
- Image-bound coordinate checks.
- Provider injection for the future SAM 3 point API adapter.
- Revision increments after every accepted click.
- Stale revision rejection for concurrent editor updates.
- Immutable snapshots containing points and current predictions.

## Local development

```text
uv run pytest -q
```

The tests use a fake click provider and CPU images. No gated model weights, GPU, editor UI, or cloud service is required.

## Branch boundary

This branch builds on `feature/annotation-revision-store`. Sessions are currently process-local; they do not yet persist session state, expose WebSocket endpoints, or convert accepted masks into annotation revisions automatically.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
