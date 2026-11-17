# Agentic CV Platform: Annotation Revisions

This branch adds immutable annotation history with optimistic concurrency. Manual edits, reviewer corrections, and future point-click refinements create new revisions instead of overwriting raw predictions.

## What this branch provides

- SQLite-backed annotation revisions.
- Author and creation-time provenance for every edit.
- Revision numbers starting at one for each tenant/image/annotation.
- Stale-write rejection through `expected_revision`.
- Latest-revision reads and complete history reads.
- JSON round-tripping of boxes, polygons, categories, and scores.

## Local development

```text
uv run pytest -q
```

The tests use temporary SQLite files and validated in-memory annotations. No editor UI, GPU, model, or cloud service is required.

## Branch boundary

This branch builds on `feature/sub-asset-persistence`. It persists revision history but does not yet expose annotation HTTP endpoints, support deletion/tombstone revisions, resolve collaborative merges, or connect click refinement sessions.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
