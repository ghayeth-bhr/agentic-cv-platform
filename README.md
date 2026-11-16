# Agentic CV Platform

A fresh local-first implementation of an agentic computer-vision platform for object detection and segmentation. Users will upload image/video assets, search them semantically, calibrate SAM 3 prompts, review labels, and export training-ready datasets.

## Baseline status

`main` is the clean project baseline. It contains the Python package scaffold, dependency metadata, and the first smoke test. Implementation work is intentionally developed in feature branches and merged only after focused tests and review.

## Development

```text
uv run pytest -q
```

The target local environment is Windows with an NVIDIA GPU. Production deployment will add cloud GPU workers, durable workflows, object storage, vector search, authentication, and multi-tenancy behind stable contracts.

## Branch workflow

- `feature/platform-foundation-contracts` defines shared task, artifact, and event schemas.
- `feature/local-resumable-upload` adds local asset metadata and resumable object storage.
- `feature/sam3-inference-adapter` adds lazy local SAM 3 inference.

Each feature branch has its own README describing its exact scope and validation status. The existing `frames-extractor` project is context only and is not part of this repository or Git history.
