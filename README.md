# Agentic CV Platform: Batch SAM 3 Labeling

This branch adds the whole-asset labeling worker. It applies a calibrated SAM 3 prompt and threshold to image records, preserves raw predictions, and converts boxes or masks into validated detection/segmentation annotations.

## What this branch provides

- Batch labeling over multiple image artifacts.
- Raw prediction preservation with score, box, and mask data.
- Detection annotations from SAM 3 boxes.
- Segmentation polygons extracted from SAM 3 masks with OpenCV contours.
- Dataset validation before a batch result is returned.
- Explicit prompt, threshold, task, and empty-batch validation.

## Local development

```text
uv run pytest -q
```

The batch tests use a fake provider and tiny CPU images. No gated model weights, GPU, cloud service, or batch queue is required.

## Branch boundary

This branch builds on `feature/semantic-search-api`. It does not yet persist raw prediction artifacts, create durable batch jobs, stream previews, or connect output revisions to the annotation store.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
