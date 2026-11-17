# Agentic CV Platform: Video Frame Extraction

This branch adds the CPU media worker that turns a video source into timestamped JPEG frame artifacts. It preserves frame provenance while keeping capture and image-writing seams injectable for deterministic testing.

## What this branch provides

- Configurable frame stride for sampling videos.
- Optional maximum frame limit for bounded local jobs.
- Frame index, timestamp, and output-path provenance records.
- Atomic cleanup through capture release on success or failure.
- Explicit failures for unavailable videos and failed image writes.
- Headless OpenCV runtime dependency for server-side processing.

## Local development

```text
uv run pytest -q
```

The extractor tests use a fake capture and writer, so they do not require a real video, camera, GUI, GPU, or cloud service. Real extraction uses `opencv-python-headless` and writes JPEG artifacts to the configured output directory.

## Branch boundary

This branch builds on `feature/dataset-annotation-export`. It does not yet persist frame records into the asset registry, extract keyframes using motion policies, generate thumbnails, or enqueue embedding jobs after extraction.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
