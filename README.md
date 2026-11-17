# Agentic CV Platform: Dataset Annotation Export

This branch defines validated annotation records and deterministic COCO/YOLO exporters for detection and segmentation. It gives calibrated or manually edited predictions a stable training-dataset representation.

## What this branch provides

- Immutable image, category, and annotation records.
- Bounding-box validation against image dimensions.
- Polygon validation for segmentation labels.
- COCO JSON export with images, categories, boxes, masks, areas, and crowd flags.
- YOLO detection export with normalized box coordinates.
- YOLO segmentation export with normalized polygon coordinates.
- Explicit failure for unknown references, invalid geometry, and missing segmentation polygons.

## Local development

```text
uv run pytest -q
```

The exporter tests are CPU-only and require no images on disk, model weights, database server, or cloud service.

## Branch boundary

This branch builds on `feature/sam3-calibration-loop`. It exports in-memory dataset versions but does not yet persist annotation revisions, assign train/validation/test splits, stream previews, or create downloadable archives.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
