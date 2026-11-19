# P10 — dataset gallery and live labeling view

Branch: `feature/dataset-gallery-live-view`  
Base: `48a4259` (P8 merged into `main`)  
Status: implementation and deterministic gates complete; pending owner review.

## Scope

P10 adds a browsable dataset gallery and a live per-image view inside the Agent
run panel. The shipped P0–P8 upload, search, Agent, refinement, export, and
download journey remains intact. No default branch or unowned work was changed.

## Changes

### Dataset gallery

- Dataset cards are clickable and open `#/datasets/<version_id>`. A back link
  returns to `#/datasets`; browser back/forward works through `hashchange`.
- The gallery renders a filmstrip, a large viewer with previous/next buttons,
  left/right arrow-key navigation, and an `image N of M` counter.
- Overlays are drawn on a canvas: detection boxes, segmentation polygons,
  stable per-class colors, class names, and scores. Boxes, masks, and labels
  can be toggled independently.
- Filters cover class, split, images without annotations, and low confidence
  (`score < 0.50`). Class chips show annotation counts per class.
- The side panel shows file name, size, split, dimensions, and each
  annotation's class, score, and pixel area. `Refine this image` opens the
  existing click-refinement panel on the same image.
- Cards show a first-four-image thumbnail mosaic, counted class chips, source
  asset name, creation time, and status.
- `GET /v1/datasets/{id}/images` now accepts `limit`/`offset`, returns
  `total`, `limit`, `offset`, and `has_more`, and includes `split` and
  `file_size`. The gallery pages through results with `Load more images`.
- Dataset summaries expose `source_asset`, `status`, and `class_counts`.
  `source_asset` is parsed from the recorded dataset name; `status` is `ready`
  for any stored immutable version. Display splits are derived deterministically
  from `image_id` (80% train, 10% validation, 10% test) because the dataset
  record does not persist a split. They are display filters, not exported COCO
  metadata.

### Agent live view

- `Live view` appears in the run panel when a whole-asset label job starts.
  It shows a `Live` badge, a done/total progress bar, and a grid that fills as
  each image completes.
- The browser consumes the existing replayable SSE endpoint
  `GET /v1/jobs/{id}/preview`, tracking `event_id` as a cursor. Reconnects
  replay only unseen events, so already displayed images are never lost. A
  dropped stream shows a clear reconnecting notice.
- `GET /v1/label-jobs/{id}/preview/{image_id}/image` renders the source image
  with the latest committed annotations for that job. Annotation revisions
  are now saved per image from the labeler's `on_image` callback, which makes
  mid-run overlays possible without waiting for the whole job.
- When the dataset version is ready, the live badge stops, and the panel offers
  `Open gallery` into the version's deep link. Cancel remains available while
  the run is active.

## Deterministic gates

```text
$ uv run pytest -q
282 passed, 10 skipped, 1 warning in 55.22s

$ uv run pytest tests/test_dataset_gallery_playwright.py -q
2 passed in 26.50s

$ node --check src/agentic_cv/web/app.js
(no output; exit 0)

$ python -m compileall -q src tests
(no output; exit 0)

$ uv lock --check
Resolved 90 packages in 1ms

$ git diff --check
(no output; exit 0)

$ uv run agentic-cv doctor
Summary: 11 ok, 0 warn, 0 fail
```

The 10 skips are the opt-in PostgreSQL tests. New coverage includes
`tests/test_dataset_gallery_api.py` (pagination, summary fields, SSE replay,
live overlay PNG) and `tests/test_dataset_gallery_playwright.py` (navigation,
filters, toggles, deep link, refine handoff, and live grid filling with a
deterministic fake provider).

## Caveats

- Live preview overlays read the latest annotation revisions from the
  annotation store. A manually refined image after the run will change that
  revision; the dataset version remains immutable until explicitly rebuilt.
- The SSE endpoint is a durable replay stream that closes after the current
  snapshot. The client reconnects with a cursor; no in-memory event bus was
  added.
- Real VLM calibration was not re-run for P10. P8 remains authoritative for
  real-model reliability, timing, and memory measurements in
  [`docs/PERFORMANCE.md`](../PERFORMANCE.md).
