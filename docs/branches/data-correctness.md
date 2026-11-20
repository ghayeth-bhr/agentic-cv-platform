# Phase 0 — data correctness

Branch: `fix/data-correctness`
Base: `ecb1854` (`feature/dataset-gallery-live-view`)
Plan: [`docs/OPTIMIZATION_PLAN.md`](../OPTIMIZATION_PLAN.md), Phase 0 (defects B1–B7, B14, B15).

## Why

Several defects silently corrupted datasets or exports. This branch fixes them before any new agent features are built on top.

## Changes

| Defect | Fix | Where |
|---|---|---|
| B1: a second label run on the same asset inherited the first run's labels | **Annotation sets.** Label runs write `job-<job_id>/<image_id>:<n>`. A dataset version records its `annotation_set` and builds and rebuilds read only that set. Manual corrections write into the version's set. | `annotation_sets.py`, `batch_labeling.py`, `label_workflow.py`, `dataset_builder.py` |
| B1 (retry) | A retried job retires annotations an earlier attempt wrote for the same image but the new attempt did not produce. | `label_workflow.py` |
| B2: false positives could not be removed | **Tombstone revisions** (`deleted: true` in the stored annotation JSON, no schema change) and `DELETE /v1/datasets/{version_id}/images/{image_id}/annotations/{annotation_id}` with optimistic `expected_revision`. The next `/rebuild` drops the annotation. | `annotation_store.py`, `postgres_annotation_store.py`, `api.py` |
| B3: accepting a click overwrote the first annotation of the class | No `annotation_id` now means **add a new object**. Replacing requires naming the annotation, which must belong to the version's set. The review UI had the same bug (it always sent the first annotation). It now replaces only the annotation whose box contains the positive click and adds a new one otherwise. | `api.py`, `web/app.js` |
| B4: YOLO class index = `category_id` (1-based) | Contiguous 0-based indices, with `nc` and 0-based `names` in `data.yaml`. | `dataset.py`, `dataset_export.py` |
| B5: `val` == `train`, no splits | **Persisted, source-aware splits** (80/10/10). Frames of one video stay in one split. YOLO exports use `images/{train,val,test}` + `labels/...`. COCO exports keep `annotations.json` (each image tagged with `split`) and add `annotations/instances_<split>.json`. The gallery shows the same split as the export. Legacy versions get the same deterministic assignment. | `dataset.py`, `dataset_export.py`, `api.py` |
| COCO used string ids | Integer `id`/`image_id`. The durable ids are kept as `image_uid`/`annotation_uid`. `area` is the polygon area, and `score` is included. | `dataset.py` |
| B6: multi-part masks became one self-intersecting YOLO polygon, and polygons were bloated | `approxPolyDP` simplification, speckle removal (the largest part always kept), and Ultralytics-style zero-area bridging of parts into one YOLO-seg line. | `batch_labeling.py`, `dataset.py` |
| B7: videos cut at 120 frames, extracted twice, and decoded on every resolve | **Time-based sampling** (`AGENTIC_CV_VIDEO_SAMPLE_FPS`, default 1) with `AGENTIC_CV_VIDEO_MAX_FRAMES` (default 1000, 0 = unlimited; a warning is logged when reached), and **one cached frame set** per source shared by search and labeling (`derived/frames/…/frames.json`). The cache is invalidated by video size/mtime or a config change. | `video_frames.py`, `label_images.py`, `semantic_indexing.py` |
| B14: dataset source parsed from its display name | Versions record `source_asset_ids`. Summaries resolve names from the registry. | `dataset.py`, `dataset_store.py`, `api.py` |
| B15: SQLite without WAL/busy timeout | `connect_sqlite()` sets WAL, `synchronous=NORMAL`, and a 5 s busy timeout for every store. | `sqlite_utils.py` + 9 stores |
| Hygiene | `HANDOFF.md` (personal details) and `.agentic-cv-demo/` are git-ignored. | `.gitignore` |

## Compatibility

- Stored dataset versions without the new keys load unchanged. New keys are written only when set, so the idempotency encodings of existing versions don't change.
- Versions created before this branch have no set. Their rebuilds read only unscoped (legacy) annotations, which is the previous behaviour minus contamination from newer runs.
- Videos indexed before this branch use the old frame ids (stride 30). Re-index an asset to move it to time-based frames. Until then, calibration falls back to evenly spaced samples for frames without vectors.
- Export archive layout changed for YOLO (split folders). COCO keeps `annotations.json` at the root.

## Validation

```text
$ .venv/Scripts/python -m pytest -q
299 passed, 10 skipped, 1 warning

$ .venv/Scripts/python -m compileall -q src/agentic_cv tests   # ok
$ git diff --check                                             # ok
$ node --check src/agentic_cv/web/app.js                       # ok
```

New: `tests/test_data_correctness.py` (17 tests) covers two runs on one asset, an empty second run, retry leftovers, legacy rebuilds, delete then rebuild over HTTP, click-to-add over HTTP, source-aware persisted splits, the YOLO split layout and 0-based classes, re-export cleanup, COCO integer ids and per-split files (loaded with `pycocotools`), polygon area, multi-part merge, simplification IoU, 10-minute video coverage at 1 fps, the frame cache, and the SQLite pragmas. Four existing tests were updated because they asserted the old behaviour: the 1-based YOLO class, the flat YOLO/COCO layout, and unscoped annotation ids.

## Limitations

- The gallery lists the immutable version, so a deletion appears after **Rebuild**. A live working-set review view is Phase 4.
- There is no delete button in the UI yet (API only). It arrives with the Phase 4 review UI.
- Real-model runs were not repeated for this branch. The changes are in data handling, not inference.

## Next boundary

Phase 1: `feature/agent-dialogue` (multi-turn slot-filling agent, multi-asset, multi-class, prompt agent wired in).
