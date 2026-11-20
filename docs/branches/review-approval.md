# Phase 4 — review, approval, and real point refinement

Branch: `feature/review-approval`
Base: `feature/calibration-live` (Phase 2b)
Plan: [`docs/OPTIMIZATION_PLAN.md`](../OPTIMIZATION_PLAN.md), Phase 4.

## Why

The brief asks to navigate the annotated images in the gallery, approve selected images or all of them, refine detections by clicking with SAM, and export once approved. Before this phase:

- There was no approval state.
- The gallery showed the immutable version, so corrections appeared only after a rebuild.
- Wrong masks could only be removed through the API.
- "Click refinement" re-ran SAM 3's text prompt and picked the nearest mask, so it could not add a missed object or isolate one instance.

## 4a — review model (API)

| Endpoint | Purpose |
|---|---|
| `GET /v1/datasets/{v}/review?status=&offset=&limit=` | Images with their **live** annotations (corrections, additions and deletions visible at once), split, and review status, plus counts. |
| `POST /v1/datasets/{v}/review` | `{status, image_ids}` or `{status, all: true, only_status?}`, e.g. "approve all pending". |
| `PATCH /v1/datasets/{v}/images/{i}/annotations/{a}` | Move an annotation to another class (optimistic `expected_revision`). |
| `POST /v1/datasets/{v}/rebuild-approved` | New version with approved images only (`include_pending` optional). |

Review status is stored per **annotation set** (`review_store.py`, `reviews.sqlite3`), not per immutable version, so approvals carry over to rebuilt versions. Versions created before annotation sets existed use `version-<id>` as their key.

## 4b — point-prompted segmentation

- `sam3_points.py`: `Sam3TrackerModel` (SAM 3's point/box head) loaded lazily under the shared GPU manager as `sam3-tracker`.
- The image embedding is cached per refinement image. All positive and negative clicks form one object, and the best of 3 candidate masks (by predicted IoU) is used.
- It is the default refinement provider when the app runs real models. It falls back automatically to the ranked text-mask mode if the tracker cannot load; `AGENTIC_CV_REFINEMENT=ranked` forces the old mode.
- The refinement websocket accepts `?format=polygons`, which sends simplified polygons instead of a full boolean mask grid (about 920 k values per 720×1280 click).
- Accept takes an optional `category_id`, so a new object can belong to any class of the dataset.

Measured on an RTX 4050 Laptop (6 GB), 720×1280 frame: tracker load 5.8 s and about 0.9 GB, embedding 0.78 s once per image, **clicks 0.03–0.15 s**. One click isolated an elephant calf from its mother (predicted IoU 0.98).

## 4c — review in the dataset gallery

- The gallery reads from `/review`, so it always shows live annotations and status.
- A **status filter** (all / pending / approved / rejected), counts, and ✓/✗ badges on filmstrip thumbnails.
- **Approve / Reject** for the current image (buttons, or keys **A** and **R**), moving to the next image. ← and → navigate.
- **Bulk:** checkboxes on thumbnails, *Select shown*, *Approve/Reject N selected*, *Approve all pending*.
- **Per annotation:** select it (highlighted on the image), change its class (dropdown when the dataset has more than one class), delete it (✕).
- **Click mode:** click the object (Shift+click for background). The tracker mask previews in yellow with the click points. Choose the class, then **Save object**: it adds a new annotation, or replaces the selected one.
- **Build approved dataset** creates the approved-only version and opens it.

## Bugs found and fixed

- **Duplicate DOM id `gallery-status`** (Assets view and dataset gallery): every dataset-gallery status message went to the hidden Assets view. The dataset gallery now uses `dataset-gallery-status`, and a contract test fails on any duplicate id.
- **Overlay misalignment on letterboxed images.** The gallery image uses `object-fit: contain` with a max height, but the overlay canvas stretched over the whole box, so on portrait frames masks were drawn (and clicks would land) in the wrong place. The canvas is now laid exactly over the displayed pixels, and redrawn on resize.

## Real-model validation

On the elephants dataset from the Phase 2b run, with real models in the browser:
- The overlay lines up on the portrait frames.
- One click on the mother elephant produced a tight mask separate from the calf (first click 8.4 s including tracker load).
- Saving took 0.07 s (6 → 7 annotations), and approving updated the counts.

## Validation

```text
$ .venv/Scripts/python -m pytest -q
354 passed, 10 skipped, 1 warning
$ compileall / node --check / git diff --check   # ok
```

New tests:
- `tests/test_review_api.py`: live view, approve/reject/bulk/reset, approved-only rebuild keeping approvals and corrections, class change with conflict, legacy versions.
- `tests/test_sam3_points.py`: best-mask choice, embedding cache, adding an object the text prompt missed, fallback.
- `tests/test_review_ui_playwright.py`: the full gallery review journey (buttons, keyboard, filters, bulk reject, delete, click-to-add, approve pending, build approved).

## Limitations

- Review status is SQLite-only (no Postgres store yet).
- Click mode handles points. Box prompts (drag a rectangle) are supported by the tracker but not wired into the UI yet.
- The "classic review panel" (dataset card) still exists and uses the old mask format. The gallery supersedes it.
- File names in the gallery show the stored `<source-id>.jpg`, which the UI masks as "this item". Showing the original upload name is a Phase 5/6 polish item.
