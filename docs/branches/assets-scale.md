# Phase 5 — assets and search at scale

Branch: `feature/assets-scale`
Base: `feature/label-monitor` (Phase 3)
Plan: [`docs/OPTIMIZATION_PLAN.md`](../OPTIMIZATION_PLAN.md), Phase 5 (selection of search results and saving the selection was delivered earlier in `feature/search-selection`).

## Changes

**Vector search: in-memory matrix with top-k** (`persistent_vectors.py`)
- Every query used to read all rows from SQLite and score them in Python: **~160 ms at 20k vectors**, linear in library size.
- Now a normalized float32 matrix is built once and reused, with `argpartition` top-k. Ties at the cut are kept, so ordering by record id stays exact.
- Measured at 20k × 768: **~2 ms per query**. The first query after a write rebuilds the matrix in ~125 ms.
- The cache is invalidated on this connection's writes and when another process commits (SQLite `PRAGMA data_version`), for example a separate worker.

**Thumbnail cache** (`api.py`)
- Asset thumbnails were regenerated on every request (videos decoded each time).
- They are now generated once per source into `<data>/derived/thumbnails/<source_id>.jpg`. Sources are immutable, so this is safe.
- Responses carry an `ETag` and a one-day private cache.

**Original file names**
- Dataset images are stored as `<source-id>.jpg` (unique, stable export names), which the UI displayed as "this item.jpg".
- `GET /v1/datasets/{v}/images` and `/review` now include `source_name`: the uploaded file name, or `video.mp4 · frame N` for video frames. The gallery caption and details use it.
- Export file names are unchanged.

**Multi-asset selection** (Assets view)
- A checkbox on every asset card. The ticked set is the **same set the Agent uses**.
- A selection bar with **N selected**, **Use in Agent**, **Search these**, **Clear**.
- New search scope **Ticked assets (N)**: one search across several chosen assets. The scope chip shows the asset count.
- **Use in Agent** opens the Agent with exactly those assets ticked, as a new task. If a run is in progress it says so instead.

## Real-model check

The 9 wildlife frames as two assets ("Clip A" with 5 frames, "Clip B" with 4), real SigLIP in the browser:
- Ticking Clip A and searching "an elephant" returned only Clip A's 5 frames, in 0.09 s.
- The three elephants ranked first (14%, 14%, 13%), then the buffalo (7%) and the dik-dik (5%). A min-score of about 10% selects exactly the elephants.

## Validation

```text
$ .venv/Scripts/python -m pytest -q
366 passed, 10 skipped, 1 warning
$ compileall / node --check / git diff --check   # ok
```

New tests:
- The vector cache sees its own writes, deletes, and writes from a second connection. Asset filter and tie order are preserved.
- Thumbnails are cached per source.
- Dataset and review images carry the original file names.
- Playwright: tick two of three assets, then "Search these" (scope chip "2 assets", results only from them), then "Use in Agent" (two assets ticked, the plan names both).

## Limitations

- The matrix holds every vector in RAM (about 3 KB per 768-d vector, so roughly 300 MB at 100k). Beyond that, use the opt-in pgvector store or an ANN index.
- SigLIP scores are raw cosine similarities, so they look low (0.05–0.15). Only their order and the gaps matter; the min-score slider works on the same scale.
- The thumbnail cache is never pruned (sources are not deleted today).
