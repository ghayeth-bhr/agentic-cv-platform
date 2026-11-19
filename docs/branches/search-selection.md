# Search-result selection → new asset

Branch: `feature/search-selection`
Base: `feature/agent-dialogue` (Phase 1)
Plan: [`docs/OPTIMIZATION_PLAN.md`](../OPTIMIZATION_PLAN.md) defect B10 (scheduled for Phase 5; done early at the owner's request).

## Why

Semantic search returns the N nearest images, whether or not they match. **Save results as sub-asset** re-ran the query and saved every result, so a search for elephants that returned 9 frames produced an asset with 9 images, 3 of them a dik-dik, a buffalo, and a gerenuk. The Phase 1 real-model run then calibrated on those frames and failed. The brief asks to *select* images from the search result and create an asset from the selection.

## Changes

**API** (`POST /v1/sub-assets/from-search`, `POST /v1/search`)
- `record_ids: [...]` saves **exactly** the selected results, in selection order, with duplicates dropped. Each record must exist and belong to the caller's tenant (404 otherwise). An empty selection returns 422. Scores are recomputed on the server (`SemanticSearchService.score_records`) rather than taken from the client.
- `name` sets the new asset's name. The default stays `Sub-asset: <query>`.
- `min_score` drops weak matches from `/v1/search` (and from the legacy save-everything path).
- `limit` is capped at 500.
- Search is now **scoped to the tenant's assets**. Before, a search with no asset filter scanned every record in the index, including other tenants'. Requested `asset_ids` outside the tenant match nothing.
- Requests without `record_ids` behave as before.

**UI** (Assets → Search the library)
- A **Results** count input.
- Every result card is a toggle (click, Space or Enter), with a checkbox. Unselected cards are dimmed and greyed. Results start fully selected: untick what does not belong.
- A selection bar with **N of M selected**, **Select all**, **Clear**, a **Min score** slider (cards below it fade), and **Select ≥ min score**.
- **New asset name** field and **Save N selected as asset**. The source-asset link moved to a small `↗` link so clicking the image toggles selection.

## Validation

```text
$ .venv/Scripts/python -m pytest -q
333 passed, 10 skipped, 1 warning
$ compileall / node --check / git diff --check   # ok
```

New `tests/test_search_selection.py`:
- Only the selected records are saved, in order and de-duplicated, with the custom name and 2 sources.
- Unknown records, other tenants' records, and an empty selection are rejected.
- Tenant-scoped search and `min_score`.
- A Playwright journey: search, untick, clear, select all, untick, name, save, and the new asset shows 2 sources.

The existing asset browser journey (search → save) still passes unchanged, because all results start selected. A headless screenshot with the 9 real wildlife frames confirmed the selection rendering.

## Limitations

- Results are capped by **Results** (default 20). To curate a large search, raise the count and use the min-score slider.
- Selection is kept per search and resets when you search again.
- A sub-asset is still a copy of the selected images, not a live filter of the parent.
