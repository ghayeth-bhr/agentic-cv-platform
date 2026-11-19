# P6 — UX, end-to-end quality, and contracts

Branch: `feature/ux-e2e-quality` (stacked on the pushed P5 `feature/gpu-resource-manager`).

## Goal

Make the product's states explicit and keep the browser and API from drifting:
loading, empty, and recoverable error states in the shipped UI; broader browser
journeys; and contract tests for the routes, response envelopes, and DOM ids the
UI depends on.

## What changed

### Loading states

- The Assets gallery paints three skeleton cards and sets
  `aria-busy="true"` while the library request is open.
- The Agent asset picker shows `Loading assets…` and stays disabled while the
  library is loading.
- Datasets and Jobs render a `data-state="loading"` panel and set `aria-busy`
  until their requests settle.

### Empty states

- Assets: `No assets yet.` with an **Upload media** action that opens the upload
  panel; a filtered result offers **Clear filter**.
- Datasets: `No dataset versions yet.` with an **Open Agent** action.
- Jobs: `No background jobs yet.` with an **Open Assets** action.
- The Agent view explains that assets must be uploaded before a task can be
  planned and keeps **Plan task** disabled.

### Error states

- Assets, Datasets, and Jobs render an inline error panel with the backend's
  `detail` message and a **Try again** button.
- Error text is passed through the existing UUID scrubber, so a failure cannot
  leak an internal identifier into the page.
- Added the missing `[hidden] { display: none !important; }` rule. Before this,
  `.card { display: flex; }` overrode the HTML `hidden` attribute on the upload
  panel. The browser test now verifies that closing the panel really hides it.

### Contract tests

- `tests/test_ui_api_contracts.py`
  - every HTTP route and method used by `app.js` exists under its exact path
    template;
  - the dataset-targeted refinement WebSocket route exists;
  - empty `/v1/assets`, `/v1/jobs`, `/v1/datasets`, `/v1/sub-assets`, and
    `/v1/agent/sessions` envelopes are stable;
  - `/api-info` exposes the model-runtime resource status;
  - API errors use the `detail` field the UI reads;
  - every `$("id")` lookup in `app.js` has a matching `id` in `index.html`, and
    sidebar navigation stays `Assets`, `Agent`, `Datasets`, `Jobs`.

### Browser coverage

- `tests/test_ux_states_playwright.py` holds `/v1/assets` requests to assert the
  skeleton and Agent loading option, then releases them and verifies the empty
  states and their navigation actions.
- A second journey fails `/v1/assets`, `/v1/datasets`, and `/v1/jobs` at the
  browser route layer, verifies the error panels and messages, retries each one,
  and confirms the underlying empty state appears.
- Existing Assets, Agent, and manual-refinement journeys remain green.

## Acceptance evidence

```text
.\.venv\Scripts\python -m pytest tests/test_ux_states_playwright.py tests/test_assets_ui_playwright.py -q
3 passed in 9.45s

.\.venv\Scripts\python -m pytest tests/test_ui_api_contracts.py -q
4 passed, 1 warning in 1.19s

.\.venv\Scripts\python -m pytest -q
253 passed, 10 skipped, 1 warning in 49.70s
```

The branch also merges the P0 automatic CPU fallback, so a fresh
`uv sync --extra ml` on a CPU-only torch build reports a green doctor instead of
requesting CUDA.

The 10 skips are the opt-in PostgreSQL tests. The browser tests skip when the
Playwright Chromium binary is absent.

## Limitations

- Loading states are tested deterministically by holding browser requests. Real
  network latency and slow disks are not simulated.
- Error states are tested by failing responses in the browser route layer, not
  by crashing the backend during the test.
- The contract tests check route existence, envelope keys, and selector ids;
  they do not validate every field's type or every request payload.
- These tests do not establish labeling quality on production photos; that
  remains the manual real-model acceptance run.
