# UI redesign — "light table" studio

Branch: `feature/ui-redesign`
Base: `main` (`cfc9aa6`, phases 0–5 merged)

## Brief

A friendly, modern, easy-to-use interface for a local studio that turns raw photos and video into a labeled computer-vision dataset with an AI agent. The user's job is to move a batch through collect → label → review and export, and always know the next step.

## Design decisions

- **Signature: the mask highlight.** Anything selected wears the warm yellow ring and tint that SAM masks get on the image: the current section, ticked assets, selected search results, the active gallery thumbnail, the chosen calibration threshold, the selected annotation, and the asset list in the Agent. Selecting something looks like masking it. This is the one bold element; everything else is quiet.
- **The sidebar is the workflow.** Assets → Agent → Datasets are numbered steps joined by a rail, because they really are a sequence. Jobs sits apart as a utility. Hints describe each step in plain words.
- **Light, cool canvas; cobalt for actions.** Photos sit on a dark stage, as in labeling tools.
- **Radius follows hierarchy:** panels 16px, controls 10px, chips fully round. Panels use borders instead of shadows.
- **One typeface (Figtree) with tabular figures.** No monospace labels, no ALL-CAPS eyebrows, no decorative numbering.
- **Copy says what each screen is for.** Empty screens make the next step the primary button.
- **The only ambient motion** is the pulsing dot on the Live badge while something runs, and it respects reduced motion.
- An automatic **dark variant** follows the system setting.

Tokens: canvas `#F3F5F9`, surface `#FFFFFF`, ink `#1A2230`, action `#2B59F0`, mask `#FFC93C`, stage `#1C212B`, success `#1C9A5B`, danger `#D9423C`.

## Fixes along the way

- **The dataset gallery opened with no image** in the viewer until you navigated (since P10; the test only checked the counter). The viewer now always loads the current image, however the index changes.
- `<textarea>` did not inherit the UI font (it showed monospace).
- The run panel's "Open gallery" button overlapped its status line, and "New task" wrapped.
- On phones the sidebar kept its desktop width; it now becomes a full-width step bar.

## Scope

A re-skin: every element id, data attribute, and visible text that behaviour or tests rely on is unchanged. Implementation: new tokens at the top of `app.css` (the old variable names map onto them, so existing component rules re-theme), a theme layer at the end, sidebar markup and copy in `index.html`, and three small JS changes (viewer loading, the empty-state primary button, header descriptions).

## Validation

```text
$ .venv/Scripts/python -m pytest -q
366 passed, 10 skipped, 1 warning   # includes all Playwright journeys
$ node --check src/agentic_cv/web/app.js   # ok
```

Screenshots of every view (empty, assets with search, agent run, datasets, gallery, jobs) were reviewed at 1440 px and at 390 px, in light and dark.

## Not done

- No new layout for the classic review panel inside dataset cards. It is restyled only; the gallery replaces it.
- No component library or build step; this remains plain HTML, CSS and JS (Phase 6 splits `app.js` into modules).
