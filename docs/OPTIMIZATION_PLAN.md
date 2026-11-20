# Agentic CV Platform: Review, Rating, and Optimization Plan

Reviewed at commit `ecb1854` (branch `feature/dataset-gallery-live-view`) on 2026-11-19.
Baseline: `pytest` gives 275 passed and 10 skipped, not counting Playwright. Every finding below comes from reading the code and cites `file:line`. Nothing was changed in `src/`.

---

## 1. Executive summary

The repository is a well-tested, durable **linear pipeline** with a chat front end:
`one message → one LLM call → fixed threshold ladder → label → COCO`.
The product you described is an **agentic, multi-turn, human-in-the-loop dataset factory**. The infrastructure (jobs, leases, revisions, stores, tests) is good. Most of the gap is in what makes the product useful:

1. **The agent is not agentic.** It never asks a question, has no follow-up turn, and works on one asset with one class. The SAM 3 prompt-engineering agent was built (`prompt_enhancement.py`) but nothing calls it.
2. **Calibration is a ladder, not a search.** The first threshold that passes wins, starting at the most permissive one (0.30). The model also runs again for every threshold, even though the threshold is only post-processing.
3. **Data correctness bugs** silently corrupt datasets: annotations from earlier runs leak into new ones, YOLO exports are unusable for training, and there is no way to delete a false positive.
4. **The review and approval workflow is missing.** You can't approve images or annotations. "Refinement" re-ranks SAM 3 text-prompt masks instead of doing real point-prompted segmentation, so it can't add a missed object.
5. **Effort was spent on enterprise plumbing** (OIDC, RBAC, RabbitMQ, S3, Postgres, outbox) that a local single-user tool doesn't need, while the core features above went unbuilt.

**Recommendation:** freeze the enterprise surface. Spend the next cycle on data correctness, then on the conversational agent, the calibration engine, and review/approval, in that order (Section 6).

---

## 2. Rating

| Dimension | Score /10 | Why |
|---|---|---|
| Engineering hygiene (tests, durability, docs, honesty) | **7.5** | 275 deterministic tests, provider injection and fakes, leases and heartbeats, an honest README, and `doctor`. |
| Fit with the product vision | **4** | See the traceability matrix (Section 3). Core agentic behaviours are missing. |
| Agentic design (orchestration, autonomy, dialogue) | **3** | A single planner call and a hard-coded state machine. No clarification, no prompt agent, no monitor agent. |
| Business logic correctness (labels, datasets, exports) | **3.5** | Annotation namespace collision, YOLO class index off by one, val == train, no deletes. |
| Data and ML pipeline performance | **4** | About 5× redundant SAM 3 passes in calibration, one image at a time, full re-inference on every click, brute-force vector scan, video frames capped. |
| UX | **5** | The gallery and live view are decent. Missing: chat follow-ups, multi-select, approval, live calibration tiles, per-result selection. |
| Maintainability and architecture | **5** | `api.py` is 1,711 lines and `app.js` is 2,775 lines in one file. Over-built infrastructure for the scope. Clean module boundaries elsewhere. |
| **Overall** | **≈ 4.8** | Solid foundation, wrong layer finished. |

---

## 3. Requirements traceability (what you asked for vs. what exists)

| # | Your requirement | Status | Evidence |
|---|---|---|---|
| A1 | Gallery: see, navigate, select, upload images/videos | ✅ Mostly | Upload, cards, and detail exist. No multi-select of assets. |
| A2 | Semantic search within one or many assets | ⚠️ Partial | The API takes `asset_ids` (`semantic_api.py:27`). The UI scope is only "all" or "selected asset" (`index.html:130`). |
| A3 | **Select** images from search results and create a new asset from the selection | ❌ Gap | `POST /v1/sub-assets/from-search` **re-runs the query** and saves every top-N result (`semantic_api.py:381-393`). There is no per-image selection and no score cut-off. |
| B1 | Chat synced with the asset inventory; choose **one or more** assets | ❌ Gap | There is a single `<select id="agent-asset">` (`index.html:177`), and the session has one `asset_id` (`agent.py:254`). |
| B2 | One complete prompt **or** the agent keeps asking specific questions | ❌ Gap | The planner always returns a spec (`agent.py:126-168`). No "missing info" concept. No message endpoint after creation (`agent_api.py` only has create, confirm, and cancel). |
| B3 | Detection or segmentation, multiple classes | ⚠️ Partial | Targets are joined into one prompt, `"a and b with x"` (`agent.py:195-207`). Every run is hard-coded to `category_id=1` (`agent.py:443`). |
| C1 | Orchestrator agent that triggers sub-agents | ❌ Gap | `AgentService` is a hard-coded procedure, not an orchestrator with tools or agents. |
| C2 | Prompt-enhancement agent for SAM 3 | ❌ Built, not wired | `prompt_enhancement.py` has no callers (`grep OllamaPromptEnhancer` finds only its definition). |
| C3 | Threshold-calibration agent (Roboflow-style, 4 samples, live) | ⚠️ Weak | A fixed ascending ladder where the first pass wins (`calibration.py:114-154`). It runs synchronously inside the HTTP request. There is no live per-threshold sample view. |
| C4 | Auto-label agent over the whole asset | ✅ | `label_worker.py`, `label_workflow.py`. |
| C5 | Monitor agent for health | ❌ Gap | There are heartbeats and leases. Nothing watches quality or throughput or acts on anomalies. |
| C6 | Cool live preview during labeling | ⚠️ Partial | P10 live grid. The "SSE" closes after each snapshot and the client reconnects every 300 ms, so it is polling (`app.js:1301`). The server renders a PNG per tile. |
| D1 | Post-annotation gallery with navigation | ✅ | P10 gallery, filters, toggles. |
| D2 | **Approve** selected images or all of them | ❌ Gap | No approval state anywhere (`grep -i approv` finds nothing). |
| D3 | Refine with cursor pointing using SAM | ⚠️ Not real | `Sam3ClickRefinementProvider` re-runs the **text** prompt and picks the mask closest to the clicks (`click_refinement.py:303-349`). It can't add a missed object or split or merge objects. No annotation delete. |
| E1 | Export COCO and YOLO after approval | ⚠️ Buggy | See bugs B4 and B5. Export is not gated on approval. |
| F1 | Runs locally with SAM 3, VLM, Ollama | ✅ | Doctor, GPU manager, Ollama unload. Measured on the 6 GB GPU. |

---

## 4. Defects (fix before adding features)

Severity: **S1** corrupts data or blocks the core flow. **S2** gives wrong behaviour that users notice. **S3** is a quality or performance issue.

| ID | Sev | Defect | Evidence | Failure scenario |
|---|---|---|---|---|
| B1 | S1 | **Annotation namespace collision across runs.** IDs are `"{image_id}:{index}"`, revisions are global per image, every run uses `category_id=1`, and the builder takes the "latest per id". | `batch_labeling.py:326`, `agent.py:443`, `dataset_builder.py:79-86` | Run 1 ("car") finds 3 boxes on image A. Run 2 ("person") finds 1. Dataset 2 contains 1 person box **plus 2 stale car boxes labelled "person"**. If run 2 finds 0, all 3 car boxes remain. |
| B2 | S1 | **No annotation deletion or tombstone.** False positives can't be removed, and an empty result can't overwrite older results. | `annotation_store.py` (insert only), no delete route in `api.py` | A reviewer can't remove a wrong mask, so the exported dataset keeps it. |
| B3 | S1 | **Refinement accept with no `annotation_id` overwrites the first annotation of the category** instead of adding a new one. | `api.py` accept_refinement, fallback loop at the `list_latest` branch | Clicking a missed second object replaces the correct first object. |
| B4 | S1 | **YOLO class index = `category_id` (1-based).** `data.yaml` names `{1: ...}`, so `nc=1` with label class `1` is out of range for Ultralytics. | `dataset.py:180,189`, `dataset_export.py:127-130` | `yolo train data=data.yaml` fails or mislabels classes. |
| B5 | S1 | **YOLO `train: images`, `val: images`.** Validation equals training, and the export has no splits. The gallery shows hash "splits" that the export ignores. | `dataset_export.py:128`, `api.py:376-384`; `split_dataset` is unused | Leaked validation, so mAP is meaningless. The UI and the export disagree. |
| B6 | S2 | **YOLO-seg concatenates multiple polygons into one line.** Disjoint parts become one self-intersecting polygon. No simplification and no minimum area. | `dataset.py:184-189`, `batch_labeling.py:257-267` | Occluded objects produce garbage polygons. Label files are huge (`CHAIN_APPROX_SIMPLE` without `approxPolyDP`). |
| B7 | S2 | **Videos are silently truncated** to 120 sampled frames (stride 30, about 2 min at 30 fps). Frames are extracted twice: once for search and once for labeling. | `label_images.py:84-87`, `semantic_indexing.py:83-84` | A 10-minute video is labelled on its first 2 minutes. Disk and CPU usage double. |
| B8 | S2 | **Calibration is biased toward false positives.** Thresholds go up from 0.30 and the first pass wins. In fallback mode, "≥1 prediction" passes at 0.30 almost always, so calibration does nothing. The `min_prediction_score=0.2` guard can't fire because the lowest threshold is 0.3. | `agent.py:49`, `calibration.py:114-154`, `calibration_validation.py:150-155` | Datasets are labelled at the most permissive threshold. |
| B9 | S2 | **Calibration runs inside `POST /confirm`** (sync, up to 120 s by default). The P7 VLM path measured 383 s end to end. On timeout, the daemon thread keeps running inside SAM 3 while labeling may start. | `agent_api.py:267-296`, `agent.py:749-779` | Browser request timeouts, an orphaned GPU thread, and calibration that can't resume after a restart. |
| B10 | S2 | **Sub-asset = re-run the query.** The user's selection is ignored, and results can differ from what was displayed. | `semantic_api.py:381-393` | The user ticks 6 images and gets 20. |
| B11 | S3 | **Redundant SAM 3 passes.** `predict()` runs for every (threshold × sample) pair, although the threshold is only used in `post_process_instance_segmentation`. | `calibration.py:118-120`, `sam3.py:186-192` | 5 thresholds × 5 samples = 25 forward passes instead of 5. |
| B12 | S3 | **Each refinement click runs full SAM 3 again** on the whole image. No image-embedding cache. | `click_refinement.py:329` | About 1 s or more per click instead of under 100 ms. |
| B13 | S3 | **Vector search loads every row on every query** (`SELECT *`), then scores them in Python. | `persistent_vectors.py:98` | Search gets linearly slower. At 50k frames it takes seconds. |
| B14 | S3 | **Dataset `source_asset` is parsed out of the display name.** | `api.py:387` | Renaming a dataset breaks lineage. |
| B15 | S3 | **SQLite has no WAL or busy_timeout.** Four worker threads plus the request pool share connections behind Python locks. | All `sqlite3.connect(..., check_same_thread=False)` | Lock contention and `database is locked` errors under the live view's polling. |

Hygiene: the untracked `HANDOFF.md` contains a personal e-mail address and identity details, so don't commit it. `src/agentic_cv/local_storage.py` is untracked and unused. The README hard-codes a personal absolute path.

---

## 5. Gap analysis by area

### 5.1 Agent and orchestration (largest gap)
- **No dialogue model.** A session is `plan → confirm`. You asked for slot filling: the agent extracts what it can, then **asks targeted questions** until the spec is complete.
- **No explicit orchestrator.** The stages are hard-wired methods. Make them tools or sub-agents with typed inputs and outputs (`PromptAgent`, `CalibrationAgent`, `LabelingAgent`, `MonitorAgent`, `ReviewAgent`, `ExportAgent`) and keep the deterministic state machine as the control plane. The LLM proposes and code decides. That keeps it reliable on a 4B local model.
- **Single asset, single class.** Every downstream step (category ids, prompts, calibration, export) assumes one class.
- **The prompt agent is unused.** SAM 3 works best with short noun phrases, **one concept per prompt**. Today `"worker and helmet with yellow vest"` is sent as one string.
- **The VLM is only a judge.** It could also *propose* prompts from sample images ("what do the targets look like here?") and detect ambiguous specs.

### 5.2 Calibration (core differentiator; currently weakest)
Roboflow's manual step: try thresholds on about 4 images, watch the overlays, pick the best. Doing it agentically should be **faster and better than a human**, not a ladder:
- Run the model once per (sample, prompt). Sweep every threshold on the cached logits for free.
- Group thresholds by the **set of masks they keep**. Only the distinct sets (usually 3 to 6) need VLM judgment.
- Score each threshold with an objective, for example `accepted − λ·false_positive − μ·missed`, using the VLM's structured booleans (already in `ValidationJudgment`). Pick the **best**, not the first.
- Search prompt variants × thresholds, because the prompt is often the real lever.
- Stream every attempt to the UI as 4 live tiles with a threshold slider. Cached outputs make the slider instant, so the user can override.

### 5.3 Labeling engine and monitoring
- Inference runs one image at a time with decode on the GPU thread. Prefetch on CPU and add optional micro-batching.
- Multi-class: compute vision features once per image and run text prompts per class against them (Transformers SAM 3 supports precomputed `vision_embeds`).
- **Monitor agent** (deterministic rules plus optional LLM summary): throughput, ETA, empty-image rate, detections-per-image outliers, score drift against calibration, stall and OOM detection. Actions: retry an image, lower the batch size, **pause and ask the user** ("62% of images have no detections. Did the prompt drift?").

### 5.4 Review, approval, refinement
- Add per-image `review_status` (`pending | approved | rejected | needs_fix`) and per-annotation delete and reclassify.
- Bulk-approve selected or all, and filter by status, class, or low confidence. Keyboard flow: `A` approve, `R` reject, `←/→` navigate, `D` delete selected mask.
- **Real point and box prompting** with SAM 3's tracker or PVS head (`Sam3TrackerModel` in Transformers, SAM2-style points and boxes). Cache image embeddings per session.
- A dataset version is built from **approved** images only, or it shows a clear "includes N unreviewed" warning.

### 5.5 Assets and search
- Per-result checkboxes, shift-range select, "select all ≥ score", a score cut-off slider, and "create asset from selection" by `record_ids`.
- Multi-asset scope picker and multi-select assets in the gallery (shared with the Agent picker).
- Video: sample by **time** (for example 1 fps, configurable) with scene-change or near-duplicate dedup. Use one shared frame cache for search and labeling.
- Vectors: keep a normalized float32 matrix in memory per tenant, invalidated on index. That is fine to about 1M × 768. Postgres/pgvector stays opt-in.

### 5.6 UX
- The chat should look like a conversation: an input that is always present, agent questions with quick-reply chips (`Detection` / `Segmentation`, class chips), and a visible task card that fills in as slots are resolved.
- Run timeline: Plan → Prompt → Calibrate → Label → Review → Export. Each step is expandable and shows evidence.
- The live preview draws polygons client-side on a canvas from JSON events instead of server PNGs, with a real push stream.

### 5.7 Architecture and maintainability
- Split `api.py` into domain routers: assets, search, agent, labels, datasets, review, exports. Split `app.js` into native ES modules (still no build step).
- Unify job kinds (`calibrate`, `batch-label`, `index`, `export`) under one runner with typed payloads. Calibration becomes a durable job.
- One **GPU scheduler** with priority lanes: interactive refinement > calibration > batch labeling > indexing. Extend `ModelResourceManager` with a priority queue so clicks don't wait behind a 1,000-image job.
- **Freeze** OIDC, RBAC, RabbitMQ, S3, Postgres, and the outbox. Keep them behind flags and their tests, add nothing new to them, and remove them from the default mental model and the README headline.

### 5.8 Quality measurement (missing entirely)
Nothing measures **label quality**. Add a small golden set (about 50 images with ground-truth boxes and masks, 2 to 3 classes) and an `eval` script that reports mAP@0.5, mask IoU, and the chosen threshold against the oracle best threshold. Every calibration or prompt change must move these numbers. Today there is no metric to optimize.

---

## 6. Optimization and fix plan

Each phase is one branch (your workflow rule), in the order below. Estimates are focused engineering days for one developer or agent.

### Phase 0: Data correctness and hygiene (≈3 days) — branch `fix/data-correctness`
| Task | Fixes | Acceptance |
|---|---|---|
| Introduce an **annotation set** scope: `annotation_id = "{run_or_set_id}:{image_id}:{n}"`. The builder reads by set, not by global image. Manual edits write to the dataset's working set. | B1 | A test runs two prompts on the same asset; dataset 2 contains only run 2's labels. |
| Tombstone revisions (`deleted=true`) plus `DELETE /v1/datasets/{v}/images/{i}/annotations/{a}`. A label run with empty output writes tombstones for older entries in its set. | B2 | Deleting then rebuilding gives an export without the annotation. |
| Refinement accept with no `annotation_id` creates a **new** annotation. Replacing requires an explicit id. | B3 | A test adds a second object and the first is untouched. |
| YOLO: map categories to contiguous 0-based indices. `names` is a 0-based list. Persist splits at build time (source-aware: all frames of one video go to the same split) and emit `images/{train,val,test}`, `labels/...`, and a correct `data.yaml`. COCO gets one JSON per split. | B4, B5 | An Ultralytics `check_det_dataset` or a tiny `yolo train epochs=1` smoke runs on the export (opt-in test). `pycocotools` loads every split. |
| Polygons: `approxPolyDP` (epsilon ≈ 0.002·perimeter), drop contours under a minimum area, emit one polygon per instance (largest, or a merged multi-segment like Ultralytics `merge_multi_segment`). | B6 | Golden mask IoU is ≥ 0.97 after simplification, and file size drops. |
| Video sampling by time (`fps` default 1, `max_frames` configurable, default unbounded with a warning above N) plus one shared frame cache. | B7 | A 10-minute test video covers its full duration. Frames are extracted once. |
| SQLite: `PRAGMA journal_mode=WAL; busy_timeout=5000`. | B15 | Concurrency test with the live view and the worker shows no lock errors. |
| Hygiene: keep `HANDOFF.md` out of git (add it to `.gitignore`), delete or wire `local_storage.py`, store `source_asset_id` on dataset versions. | B14 | |

### Phase 1: Conversational, multi-asset, multi-class agent (≈6 days) — branch `feature/agent-dialogue`
1. **Spec model v2:** `asset_ids[]`, `task_type`, `classes[] = {name, description, attributes[], sam_prompts[]}`, `export_formats[]`, optional `sample_size`. Track `missing_slots[]` and `assumptions[]`.
2. **`POST /v1/agent/sessions/{id}/messages`:** each user turn → the LLM returns a **spec patch plus at most one question** (JSON schema). A deterministic validator computes `missing_slots`. Status is `gathering` until complete, then `awaiting_confirmation`.
3. Question policy: ask only about required slots (task type, classes) and real ambiguity, at most 3 questions. Then propose defaults the user can accept with one click.
4. **Prompt agent (wire `prompt_enhancement.py`):** for each class, generate 2 to 3 SAM 3 noun phrases. Optionally ground them with the VLM on 2 sample images. These are the candidates for calibration.
5. UI: always-on chat input, quick-reply chips, a multi-select asset picker synced with the gallery selection ("Use selected assets in Agent"), and a live task card.
6. Keep the rules planner as the offline fallback. Its fallback must also ask instead of guessing when it can't extract classes.

**Acceptance:** "make a dataset" → the agent asks for task type and classes. "Segment helmets and vests on assets X, Y, export YOLO" → no questions, goes straight to confirmation. HTTP and Playwright tests for both paths with a fake LLM.

### Phase 2: Calibration engine (≈5 days) — branch `feature/calibration-engine`
1. Calibration becomes a **durable `calibrate` job** run by the worker. `/confirm` returns 202. The UI streams progress. It is resumable and cancellable. (B9)
2. `Sam3Provider.infer(image, prompt) → raw outputs` plus `postprocess(raw, threshold)`. Cache raw outputs per (sample, prompt) on CPU. (B11)
3. Sweep a dense grid (0.20–0.90, step 0.05). Group thresholds by kept-mask set and call the VLM only on distinct sets, reusing the existing overlay digest cache.
4. Objective: maximize `Σ accepted − λ·FP − μ·missed` over samples (from the `ValidationJudgment` booleans), with a tie-break toward the higher threshold. Search over prompt variants × thresholds per class, and pick the best (prompt, threshold) per class.
5. Fallback mode (no VLM): pick the **score-gap** threshold (the largest gap in the sorted score distribution across samples) instead of "≥1 prediction", and label it heuristic. (B8)
6. "Cannot segment" outcome: when the best objective is below a floor, say which class failed, show the evidence, and offer alternatives (other prompt or manual threshold).
7. UI: a **4-tile live calibration board** per class, showing the current threshold overlays, VLM verdict badges, and a slider that re-renders instantly from cached outputs. The user can override and then confirm.

**Acceptance:** SAM 3 forward passes = samples × prompts (assert with a counting fake). On the golden set, the chosen threshold's mAP is within 2 points of the oracle best threshold. Calibration time with VLM at 5 samples is under 90 s on the 6 GB laptop (measured and recorded in `PERFORMANCE.md`).

### Phase 3: Labeling throughput and Monitor agent (≈4 days) — branch `feature/label-engine-v2`
1. A CPU prefetch thread for decode and resize. Optional micro-batch (auto-sized from free VRAM, fallback 1).
2. Multi-class: one vision pass per image, then a text-prompt head per class.
3. Real push events: an in-process pub/sub fanned out to SSE (keep the durable preview table for replay). Events carry `{image_id, thumb_url, polygons/boxes, scores}` and the client draws them. Remove the 300 ms reconnect loop.
4. **Monitor agent:** rolling metrics (img/s, ETA, empty rate, mean score versus calibration, per-image detection outliers, heartbeat stalls, CUDA OOM). Rules trigger actions: auto-retry, reduce batch, or **pause and ask the user in chat**. Show health in the run panel.
5. GPU priority lanes in `ModelResourceManager` (interactive > calibrate > batch > index).

**Acceptance:** throughput is at least 1.5× P8 on the 20-image journey. An injected "empty-rate spike" fake pauses the job and posts a chat question. A refinement click during a batch job returns in under 1 s.

### Phase 4: Review, approval, real point refinement (≈5 days) — branch `feature/review-approval`
1. `review_status` per (dataset working set, image) with bulk endpoints (`approve selected`, `approve all filtered`) and status filters and counters in the gallery.
2. Annotation edit: delete, change class, undo (revision history already exists).
3. **Point and box prompted segmentation** through `Sam3TrackerModel` (positive and negative points, box). Cache the image embedding per refinement session. Add a new object or replace the selected one. Keep the old ranked-mask mode only as a fallback. (B12)
4. "Build dataset" uses approved images only by default (a toggle includes unreviewed ones, with a warning). Export is offered from the gallery and the agent panel with format choice.
5. Keyboard-first review flow and an "approve and next" action.

**Acceptance:** Playwright: approve 3, reject 1, add a missed object by clicking, delete a false positive, build, export YOLO. The archive contains exactly the approved images with the corrected labels. Click-to-mask latency is under 300 ms after the first click (measured).

### Phase 5: Assets and search UX (≈3 days) — branch `feature/search-selection`
1. Selectable results, a score cut-off slider, and "create asset from selected" by `record_ids` (new request model; the old query-based endpoint stays for compatibility). (B10)
2. Multi-asset scope picker. Gallery multi-select shared with the Agent.
3. In-memory normalized vector matrix per tenant with invalidation on index. Top-k with `argpartition`. (B13)
4. Thumbnail cache (on-disk, sized), and lazy-loaded result grids.

**Acceptance:** select 6 of 20 results and the new asset has exactly 6 sources. Search over 50k vectors takes under 100 ms (benchmark script).

### Phase 6: Structure and maintainability (≈3 days, can overlap) — branch `refactor/routers-modules`
- Split `api.py` into routers and `app.js` into ES modules (`assets.js`, `search.js`, `agent.js`, `gallery.js`, `review.js`, `api-client.js`).
- One job runner. Typed payload dataclasses per job kind.
- README: lead with the product journey and move the enterprise options to an "Advanced / optional" section. Remove personal paths.

### Phase 7: Quality harness (≈2 days, start in parallel with Phase 2) — branch `feature/eval-harness`
- Golden set plus a `scripts/eval_labels.py` that reports mAP@0.5, mask IoU, the chosen threshold versus the oracle, calibration wall time, and VLM call count. Write the results to `docs/PERFORMANCE.md`.
- Wire it as an opt-in pytest marker `realmodel`.

**Total ≈ 31 focused days.** Critical path: 0 → 1 → 2 → 4. Phases 3, 5, 6, and 7 can be interleaved.

---

## 7. Keep (don't rewrite)
- Durable job store with leases, heartbeat, cancel, and expired-lease recovery.
- Provider-injection pattern and fake providers. Deterministic default `pytest`.
- Immutable dataset versions plus annotation revisions (extend with sets and tombstones, don't replace).
- `ModelResourceManager` (extend with priorities), `doctor`, Ollama unload after calibration.
- Diversity sampling from SigLIP vectors (`calibration_samples.py`). It's good; feed it per-class.

## 8. Risks and mitigations
| Risk | Mitigation |
|---|---|
| A 4B local LLM produces unreliable dialogue JSON | Schema-constrained output, a deterministic validator, one question per turn, the rules fallback, and the golden dialogue tests. |
| VRAM: SAM 3 + VLM + tracker on 6 GB | Sequential stages, VLM unload (exists), tracker shares the SAM 3 backbone where the Transformers API allows it, priority lanes. |
| `Sam3TrackerModel` API differences across Transformers versions | Pin to the validated `transformers 5.14.x`, add a contract test with a tiny fake, and keep the fallback ranked-mask mode. |
| Annotation-set migration breaks existing local datasets | A migration that assigns existing revisions to a legacy set per dataset version. Old versions remain readable. |
| Scope creep returns | This plan's phase list is the backlog. Enterprise items are explicitly frozen. |

## 9. Definition of done (product level)
From a fresh `agentic-cv serve`: upload 2 assets (photos plus a video) → search and select 30 frames → create an asset → in chat, "build a dataset" → answer 2 questions (segmentation; helmets, vests) → watch 4-tile calibration per class → watch live labeling with health → review: approve most, fix 2 by clicking, delete 1 false positive → export YOLO-seg plus COCO with train/val/test → `yolo train epochs=1` runs on the export. All of it is covered by one Playwright journey with fakes and by one recorded real-model run.
