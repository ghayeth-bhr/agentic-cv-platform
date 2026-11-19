# Phase 1 — conversational, multi-asset, multi-class agent

Branch: `feature/agent-dialogue`
Base: `fix/data-correctness` (Phase 0)
Plan: [`docs/OPTIMIZATION_PLAN.md`](../OPTIMIZATION_PLAN.md), Phase 1.

## Why

The agent used to turn one message into a fixed plan with one asset and one class. It never asked a question, and the SAM 3 prompt agent (`prompt_enhancement.py`) had no callers. The product brief asks for a chat that is synced with the asset inventory, works across one or more assets, takes everything in one prompt, or keeps asking specific questions until the task is complete.

## Design: the LLM proposes, code decides

A 4B local model is unreliable at running a conversation, but it is good at filling a JSON schema. So:

1. **Extract.** Each chat turn becomes a `DraftPatch` (task type, classes with attributes, export format). `OllamaTaskPlanner.extract` asks the model for schema-constrained JSON (`OllamaTextClient.extract_draft`). The deterministic `RulesDraftExtractor` always runs too. It backs up fields the model leaves `unknown`, and it is the whole answer when Ollama is down. Model classes must be **grounded** (share a word stem with the user's message or the current classes), so the model cannot invent classes.
2. **Merge and check.** `apply_patch` handles replace, `also …` (append) and `remove …`. `missing_slots` checks the three required slots (assets, classes, detection/segmentation).
3. **Ask.** `next_question` asks **one templated question at a time**, with quick-reply chips for task type. Detection is assumed (and said so) after one unanswered task-type question. COCO is the default export, noted as an assumption.
4. **Prompt agent.** When the plan is complete, each class gets SAM 3 noun phrases from `OllamaPromptEnhancer` (up to three). The rules phrase is always kept as the last alternative. Phase 2 calibration will search across these alternatives; Phase 1 uses the first.
5. **Run.** Confirming calibrates **each class separately** on a diverse sample drawn from all selected assets. It submits **one** label job carrying every asset and class (category ids 1..N), builds one dataset, and packages it in the chosen format (COCO, YOLO detection, or YOLO segmentation).

A dead Ollama endpoint used to cost about 4 s per turn on Windows (`localhost` tries IPv6, then IPv4). The planner and the prompt agent now skip the model for 30 s after a failure.

## API

| Change | Notes |
|---|---|
| `POST /v1/agent/sessions` accepts `asset_ids: [..]` | `asset_id` (v1) still works. No asset is allowed: the agent asks for one. |
| **`POST /v1/agent/sessions/{id}/messages`** | `{message, asset_ids?}`. Allowed in `gathering`, `awaiting_confirmation`, `needs_attention`; otherwise 409. |
| New status `gathering` | The agent is waiting for an answer. `confirm` returns 409 until the plan is complete. |
| Session response adds `asset_ids, task_type, classes[{name, attributes, prompts}], export_format, missing_slots, pending_slot, quick_replies, assumptions` | `spec` (v1 `TaskSpec`) is present once the plan is complete. |
| Label-job payload v2: `asset_ids`, `classes[{category_id, name, prompt, threshold}]` | v1 keys are still written. `label_plan()` reads both, and is used by the worker, the dataset builder, and the live-preview route. |
| `calibration.classes[]` | Per-class attempts and verdicts. The top level mirrors the first class for older clients. |

Config: `AGENTIC_CV_AGENT_PROMPTS=rules` keeps the LLM planner but skips prompt rewriting. `AGENTIC_CV_AGENT_PLANNER=rules` disables both.

## UI

- Multi-select asset list (preselects the asset open in **Assets**). The selection is sent with every turn.
- The composer stays available while the agent is gathering or awaiting confirmation. Enter sends; Shift+Enter adds a new line. The button reads **Send** mid-conversation.
- Quick-reply chips (`Detection` / `Segmentation`).
- The task card lists classes with their SAM 3 prompt, task, assets, export format, still-needed slots, and assumptions.
- The calibration card shows a section per class. Progress and status texts name the real export format.

## Other fixes found while building this

- `clean_target_phrase` (v1 rules planner) had no word boundaries: "Create forklifts" became "klifts" because `for` matched inside the word. Fixed in both the v1 and the new lead patterns.
- `Sam3BatchLabeler` kept every full-resolution mask of a job in memory (`raw_predictions`). The durable workflow now passes `keep_raw=False`.

## Validation

```text
$ .venv/Scripts/python -m pytest -q
326 passed, 10 skipped, 1 warning

$ .venv/Scripts/python -m compileall -q src/agentic_cv tests   # ok
$ node --check src/agentic_cv/web/app.js                       # ok
$ git diff --check                                             # ok
```

New: `tests/test_agent_dialogue.py` (27 tests):
- Extraction cases, append/remove, answers that must not overwrite classes, slot order and assumptions, rules prompts.
- Grounding of model classes, model use plus fallback, and the circuit breaker.
- HTTP journeys:
  - question → answer → chip → confirm → two-class COCO, checked with `pycocotools`
  - one message across two assets → YOLO with `nc: 2` and classes `{0, 1}`
  - editing the plan before confirming
  - missing or empty assets
  - prompt agent success and failure
- A Playwright conversation driven by a quick-reply chip.

Existing tests updated for the new UI and API shape: the asset picker in two browser tests, the UI route/id contract, and one private sampler call.

## Real-model validation (qwen3-vl:4b on Ollama 0.31.2, RTX 4050 6 GB)

**Extraction.** Run against the real model, the first version showed four problems the fakes could not:
- `classes_changed` was `false` almost every time, even when the classes were correct, so classes were dropped.
- Task types were invented for messages that never mention one.
- One class came back twice (with and without its attribute).
- The prompt enhancer drifted ("yellow-orange safety vest with reflective strips") or lost meaning ("yellow vest workers").

Fixes: the flag is ignored and code compares lists; the model's task type needs a hint in the message; near-duplicates are merged; "only X" is handled; possessives are stripped ("their tusks" → "tusks"); and the user's phrase leads the prompt list, with enhancer phrases kept only if they have at most 4 words and still name the object. After the fixes, 9 of 9 probe phrases parse as intended, at about 3.5 s per turn (about 10–14 s on a cold model). Each measured shape is pinned by a unit test.

**End-to-end** (production wiring, real LLM + SAM 3 + VLM, 9 wildlife frames in 2 assets): the conversation went from question, to answer, to the task-type question, to "Segmentation, and export it as YOLO", reaching a correct plan. Calibration ran for both classes with the VLM: 459 s, peak whole-GPU 5,914 MB of 6,141 MB, no OOM. **It ended in `needs_attention` for a reason that belongs to Phase 2**, and this run was its first real evidence:
- SAM 3 was correct on all 5 samples. It segmented the elephants and tusks on 2 frames (VLM confidence 0.98 and 0.95) and returned nothing on 3 frames that contain no elephant (dik-dik, buffalo, gerenuk).
- Calibration rejects "no masks" without asking whether the target is present. Correct negatives therefore count as failures, and agreement stalled at 2/5 at every threshold.
- The diversity sampler picked exactly those three outlier frames.

## Limitations

- The rules extractor covers clear requests but not free-form phrasing. Free-form phrasing relies on the LLM path (validated below).
- Calibration is still the Phase 0 ladder, run per class, and still synchronous inside `/confirm`. With N classes the timeout budget is N × `AGENTIC_CV_CALIBRATION_TIMEOUT_SEC`. Phase 2 makes calibration a durable job, searches prompt alternatives, and reuses one forward pass per sample.
- Labeling runs one SAM 3 forward per class per image. Phase 3 shares the vision features across classes.
- The prompt agent sees only text. Grounding prompts in sample images with the VLM is a Phase 2 candidate.

## Next boundary

Phase 2: `feature/calibration-engine` (durable calibration job, one forward pass per sample, objective-based threshold choice, prompt-variant search, live 4-tile calibration board).
