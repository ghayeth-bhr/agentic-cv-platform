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

## Limitations

- **The real Ollama extraction path was not exercised in this session:** Ollama was not running on `127.0.0.1:11434`. The fallback path was exercised against the dead endpoint. Run a manual check with `qwen3-vl:4b` before relying on the LLM planner. The rules extractor covers clear requests but not free-form phrasing.
- Calibration is still the Phase 0 ladder, run per class, and still synchronous inside `/confirm`. With N classes the timeout budget is N × `AGENTIC_CV_CALIBRATION_TIMEOUT_SEC`. Phase 2 makes calibration a durable job, searches prompt alternatives, and reuses one forward pass per sample.
- Labeling runs one SAM 3 forward per class per image. Phase 3 shares the vision features across classes.
- The prompt agent sees only text. Grounding prompts in sample images with the VLM is a Phase 2 candidate.

## Next boundary

Phase 2: `feature/calibration-engine` (durable calibration job, one forward pass per sample, objective-based threshold choice, prompt-variant search, live 4-tile calibration board).
