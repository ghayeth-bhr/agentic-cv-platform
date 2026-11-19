# P7: Real Validation Loop

This branch replaces the placeholder calibration check with the real local VLM
validator, makes calibration interruptible, explains the validator decision in
the browser, and exposes COCO/YOLO export formats from the dataset card. The
fallback remains available, but it is now explicit and labeled instead of being
the silent default.

## What changed

### Default validator

- `default_validator_selection` probes Ollama lazily when a chat is confirmed,
  so app startup and the default test suite do not touch a live service.
- If `qwen3-vl:4b` (or `AGENTIC_CV_VLM_MODEL`) is installed, calibration uses
  `VlmCalibrationValidator` and records `validator_mode="vlm"`.
- If Ollama is unreachable or the model is missing, calibration uses the
  `at_least_one_prediction` fallback and the chat plus calibration payload carry
  the reason and the text `at least one prediction`.
- `AGENTIC_CV_AGENT_VALIDATOR=any-prediction` explicitly disables VLM
  validation for a run. That is a supported fallback, not a silent downgrade.
- If a VLM call fails after the probe succeeds, the validator records the
  failure and continues with the fallback for the remaining samples; the
  calibration card and chat show the failure reason. The VLM is not retried for
  every sample after a runtime failure.

### Evidence in the calibration result

Each attempt now stores the threshold, sample count, accepted count, agreement,
and one verdict per sample:

- accepted,
- confidence,
- VLM reasoning,
- overlay id.

The overlay is a real PNG of the source image with SAM 3 masks and boxes. It is
written under the data dir and served through the authenticated route:

```text
GET /v1/agent/sessions/{session_id}/calibration-overlays/{overlay_id}
```

The browser renders the sample overlays lazily, shows the validator mode/model,
the notice when the fallback is active, every threshold tried, accepted counts,
and each VLM verdict. Overlay URLs are revoked when the calibration is replaced
or the session is reset; polling a session does not collapse an open details
panel or revoke URLs on every refresh.

### Cancellation and timeouts

- `CalibrationCancelled` and a `should_cancel` check run between samples and
  between thresholds.
- The cancel endpoint signals the active calibration event before marking the
  session cancelled.
- A calibration timeout releases the model/GPU lock, stores the timeout reason,
  and does not block the next job. The HTTP test starts a new run immediately
  after a timed-out one.
- The agent joins the calibration thread for up to 2 seconds after setting the
  cancellation flag, so a cancelled run cannot keep writing after the session
  is marked.

### Diversity sampling

`select_calibration_images` reads the stored SigLIP vectors and uses
farthest-point selection, starting from the image farthest from the embedding
centroid. If vectors are absent, zero, or mismatched, it falls back to evenly
spaced images. If the sample size covers the asset, every image is returned. The
agent reads vectors from the tenant's vector index before sampling.

### Dataset export formats

- The dataset card has a `select[data-export-format]` with `coco`,
  `yolo-detection`, and `yolo-segmentation`.
- The selected format is sent to `POST /v1/dataset-exports`; the download is
  named with the chosen format.
- The Agent's own run-panel download remains COCO. YOLO is available from the
  dataset card and the export API.

### Doctor and CUDA wheels

`agentic-cv doctor` now detects an NVIDIA GPU while `torch.version.cuda` is
empty, warns that torch is CPU-only, and prints the Windows fix. This run's
`doctor --no-network` output on the CUDA environment was `11 ok, 0 warn, 0 fail`.

`pyproject.toml` adds the `pytorch-cu124` index and Windows-only
`[tool.uv.sources]` for `torch`/`torchvision`. `uv sync --extra ml` therefore
resolves the CUDA wheels on this Windows machine. `uv lock` and
`uv lock --check` pass for the locked set (90 packages).

## Measured run

The real 20-image HTTP journey with SigLIP2, SAM 3, and `qwen3-vl:4b`:

```powershell
$env:AGENTIC_CV_VLM_TIMEOUT_SEC = "180"
..\frames-extractor\.venv\Scripts\python.exe scripts/measure_full_journey.py `
  --images 20 --source .agentic-cv/derived/semantic-frames `
  --message "Create a segmentation dataset of animals." --prompt "animals" `
  --require-vlm --timeout 900 --calibration-timeout 600 `
  --json-out ..\p7-vram-measurement.json
```

Observed:

- calibration status `calibrated`, threshold `0.3`, validator `vlm`,
  model `qwen3-vl:4b`;
- 5 samples, 5 accepted verdicts, 5 real overlay ids, one threshold tried;
- 20 images, 34 annotations, COCO archive 3,602,141 bytes;
- refinement accepted, revision 2, rebuilt/refined archive 3,602,143 bytes;
- PyTorch peak 4,202.8 MB allocated / 4,390.0 MB reserved;
- whole-GPU `nvidia-smi` peak 5,822 MB of 6,141 MB, with Ollama already
  resident at the start (window baseline 3,437 MB);
- calibrate/label/export checkpoint 383.21 s versus 40.29 s on the P5 no-VLM
  run.

Exact output, command, and the difference between the torch and `nvidia-smi`
counters are recorded in [`docs/PERFORMANCE.md`](../PERFORMANCE.md).

## Fallback and safety boundaries

- The VLM is an external Ollama process; the app's shared model manager does not
  unload it. On the 6 GB laptop the VLM and SAM 3 coexist by using shared/WDDM
  memory, and the run is slower.
- A VLM timeout during calibration degrades to the labeled fallback for the
  remaining samples; it does not silently claim VLM evidence.
- The VLM judges the overlay, not the original full-resolution image. Overlays
  are generated at display size, so very small objects may be harder to judge.
- The fallback says `at least one prediction`; it is not a quality check.
- COCO and YOLO archives are generated by the existing export service.
  This branch does not add a new annotation-quality claim.
- The real run used cached 720x1280 video frames, not the owner's production
  photos. Segmentation quality on production photos is still unvalidated.

## Tests

Added or extended:

- `tests/test_calibration_samples.py`: diversity selection, even-spacing
  fallback, zero/mismatched vectors, full-sample short circuit, vector-index
  integration.
- `tests/test_calibration_validation.py`: Ollama probe, missing model, explicit
  disabled mode, unreachable fallback notice, VLM selection, real PNG overlay.
- `tests/test_vlm_calibration.py`: traced verdicts, empty-prediction skip,
  runtime failure fallback and notice.
- `tests/test_ollama_vlm.py`: JSON in `message.thinking`, malformed response.
- `tests/test_calibration.py`: cancellation/trace evidence.
- `tests/test_agent_chat_e2e.py`: VLM calibration and authenticated overlay
  route, fallback notice, timeout non-blocking, cancel during calibration.
- `tests/test_persistent_vectors.py` plus vector-index tests: `get_embedding`
  for local, SQLite, and Postgres indexes.
- `tests/test_doctor.py`: NVIDIA GPU with CPU-only torch warns and prints the
  Windows fix.
- `tests/test_agent_ui_playwright.py`: calibration overlays/VLM verdicts and
  YOLO dataset download.
- `tests/test_ui_api_contracts.py`: dataset-format selector and export contract.

Default `pytest` remains deterministic: agent confirmation tests inject a
validator factory, so a locally running Ollama cannot change the result.

Verification output from the release gate:

```text
$ .venv\Scripts\python -m pytest -q
275 passed, 10 skipped, 1 warning in 44.05s

$ .venv\Scripts\python -m pytest tests/test_agent_ui_playwright.py tests/test_manual_refinement_ui_playwright.py tests/test_assets_ui_playwright.py -q
3 passed in 18.13s

$ node --check src/agentic_cv/web/app.js
(no output; exit 0)

$ .venv\Scripts\python -m compileall -q src tests scripts
(no output; exit 0)

$ uv lock --check
Resolved 90 packages in 3ms

$ agentic-cv doctor --no-network --port 8124
...
Summary: 11 ok, 0 warn, 0 fail
```

The 10 skips are the opt-in PostgreSQL tests. The real 20-image VLM run and
its exact JSON output are in [`docs/PERFORMANCE.md`](../PERFORMANCE.md).

## Demo (section 7)

1. Start Ollama and pull `qwen3-vl:4b`.
2. `agentic-cv doctor` and `agentic-cv serve`.
3. Upload several photos into one asset and wait for indexing.
4. Open **Agent**, choose the asset, enter a task, and confirm.
5. Confirm the calibration card says **Vision (VLM)**, shows the model, the
   thresholds tried, sample overlays, and accepted/rejected verdicts. If it says
   **at least one prediction**, read the notice and start Ollama before treating
   the run as VLM-validated.
6. Watch labeling and dataset packaging, then download the COCO archive.
7. Open **Datasets**, choose **YOLO detection** or **YOLO segmentation** in the
   new format selector, click **Export**, and verify the downloaded archive.
8. Open **Review labels**, correct one image, accept, rebuild, and export the
   refined version.
9. Reload the page and confirm the finished Agent run and its calibration
   evidence are restored.
