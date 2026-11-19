"""Reliability checks for the real SAM 3 + Ollama VLM calibration loop.

This is a manual harness. It is not collected by pytest because it loads real
model weights and calls a local Ollama server. It writes machine-readable JSON
and prints the same report. Do not turn these measurements into targets.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from agentic_cv.agent import RulesTaskPlanner
from agentic_cv.application import LocalAppConfig, create_local_app
from agentic_cv.calibration import CalibrationSample
from agentic_cv.calibration_validation import default_validator_selection
from agentic_cv.ollama_vlm import OllamaVlmClient
from agentic_cv.overlay import render_prediction_overlay
from agentic_cv.sam3 import LocalSam3Provider
from agentic_cv.vlm_validator import OverlayArtifact, VlmCalibrationValidator, VlmValidator

TENANT_ID = "00000000-0000-0000-0000-000000000001"
TARGET_PROMPT = "orange rectangle"
ABSENT_PROMPT = "purple unicorn"
IMAGE_SIZE = 320


def _case(index: int, present: bool) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(7000 + index)
    shade = int(70 + (index % 5) * 8)
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (38, shade, 42))
    draw = ImageDraw.Draw(image)
    draw.ellipse((16, 18, 66, 68), fill=(92, 92, 92))
    draw.rectangle((260, 20, 300, 60), fill=(30, 40, 30))
    ground_truth = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)

    if present:
        x = int(rng.integers(45, 115))
        y = int(rng.integers(45, 105))
        width = int(rng.integers(130, 165))
        height = int(rng.integers(140, 175))
        box = (x, y, x + width, y + height)
        draw.rectangle(box, fill=(205, 145, 85))
        ground_truth[y : y + height, x : x + width] = True
    return image, ground_truth


def _mask_iou(mask, ground_truth: np.ndarray) -> float:
    predicted = np.asarray(mask).astype(bool)
    if predicted.shape != ground_truth.shape:
        return 0.0
    intersection = int(np.logical_and(predicted, ground_truth).sum())
    union = int(np.logical_or(predicted, ground_truth).sum())
    return intersection / union if union else 0.0


def _best_iou(predictions, ground_truth: np.ndarray) -> float:
    return max((_mask_iou(item.mask, ground_truth) for item in predictions), default=0.0)


def _false_positives(predictions, ground_truth: np.ndarray) -> int:
    return sum(1 for item in predictions if _mask_iou(item.mask, ground_truth) < 0.1)


def _build_validator(root, client: OllamaVlmClient, minimum_prediction_score: float):
    directory = root / "overlays"

    def build(sample: CalibrationSample, predictions) -> OverlayArtifact:
        digest = hashlib.sha256(sample.image_id.encode("utf-8"))
        for item in predictions:
            digest.update(f"{item.score:.6f}|{item.box}".encode("ascii"))
        overlay_id = f"{digest.hexdigest()[:24]}.png"
        path = directory / overlay_id
        render_prediction_overlay(sample.image, predictions, path)
        return OverlayArtifact(uri=str(path), overlay_id=overlay_id)

    validator = VlmValidator(client, max_calls=250, minimum_confidence=0.7)
    return VlmCalibrationValidator(
        validator,
        build,
        model_revision=client.model,
        fallback=None,
        minimum_prediction_score=minimum_prediction_score,
    ), validator


def _upload_png(api: TestClient, headers: dict[str, str], asset_id: str, image: Image.Image) -> None:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    begin = api.post(
        f"/v1/assets/{asset_id}/uploads",
        headers=headers,
        json={
            "filename": f"absent-{digest[:8]}.png",
            "media_type": "image/png",
            "expected_sha256": digest,
        },
    )
    begin.raise_for_status()
    upload_id = begin.json()["upload_id"]
    part = api.put(
        f"/v1/uploads/{upload_id}/parts/1",
        files={"part": ("absent.png", payload, "image/png")},
    )
    part.raise_for_status()
    complete = api.post(f"/v1/uploads/{upload_id}/complete")
    complete.raise_for_status()


def _absent_prompt_check(root, provider: LocalSam3Provider) -> dict:
    app = create_local_app(
        LocalAppConfig(
            data_dir=root / "absent-app",
            sam3_provider=provider,
            agent_planner=RulesTaskPlanner(),
            calibration_validator_factory=lambda tenant_id, session_id: (
                default_validator_selection(
                    overlay_root=root / "absent-overlays",
                    tenant_id=tenant_id,
                    session_id=session_id,
                )
            ),
            calibration_timeout_sec=300,
            start_workers=False,
        )
    )
    headers = {"X-Tenant-ID": TENANT_ID}
    with TestClient(app, raise_server_exceptions=False) as api:
        asset = api.post("/v1/assets", headers=headers, json={"name": "Absent target"})
        asset.raise_for_status()
        asset_id = asset.json()["asset_id"]
        for index in range(3):
            image, _ = _case(index, present=False)
            _upload_png(api, headers, asset_id, image)
        created = api.post(
            "/v1/agent/sessions",
            headers=headers,
            json={
                "asset_id": asset_id,
                "message": f"Create a segmentation dataset of {ABSENT_PROMPT}.",
                "sample_size": 3,
                "max_iterations": 2,
            },
        )
        created.raise_for_status()
        session = created.json()
        confirmed = api.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm",
            headers=headers,
            json={},
        )
        confirmed.raise_for_status()
        body = confirmed.json()
        last_message = body["messages"][-1]["content"]
        return {
            "status": body["status"],
            "last_message": last_message,
            "message_ok": "can't segment reliably" in last_message,
            "prompt": body.get("prompt"),
        }


def _run(args) -> dict:
    import tempfile

    provider = LocalSam3Provider()
    client = OllamaVlmClient(
        model=args.model,
        base_url=args.ollama_url,
        timeout_sec=args.vlm_timeout,
        num_ctx=2048,
        num_predict=96,
    )
    report: dict = {
        "model": args.model,
        "prompt": TARGET_PROMPT,
        "threshold": args.threshold,
        "low_threshold": args.low_threshold,
        "minimum_prediction_score": args.min_prediction_score,
        "cases": [],
        "low_threshold_check": {},
        "absent_prompt_check": {},
    }
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
        root = Path(temp)
        validator, core_validator = _build_validator(
            root, client, args.min_prediction_score
        )
        try:
            started = time.perf_counter()
            present_count = max(1, args.images - max(1, args.images // 3))
            for index in range(args.images):
                present = index < present_count
                image, ground_truth = _case(index, present)
                sample = CalibrationSample(
                    image_id=f"case-{index:02d}-{ 'present' if present else 'absent'}",
                    image=image,
                )
                predictions = provider.predict(
                    image, TARGET_PROMPT, threshold=args.threshold
                )
                iou = _best_iou(predictions, ground_truth)
                false_positives = _false_positives(predictions, ground_truth)
                trace = validator.validate_with_trace(
                    sample, predictions, TARGET_PROMPT, threshold=args.threshold
                )
                expected = bool(predictions) and iou >= 0.5 and false_positives == 0
                report["cases"].append(
                    {
                        "image_id": sample.image_id,
                        "present": present,
                        "prediction_count": len(predictions),
                        "best_iou": round(iou, 4),
                        "false_positive_masks": false_positives,
                        "vlm_accepted": trace.accepted,
                        "vlm_confidence": trace.confidence,
                        "vlm_reasoning": trace.reasoning,
                        "expected_good_segmentation": expected,
                        "agrees": trace.accepted == expected,
                    }
                )
                print(
                    f"case {index + 1}/{args.images}: iou={iou:.3f} "
                    f"vlm={trace.accepted} expected={expected}",
                    flush=True,
                )

            low_image, low_truth = _case(0, present=True)
            low_predictions = provider.predict(
                low_image, TARGET_PROMPT, threshold=args.low_threshold
            )
            low_trace = validator.validate_with_trace(
                CalibrationSample(image_id="low-threshold", image=low_image),
                low_predictions,
                TARGET_PROMPT,
                threshold=args.low_threshold,
            )
            report["low_threshold_check"] = {
                "prediction_count": len(low_predictions),
                "best_iou": round(_best_iou(low_predictions, low_truth), 4),
                "false_positive_masks": _false_positives(low_predictions, low_truth),
                "vlm_accepted": low_trace.accepted,
                "vlm_confidence": low_trace.confidence,
                "vlm_reasoning": low_trace.reasoning,
                "rejected": not low_trace.accepted,
            }

            report["absent_prompt_check"] = _absent_prompt_check(root, provider)
        finally:
            try:
                client.unload()
                report["vlm_unloaded"] = True
            except Exception as exc:
                report["vlm_unloaded"] = False
                report["vlm_unload_error"] = str(exc)

    cases = report["cases"]
    agreements = sum(1 for item in cases if item["agrees"])
    false_accepts = sum(
        1 for item in cases if item["vlm_accepted"] and not item["expected_good_segmentation"]
    )
    false_rejects = sum(
        1 for item in cases if not item["vlm_accepted"] and item["expected_good_segmentation"]
    )
    report["summary"] = {
        "case_count": len(cases),
        "vlm_calls": core_validator.calls_made,
        "agreement_count": agreements,
        "agreement_rate": round(agreements / len(cases), 4) if cases else None,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "wall_sec": round(time.perf_counter() - started, 2),
        "low_threshold_rejected": report["low_threshold_check"].get("rejected", False),
        "absent_prompt_message_ok": report["absent_prompt_check"].get("message_ok", False),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--low-threshold", type=float, default=0.02)
    parser.add_argument("--min-prediction-score", type=float, default=0.2)
    parser.add_argument("--model", default="qwen3-vl:4b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--vlm-timeout", type=float, default=180)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = _run(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
