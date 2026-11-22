"""Benchmark BotFlow's segmentation pipeline against a hand-labeled COCO set.

For every prompt this script:

1. runs SAM 3 once per image at a low score floor and keeps every prediction
   (so any threshold can be scored offline), timing each image and recording
   peak GPU memory;
2. runs BotFlow's real calibration (``Sam3Calibrator`` with the Qwen3-VL judge,
   the app's default thresholds and agreement) on a few sample images, which
   are then excluded from scoring, exactly as a user's labeled set would not
   contain the images the agent tuned on;
3. scores the remaining images: COCO mask and box mAP, and precision, recall,
   F1 and mask IoU at fixed thresholds, including SAM 3's default (0.5) and
   the threshold BotFlow calibrated.

Usage (needs the ``ml`` extra, a GPU, and Ollama with qwen3-vl:4b)::

    uv run python scripts/benchmark_segmentation.py \\
        --gt C:/path/openimages-taxi-approved-coco.zip \\
        --prompt taxi --prompt "taxi car" --out benchmarks/openimages-taxi

Ground truth must be a COCO ZIP as BotFlow exports it (``annotations.json`` and
``images/``). The class name in the ground truth is ignored; every annotation is
treated as the target class.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import platform
import random
import statistics
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import uuid4

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_cv.agent import _DEFAULT_THRESHOLDS  # noqa: E402  (the app's own grid)
from agentic_cv.calibration import CalibrationSample, Sam3Calibrator  # noqa: E402
from agentic_cv.calibration_validation import default_validator_selection  # noqa: E402
from agentic_cv.model_runtime import ModelRuntimeSettings  # noqa: E402
from agentic_cv.sam3 import LocalSam3Provider, Sam3Config, Sam3Prediction  # noqa: E402

APP_MINIMUM_AGREEMENT = 0.6  # AgentServiceDependencies.minimum_agreement
OPERATING_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MATCH_IOU = 0.5


@dataclass
class GroundTruth:
    coco: dict
    images: dict[int, Image.Image]


def load_ground_truth(path: Path) -> GroundTruth:
    with zipfile.ZipFile(path) as bundle:
        names = bundle.namelist()
        json_name = "annotations.json" if "annotations.json" in names else max(
            (name for name in names if name.endswith(".json")),
            key=lambda name: bundle.getinfo(name).file_size,
        )
        coco = json.loads(bundle.read(json_name))
        by_base = {PurePosixPath(name).name: name for name in names}
        images = {}
        for entry in coco["images"]:
            member = by_base[PurePosixPath(entry["file_name"]).name]
            images[entry["id"]] = Image.open(io.BytesIO(bundle.read(member))).convert("RGB")
    for entry in coco["images"]:
        entry["width"], entry["height"] = images[entry["id"]].size
    coco["categories"] = [{"id": 1, "name": "target"}]
    for annotation in coco["annotations"]:
        annotation["category_id"] = 1
        annotation["iscrowd"] = 0
    return GroundTruth(coco=coco, images=images)


def encode(mask: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def run_sam3(provider, truth: GroundTruth, prompt: str, floor: float, keep_raw: set[int]):
    import torch

    torch.cuda.reset_peak_memory_stats()
    detections, timings, samples_raw = [], [], {}
    for index, (image_id, image) in enumerate(sorted(truth.images.items()), start=1):
        torch.cuda.synchronize()
        started = time.perf_counter()
        predictions = provider.predict(image, prompt, threshold=floor)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        if image_id in keep_raw:  # full masks only for calibration samples
            samples_raw[image_id] = predictions
        for prediction in predictions:
            x1, y1, x2, y2 = prediction.box
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "score": round(prediction.score, 5),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "segmentation": encode(prediction.mask),
                }
            )
        if index % 10 == 0:
            print(f"  [{prompt}] {index}/{len(truth.images)} images", flush=True)
    return {
        "detections": detections,
        "raw": samples_raw,
        "load_seconds": round(provider.load_seconds, 2),
        # the first call includes CUDA warm-up; report steady state separately
        "seconds_first_image": round(timings[0], 3),
        "seconds_per_image_median": round(statistics.median(timings[1:] or timings), 3),
        "seconds_per_image_mean": round(statistics.fmean(timings[1:] or timings), 3),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20),
    }


def calibrate(selection, samples, raw, prompt: str) -> dict:
    calibrator = Sam3Calibrator(
        provider=None, validator=selection.validator, minimum_agreement=APP_MINIMUM_AGREEMENT
    )
    floor = min(_DEFAULT_THRESHOLDS)
    filtered = [[item for item in predictions if item.score >= floor] for predictions in raw]
    started = time.perf_counter()
    result = calibrator.score(samples, filtered, prompt=prompt, thresholds=list(_DEFAULT_THRESHOLDS))
    return {
        "status": result.status,
        "threshold": result.threshold,
        "seconds": round(time.perf_counter() - started, 1),
        "validator": selection.mode,
        "notice": selection.current_notice(),
        "attempts": [
            {"threshold": attempt.threshold, "accepted": attempt.accepted_count, "of": attempt.sample_count}
            for attempt in result.attempts
        ],
    }


def gt_masks(truth: GroundTruth, image_id: int) -> list:
    from pycocotools import mask as mask_utils

    entry = next(item for item in truth.coco["images"] if item["id"] == image_id)
    height, width = entry["height"], entry["width"]
    masks = []
    for annotation in truth.coco["annotations"]:
        if annotation["image_id"] != image_id:
            continue
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, list) and segmentation:
            masks.append(mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width)))
        else:
            x, y, w, h = annotation["bbox"]
            masks.append(mask_utils.frPyObjects([[x, y, x + w, y, x + w, y + h, x, y + h]], height, width)[0])
    return masks


def audit_judge(truth: GroundTruth, raw: dict, image_ids: list[int], prompt: str, out: Path, per_image: int = 6) -> dict:
    """Compare the per-mask VLM verdict with the ground truth, mask by mask.

    A mask counts as a real target when it overlaps a ground-truth mask at
    IoU >= 0.5. Masks are taken as calibration sees them: score >= the lowest
    calibration threshold, highest score first, at most ``per_image``.
    """

    from pycocotools import mask as mask_utils

    from agentic_cv.calibration_validation import build_instance_builder
    from agentic_cv.ollama_vlm import OllamaVlmClient
    from agentic_cv.vlm_validator import VlmValidator, _instance_instruction

    client = OllamaVlmClient(
        model=os.getenv("AGENTIC_CV_VLM_MODEL", "qwen3-vl:4b"),
        base_url=os.getenv("AGENTIC_CV_OLLAMA_URL", "http://localhost:11434"),
        timeout_sec=100.0,
        max_side=768,
    )
    validator = VlmValidator(client, max_calls=10_000)
    builder = build_instance_builder(overlay_root=out / "audit", tenant_id=uuid4(), session_id=uuid4())
    counts = {"judge_yes_truth_yes": 0, "judge_yes_truth_no": 0, "judge_no_truth_yes": 0, "judge_no_truth_no": 0, "unsure": 0}
    started = time.perf_counter()
    for image_id in image_ids:
        gts = gt_masks(truth, image_id)
        floor = min(_DEFAULT_THRESHOLDS)
        candidates = sorted((item for item in raw[image_id] if item.score >= floor), key=lambda item: -item.score)[:per_image]
        sample = CalibrationSample(image_id=str(image_id), image=truth.images[image_id])
        for prediction in candidates:
            rle = encode(prediction.mask)
            best = float(np.max(mask_utils.iou([rle], gts, [0] * len(gts)))) if gts else 0.0
            truth_yes = best >= MATCH_IOU
            artifact = builder(sample, prediction)
            judgment = validator.validate(overlay_uri=artifact.uri, prompt=_instance_instruction(prompt))
            if judgment.confidence < 0.5:
                counts["unsure"] += 1
                continue
            judge_yes = judgment.target_present and not judgment.false_positive
            counts[f"judge_{'yes' if judge_yes else 'no'}_truth_{'yes' if truth_yes else 'no'}"] += 1
    validator.release()
    decided = sum(value for key, value in counts.items() if key != "unsure")
    agree = counts["judge_yes_truth_yes"] + counts["judge_no_truth_no"]
    said_no = counts["judge_no_truth_no"] + counts["judge_no_truth_yes"]
    really_no = counts["judge_no_truth_no"] + counts["judge_yes_truth_no"]
    return {
        "images": len(image_ids),
        "masks": decided + counts["unsure"],
        **counts,
        "accuracy": round(agree / decided, 4) if decided else None,
        # of the masks the judge rejected, how many were really wrong
        "rejection_precision": round(counts["judge_no_truth_no"] / said_no, 4) if said_no else None,
        # of the really wrong masks, how many the judge caught
        "wrong_mask_recall": round(counts["judge_no_truth_no"] / really_no, 4) if really_no else None,
        "seconds": round(time.perf_counter() - started, 1),
    }


def coco_metrics(truth: GroundTruth, detections: list[dict], image_ids: list[int]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # COCOeval rewrites annotations in place (polygons become RLE), so it only
    # ever sees copies: later scoring must read the original ground truth.
    gt = COCO()
    gt.dataset = copy.deepcopy({
        "images": [entry for entry in truth.coco["images"] if entry["id"] in image_ids],
        "annotations": [item for item in truth.coco["annotations"] if item["image_id"] in image_ids],
        "categories": truth.coco["categories"],
    })
    for annotation in gt.dataset["annotations"]:
        if "area" not in annotation:
            annotation["area"] = float(annotation["bbox"][2] * annotation["bbox"][3])
    gt.createIndex()
    kept = copy.deepcopy([item for item in detections if item["image_id"] in image_ids])
    out = {}
    for kind in ("segm", "bbox"):
        if not kept:
            out[kind] = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR100": 0.0}
            continue
        evaluator = COCOeval(gt, gt.loadRes(copy.deepcopy(kept)), kind)
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        stats = evaluator.stats
        out[kind] = {
            "AP": round(float(stats[0]), 4),
            "AP50": round(float(stats[1]), 4),
            "AP75": round(float(stats[2]), 4),
            "AP_small": round(float(stats[3]), 4),
            "AP_medium": round(float(stats[4]), 4),
            "AP_large": round(float(stats[5]), 4),
            "AR100": round(float(stats[8]), 4),
        }
    return out


def operating_point(truth: GroundTruth, detections: list[dict], image_ids: list[int], threshold: float) -> dict:
    """Greedy one-to-one matching by score at mask IoU >= 0.5."""

    from pycocotools import mask as mask_utils

    gt_by_image: dict[int, list[dict]] = {}
    for annotation in truth.coco["annotations"]:
        gt_by_image.setdefault(annotation["image_id"], []).append(annotation)
    tp = fp = fn = 0
    ious = []
    for image_id in image_ids:
        entry = next(item for item in truth.coco["images"] if item["id"] == image_id)
        height, width = entry["height"], entry["width"]
        gts = []
        for annotation in gt_by_image.get(image_id, []):
            segmentation = annotation.get("segmentation")
            if isinstance(segmentation, list) and segmentation:
                rles = mask_utils.frPyObjects(segmentation, height, width)
                gts.append(mask_utils.merge(rles))
            elif isinstance(segmentation, dict) and "counts" in segmentation:
                rle = segmentation
                if isinstance(rle["counts"], list):
                    rle = mask_utils.frPyObjects(rle, height, width)
                gts.append(rle)
            else:
                x, y, w, h = annotation["bbox"]
                gts.append(mask_utils.frPyObjects([[x, y, x + w, y, x + w, y + h, x, y + h]], height, width)[0])
        preds = sorted(
            (item for item in detections if item["image_id"] == image_id and item["score"] >= threshold),
            key=lambda item: -item["score"],
        )
        if not preds:
            fn += len(gts)
            continue
        if not gts:
            fp += len(preds)
            continue
        matrix = mask_utils.iou([item["segmentation"] for item in preds], gts, [0] * len(gts))
        used = set()
        for row in np.atleast_2d(matrix):
            best, best_iou = None, MATCH_IOU
            for column, value in enumerate(row):
                if column not in used and value >= best_iou:
                    best, best_iou = column, value
            if best is None:
                fp += 1
            else:
                used.add(best)
                tp += 1
                ious.append(float(best_iou))
        fn += len(gts) - len(used)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_mask_iou_of_matches": round(statistics.fmean(ious), 4) if ious else None,
    }


def bootstrap_f1(truth, detections, image_ids, threshold, rounds=500, seed=0) -> list[float]:
    """95% interval of F1 by resampling images (per-image counts, then pooled)."""

    per_image = {image_id: operating_point(truth, detections, [image_id], threshold) for image_id in image_ids}
    rng = random.Random(seed)
    values = []
    for _ in range(rounds):
        picked = [per_image[rng.choice(image_ids)] for _ in image_ids]
        tp = sum(item["tp"] for item in picked)
        fp = sum(item["fp"] for item in picked)
        fn = sum(item["fn"] for item in picked)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * p * r / (p + r) if p + r else 0.0)
    values.sort()
    return [round(values[int(0.025 * rounds)], 4), round(values[int(0.975 * rounds) - 1], 4)]


def markdown(report: dict) -> str:
    lines = [
        f"# Segmentation benchmark: {report['dataset']['name']}",
        "",
        f"- Ground truth: {report['dataset']['images']} images, {report['dataset']['annotations']} masks "
        f"({report['dataset']['scored_images']} images / {report['dataset']['scored_annotations']} masks scored; "
        f"{len(report['dataset']['calibration_image_ids'])} calibration images excluded)",
        f"- Hardware: {report['environment']['gpu']}; SAM 3 {report['environment']['precision']}",
        f"- Matching: one-to-one by score at mask IoU >= {MATCH_IOU}",
        f"- Calibration judge: {report['settings']['vlm_judge']} (Qwen3-VL), app thresholds {report['settings']['calibration_thresholds']}, agreement {report['settings']['minimum_agreement']}",
        "",
        "| Prompt | Calibrated threshold | Mask AP | Mask AP50 | Box AP50 | F1 @0.5 (plain SAM 3) | F1 @calibrated (BotFlow) | 95% CI | Mask IoU of matches | s / image | Peak GPU MB |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in report["prompts"]:
        default = item["operating_points"]["0.5"]
        calibrated = item.get("calibrated_point")
        matched = calibrated or default
        if calibrated:
            threshold, f1, interval = item["calibration"]["threshold"], f"{calibrated['f1']:.3f}", item.get("calibrated_f1_ci95", "")
        else:
            threshold, f1, interval = f"refused ({item['calibration']['status']})", "no labels", ""
        lines.append(
            f"| {item['prompt']} | {threshold} | {item['coco']['segm']['AP']:.3f} | "
            f"{item['coco']['segm']['AP50']:.3f} | {item['coco']['bbox']['AP50']:.3f} | {default['f1']:.3f} | "
            f"{f1} | {interval} | {matched['mean_mask_iou_of_matches']:.3f} | "
            f"{item['speed']['seconds_per_image_median']} | {item['speed']['peak_gpu_memory_mb']} |"
        )
    lines += ["", "## Precision / recall by threshold", ""]
    for item in report["prompts"]:
        lines += [f"### Prompt \"{item['prompt']}\"", "", "| Threshold | Precision | Recall | F1 | TP | FP | FN |", "|---|---|---|---|---|---|---|"]
        for key, point in item["operating_points"].items():
            lines.append(
                f"| {key} | {point['precision']:.3f} | {point['recall']:.3f} | {point['f1']:.3f} | {point['tp']} | {point['fp']} | {point['fn']} |"
            )
        attempts = ", ".join(f"{a['threshold']}: {a['accepted']}/{a['of']}" for a in item["calibration"]["attempts"])
        lines += ["", f"Calibration ({item['calibration']['validator']}, {item['calibration']['seconds']} s): {attempts} -> {item['calibration']['status']}", ""]
        audit = item.get("judge_audit")
        if audit:
            lines += [
                f"Judge audit ({audit['images']} images, {audit['masks']} masks vs ground truth): accuracy {audit['accuracy']}, "
                f"caught {audit['judge_no_truth_no']} of {audit['judge_no_truth_no'] + audit['judge_yes_truth_no']} wrong masks, "
                f"wrongly rejected {audit['judge_no_truth_yes']} real ones, unsure {audit['unsure']}.",
                "",
            ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gt", type=Path, required=True, help="COCO ZIP with ground truth")
    parser.add_argument("--prompt", action="append", required=True, help="text prompt (repeatable)")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--name", default=None, help="dataset name for the report")
    parser.add_argument("--calibration-samples", type=int, default=5)
    parser.add_argument("--floor", type=float, default=0.05, help="lowest score kept from SAM 3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N images (smoke test)")
    parser.add_argument("--judge-audit", type=int, default=0, help="also check the per-mask VLM judge against ground truth on N scored images")
    args = parser.parse_args()

    import torch

    truth = load_ground_truth(args.gt)
    if args.limit:
        keep = sorted(truth.images)[: args.limit]
        truth.images = {key: truth.images[key] for key in keep}
        truth.coco["images"] = [item for item in truth.coco["images"] if item["id"] in truth.images]
        truth.coco["annotations"] = [item for item in truth.coco["annotations"] if item["image_id"] in truth.images]
    all_ids = sorted(truth.images)
    calibration_ids = sorted(random.Random(args.seed).sample(all_ids, min(args.calibration_samples, len(all_ids) - 1)))
    scored_ids = [image_id for image_id in all_ids if image_id not in calibration_ids]
    audit_ids = sorted(random.Random(args.seed + 1).sample(scored_ids, min(args.judge_audit, len(scored_ids))))
    args.out.mkdir(parents=True, exist_ok=True)

    settings = ModelRuntimeSettings.from_environment()
    provider = LocalSam3Provider(Sam3Config(device=settings.device, precision=settings.precision))
    runs = {}
    for prompt in args.prompt:
        print(f"SAM 3 on {len(all_ids)} images, prompt {prompt!r}", flush=True)
        runs[prompt] = run_sam3(provider, truth, prompt, args.floor, set(calibration_ids) | set(audit_ids))
    provider.close()  # free the GPU for the VLM, as the app does

    samples = [CalibrationSample(image_id=str(image_id), image=truth.images[image_id]) for image_id in calibration_ids]
    report_prompts = []
    for prompt, run in runs.items():
        print(f"Calibrating {prompt!r} on {len(samples)} sample images", flush=True)
        # One validator per prompt, like one per agent session in the app
        # (its call budget and failure notice are per session).
        selection = default_validator_selection(
            overlay_root=args.out / "overlays", tenant_id=uuid4(), session_id=uuid4()
        )
        calibration = calibrate(selection, samples, [run["raw"][image_id] for image_id in calibration_ids], prompt)
        vlm_model = getattr(selection, "model", None)
        if selection.release is not None:
            selection.release()
        detections = run["detections"]
        points = {str(t): operating_point(truth, detections, scored_ids, t) for t in OPERATING_THRESHOLDS}
        entry = {
            "prompt": prompt,
            "calibration": calibration,
            "coco": coco_metrics(truth, detections, scored_ids),
            "operating_points": points,
            "speed": {key: value for key, value in run.items() if key not in ("detections", "raw")},
        }
        if calibration["threshold"] is not None:
            entry["calibrated_point"] = operating_point(truth, detections, scored_ids, calibration["threshold"])
            entry["calibrated_f1_ci95"] = bootstrap_f1(truth, detections, scored_ids, calibration["threshold"], seed=args.seed)
        entry["default_f1_ci95"] = bootstrap_f1(truth, detections, scored_ids, 0.5, seed=args.seed)
        if audit_ids:
            print(f"Auditing the per-mask judge for {prompt!r} on {len(audit_ids)} images", flush=True)
            entry["judge_audit"] = audit_judge(truth, run["raw"], audit_ids, prompt, args.out)
        report_prompts.append(entry)
        safe = "".join(ch if ch.isalnum() else "-" for ch in prompt)
        (args.out / f"predictions-{safe}.json").write_text(json.dumps(detections))
    report = {
        "audit_image_ids": audit_ids,
        "dataset": {
            "name": args.name or args.gt.stem,
            "source": str(args.gt.name),
            "images": len(all_ids),
            "annotations": len(truth.coco["annotations"]),
            "calibration_image_ids": calibration_ids,
            "scored_images": len(scored_ids),
            "scored_annotations": sum(1 for item in truth.coco["annotations"] if item["image_id"] in scored_ids),
        },
        "settings": {
            "floor": args.floor,
            "seed": args.seed,
            "calibration_thresholds": list(_DEFAULT_THRESHOLDS),
            "minimum_agreement": APP_MINIMUM_AGREEMENT,
            "vlm_judge": os.getenv("AGENTIC_CV_VLM_JUDGE", "instance"),
            "match_iou": MATCH_IOU,
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "device": settings.device,
            "precision": settings.precision,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "vlm": vlm_model,
        },
        "prompts": report_prompts,
    }
    (args.out / "results.json").write_text(json.dumps(report, indent=2))
    (args.out / "results.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
