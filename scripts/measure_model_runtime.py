"""Measure the real local model journey on the configured device.

This is a measurement harness, not a test. It prints machine-readable JSON and a
short human summary so a baseline can be recorded before changing the runtime.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from agentic_cv.batch_labeling import LabelImage, Sam3BatchLabeler
from agentic_cv.dataset import DatasetCategory
from agentic_cv.embeddings import HuggingFaceEmbeddingProvider
from agentic_cv.model_resources import ModelResourceManager
from agentic_cv.model_runtime import ModelRuntimeSettings
from agentic_cv.sam3 import LocalSam3Provider


def _make_images(root: Path, count: int, size: tuple[int, int] = (640, 480)) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        image = Image.new("RGB", size, (42, 52, 64))
        draw = ImageDraw.Draw(image)
        x = 40 + (index * 23) % (size[0] - 180)
        y = 80 + (index * 17) % (size[1] - 220)
        draw.rectangle((x, y, x + 120, y + 170), fill=(240, 190, 40))
        draw.ellipse((x + 30, y - 52, x + 90, y + 8), fill=(220, 170, 130))
        draw.rectangle((x + 8, y + 168, x + 52, y + 240), fill=(30, 30, 120))
        draw.rectangle((x + 68, y + 168, x + 112, y + 240), fill=(30, 30, 120))
        path = root / f"frame-{index:03d}.png"
        image.save(path)
        paths.append(path)
    return paths


def _memory(torch) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "free_mb": 0.0, "total_mb": 0.0}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "free_mb": round(free / 1024**2, 1),
        "total_mb": round(total / 1024**2, 1),
    }


def _reset_peak(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _peak(torch) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"peak_allocated_mb": 0.0, "peak_reserved_mb": 0.0}
    return {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=20)
    parser.add_argument("--prompt", default="worker")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--keep-images", type=Path, default=None)
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--keep-embedding-loaded", action="store_true")
    parser.add_argument("--managed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headroom-mb", type=int, default=512)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    import torch

    settings = ModelRuntimeSettings.from_environment()
    report: dict[str, object] = {
        "device": settings.device,
        "precision": settings.precision,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "image_count": args.images,
        "prompt": args.prompt,
        "threshold": args.threshold,
    }
    with TemporaryDirectory() as temp:
        image_root = args.keep_images or Path(temp) / "images"
        images = _make_images(image_root, args.images)
        embedding = None
        sam3 = None
        manager = None
        if args.managed:
            manager = ModelResourceManager(
                device=settings.device,
                headroom_mb=args.headroom_mb,
                idle_unload_sec=300,
            )
            report["resource_policy"] = {
                "managed": True,
                "headroom_mb": args.headroom_mb,
                "keep_embedding_loaded": args.keep_embedding_loaded,
            }
        else:
            report["resource_policy"] = {"managed": False}
        try:
            if not args.skip_embedding:
                embedding = HuggingFaceEmbeddingProvider(
                    settings.embedding_config(),
                    resources=manager,
                    estimated_vram_mb=2304,
                )
                _reset_peak(torch)
                started = time.perf_counter()
                for path in images:
                    with Image.open(path) as source:
                        embedding.embed_image(source.convert("RGB"))
                embedding.embed_text("worker without helmet")
                report["embedding"] = {
                    "wall_sec": round(time.perf_counter() - started, 2),
                    "memory_after": _memory(torch),
                    **_peak(torch),
                }
                if not args.keep_embedding_loaded:
                    del embedding
                    embedding = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            sam3 = LocalSam3Provider(
                settings.sam3_config(),
                resources=manager,
                estimated_vram_mb=2048,
            )
            labeler = Sam3BatchLabeler(sam3)
            _reset_peak(torch)
            started = time.perf_counter()
            result = labeler.label(
                [LabelImage(image_id=path.stem, path=path) for path in images],
                prompt=args.prompt,
                threshold=args.threshold,
                category=DatasetCategory(category_id=0, name=args.prompt),
                task="segmentation",
            )
            wall = time.perf_counter() - started
            report["sam3"] = {
                "wall_sec": round(wall, 2),
                "per_image_sec": round(wall / len(images), 2),
                "annotations": len(result.dataset.annotations),
                "images_with_annotations": len(
                    {item.image_id for item in result.dataset.annotations}
                ),
                "memory_after": _memory(torch),
                **_peak(torch),
            }
            if manager is not None:
                status = manager.status().as_dict()
                report["resource_status"] = {
                    "free_mb": status["free_mb"],
                    "total_mb": status["total_mb"],
                    "models": status["models"],
                    "events": status["events"],
                }
        finally:
            if embedding is not None:
                embedding.close()
            if sam3 is not None:
                sam3.close()
            if manager is not None:
                report["final_resource_events"] = manager.status().as_dict()["events"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(json.dumps(report, indent=2))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
