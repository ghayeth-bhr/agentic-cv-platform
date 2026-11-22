"""Render SAM3 predictions into local artifacts for VLM validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .dataset import Annotation
from .sam3 import Sam3Prediction

_COLORS = ((255, 64, 64), (64, 160, 255), (64, 210, 120), (245, 180, 55))


def class_color(category_id: int) -> tuple[int, int, int]:
    """Stable display color for a dataset category."""

    return _COLORS[abs(int(category_id)) % len(_COLORS)]


def render_prediction_overlay(
    image: Image.Image,
    predictions: list[Sam3Prediction],
    output_path: Path,
) -> Path:
    """Save a source image with translucent masks and score-labeled boxes."""

    canvas = image.convert("RGBA")
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)
    for index, prediction in enumerate(predictions):
        color = _COLORS[index % len(_COLORS)]
        if prediction.mask is not None:
            mask = np.asarray(prediction.mask)
            if mask.shape != (height, width):
                raise ValueError("prediction mask dimensions must match the image")
            tint = Image.new("RGBA", canvas.size, color + (0,))
            tint.putalpha(Image.fromarray(np.where(mask.astype(bool), 96, 0).astype(np.uint8)))
            canvas = Image.alpha_composite(canvas, tint)
            draw = ImageDraw.Draw(canvas)

        x1, y1, x2, y2 = prediction.box
        if not all(np.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("prediction box coordinates must be finite")
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((max(0, x1), max(0, y1 - 12)), f"{prediction.score:.2f}", fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, format="PNG")
    return output_path


def render_instance_crop(
    image: Image.Image,
    prediction: Sam3Prediction,
    output_path: Path,
    *,
    margin: float = 0.6,
    min_side: int = 320,
) -> Path:
    """Save a crop around one prediction with only that mask outlined in red.

    A small VLM judging a whole street scene with several masks drawn on it
    misses the one mask that covers a plain car. One object per question, with
    some context around it, is a question it can answer.
    """

    width, height = image.size
    x1, y1, x2, y2 = (float(value) for value in prediction.box)
    pad_x = max((x2 - x1) * margin, 16.0)
    pad_y = max((y2 - y1) * margin, 16.0)
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(width, int(x2 + pad_x) + 1), min(height, int(y2 + pad_y) + 1)
    canvas = image.convert("RGBA")
    if prediction.mask is not None:
        mask = np.asarray(prediction.mask).astype(bool)
        if mask.shape != (height, width):
            raise ValueError("prediction mask dimensions must match the image")
        tint = Image.new("RGBA", canvas.size, (255, 40, 40, 0))
        tint.putalpha(Image.fromarray(np.where(mask, 70, 0).astype(np.uint8)))
        canvas = Image.alpha_composite(canvas, tint)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=3)
    crop = canvas.crop((left, top, right, bottom)).convert("RGB")
    shortest = min(crop.size)
    if shortest < min_side:
        scale = min_side / max(shortest, 1)
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, format="PNG")
    return output_path


def render_annotation_overlay(
    image: Image.Image,
    annotations: list[Annotation],
    *,
    class_names: dict[int, str] | None = None,
) -> Image.Image:
    """Return a source image with boxes, polygons, class names, and scores.

    This is the server-side twin of the browser gallery overlay. It exists so
    a running label job can expose a finished image before a dataset version
    has been built.
    """

    canvas = image.convert("RGBA")
    fill_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill_layer)
    for annotation in annotations:
        color = class_color(annotation.category_id)
        for polygon in annotation.segmentation:
            if not polygon or len(polygon) < 6 or len(polygon) % 2:
                continue
            points = [
                (float(polygon[index]), float(polygon[index + 1]))
                for index in range(0, len(polygon), 2)
            ]
            fill_draw.polygon(points, fill=color + (70,))
    canvas = Image.alpha_composite(canvas, fill_layer)
    draw = ImageDraw.Draw(canvas)

    for annotation in annotations:
        color = class_color(annotation.category_id)
        for polygon in annotation.segmentation:
            if not polygon or len(polygon) < 6 or len(polygon) % 2:
                continue
            points = [
                (float(polygon[index]), float(polygon[index + 1]))
                for index in range(0, len(polygon), 2)
            ]
            draw.line(points + [points[0]], fill=color + (220,), width=2, joint="curve")
        x, y, width, height = annotation.bbox
        draw.rectangle((x, y, x + width, y + height), outline=color, width=3)
        name = (class_names or {}).get(
            annotation.category_id, f"class {annotation.category_id}"
        )
        score = "" if annotation.score is None else f" {annotation.score:.2f}"
        draw.text((max(0, x), max(0, y - 14)), f"{name}{score}", fill=color)
    return canvas.convert("RGB")