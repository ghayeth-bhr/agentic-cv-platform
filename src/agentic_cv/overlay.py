"""Render SAM3 predictions into local artifacts for VLM validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .sam3 import Sam3Prediction

_COLORS = ((255, 64, 64), (64, 160, 255), (64, 210, 120), (245, 180, 55))


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
        mask = np.asarray(prediction.mask)
        if mask.shape != (height, width):
            raise ValueError("prediction mask dimensions must match the image")
        color = _COLORS[index % len(_COLORS)]
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