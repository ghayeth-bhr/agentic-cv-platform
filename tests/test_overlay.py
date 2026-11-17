import numpy as np
import pytest
from PIL import Image

from agentic_cv.overlay import render_prediction_overlay
from agentic_cv.sam3 import Sam3Prediction


def test_render_prediction_overlay_saves_mask_and_box(tmp_path):
    image = Image.new("RGB", (8, 6), "white")
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:4, 2:5] = True
    output = render_prediction_overlay(
        image,
        [Sam3Prediction(score=0.91, box=(2, 2, 5, 4), mask=mask)],
        tmp_path / "nested" / "overlay.png",
    )

    assert output.exists()
    rendered = Image.open(output).convert("RGB")
    assert rendered.size == image.size
    assert rendered.getpixel((3, 3)) != (255, 255, 255)
    assert rendered.getpixel((2, 2)) != (255, 255, 255)


def test_render_prediction_overlay_allows_empty_predictions(tmp_path):
    output = render_prediction_overlay(
        Image.new("RGB", (2, 2), "white"), [], tmp_path / "overlay.png"
    )

    assert Image.open(output).getpixel((0, 0))[:3] == (255, 255, 255)


def test_render_prediction_overlay_rejects_wrong_mask_dimensions(tmp_path):
    prediction = Sam3Prediction(
        score=0.5,
        box=(0, 0, 1, 1),
        mask=np.zeros((1, 1), dtype=bool),
    )

    with pytest.raises(ValueError, match="dimensions"):
        render_prediction_overlay(Image.new("RGB", (2, 2)), [prediction], tmp_path / "overlay.png")