import numpy as np
import pytest
from PIL import Image

from agentic_cv.batch_labeling import LabelImage, Sam3BatchLabeler
from agentic_cv.dataset import DatasetCategory
from agentic_cv.sam3 import Sam3Prediction


class FakeProvider:
    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        return [Sam3Prediction(
            score=0.92,
            box=(2, 1, 8, 7),
            mask=np.array([
                [False, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, False],
            ]),
        )]


def make_image(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(path)
    return [LabelImage(image_id="frame-1", path=path)]


def test_batch_labeling_preserves_raw_predictions_and_builds_detection_dataset(tmp_path):
    result = Sam3BatchLabeler(FakeProvider()).label(
        make_image(tmp_path), prompt="brown bear", threshold=0.5,
        category=DatasetCategory(category_id=1, name="bear"), task="detection",
    )

    assert len(result.raw_predictions) == 1
    assert result.dataset.annotations[0].bbox == (2, 1, 6, 6)
    assert result.dataset.annotations[0].segmentation == ()


def test_segmentation_batch_converts_masks_to_polygons(tmp_path):
    result = Sam3BatchLabeler(FakeProvider()).label(
        make_image(tmp_path), prompt="bear", threshold=0.6,
        category=DatasetCategory(category_id=1, name="bear"), task="segmentation",
    )

    assert len(result.dataset.annotations[0].segmentation) == 1
    assert len(result.dataset.annotations[0].segmentation[0]) >= 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"images": [], "prompt": "bear", "threshold": 0.5, "task": "detection"},
        {"images": make_image.__name__, "prompt": " ", "threshold": 0.5, "task": "detection"},
    ],
)
def test_batch_labeling_rejects_invalid_requests(tmp_path, kwargs):
    images = [] if kwargs["images"] == [] else make_image(tmp_path)
    with pytest.raises(ValueError):
        Sam3BatchLabeler(FakeProvider()).label(
            images, prompt=kwargs["prompt"], threshold=kwargs["threshold"],
            category=DatasetCategory(category_id=1, name="bear"), task=kwargs["task"],
        )
