"""Point-prompted refinement: embedding cache, best-mask choice, fallback."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from agentic_cv.click_refinement import ClickPoint, ClickRefinementService
from agentic_cv.errors import ModelUnavailableError
from agentic_cv.sam3 import Sam3Prediction
from agentic_cv.sam3_points import Sam3PointProvider, Sam3PointRefinementProvider


class _Array:
    """Just enough of a torch tensor for the provider."""

    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, item):
        return _Array(self.value[item])


class _Pixels:
    def to(self, dtype):
        return self


class _Batch(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __call__(self, images=None, input_points=None, input_labels=None, return_tensors=None):
        batch = _Batch(pixel_values=_Pixels(), original_sizes=[[images.height, images.width]])
        if input_points is not None:
            batch["input_points"] = input_points
            batch["input_labels"] = input_labels
        return batch

    def post_process_masks(self, masks, original_sizes):
        # Like transformers: one (objects, 3, H, W) array per image.
        return [masks.value[index] for index in range(masks.value.shape[0])]


class FakeTracker:
    """Three candidates: a tiny one, the object (best IoU), and a large one."""

    def __init__(self):
        self.embeddings = 0
        self.calls = []

    def get_image_embeddings(self, pixel_values):
        self.embeddings += 1
        return "embedding"

    def __call__(self, *, input_points, input_labels, image_embeddings, multimask_output):
        self.calls.append((input_points, input_labels, image_embeddings))
        masks = np.zeros((1, 1, 3, 20, 30), dtype=bool)
        masks[0, 0, 0, 0:2, 0:2] = True
        masks[0, 0, 1, 5:15, 10:20] = True
        masks[0, 0, 2, :, :] = True
        return type("Out", (), {"pred_masks": _Array(masks), "iou_scores": _Array([[[0.2, 0.95, 0.5]]])})()


def make_provider():
    tracker = FakeTracker()
    provider = Sam3PointProvider(loader=lambda config: (tracker, FakeProcessor(), None))
    return provider, tracker


def test_point_segmentation_picks_the_best_mask_and_caches_the_embedding():
    provider, tracker = make_provider()
    image = Image.new("RGB", (30, 20))
    first = provider.segment(image, (ClickPoint(x=12, y=8, label="positive"),))
    second = provider.segment(
        image,
        (ClickPoint(x=12, y=8, label="positive"), ClickPoint(x=2, y=2, label="negative")),
    )
    assert tracker.embeddings == 1  # the image is embedded once
    assert tracker.calls[1][0] == [[[[12.0, 8.0], [2.0, 2.0]]]]
    assert tracker.calls[1][1] == [[[1, 0]]]
    for result in (first, second):
        assert len(result) == 1
        assert result[0].box == (10.0, 5.0, 20.0, 15.0)
        assert result[0].score == pytest.approx(0.95)
        assert result[0].mask.sum() == 100
    # A different image is embedded again.
    provider.segment(Image.new("RGB", (30, 20)), (ClickPoint(x=1, y=1, label="positive"),))
    assert tracker.embeddings == 2


def test_point_refinement_can_add_an_object_the_text_prompt_missed():
    provider, _ = make_provider()

    class EmptyText:
        def refine_with_prompt(self, image, points, prompt):
            return []

    service = ClickRefinementService(Sam3PointRefinementProvider(provider, fallback=EmptyText()))
    session = service.create(image_id="frame", image=Image.new("RGB", (30, 20)), prompt="pallet", category_id=1)
    snapshot = service.click(session.session_id, expected_revision=0, point=ClickPoint(x=12, y=8, label="positive"))
    assert len(snapshot.predictions) == 1


def test_falls_back_to_ranked_text_masks_when_the_tracker_cannot_load():
    def broken(config):
        raise ModelUnavailableError("no tracker")

    class Ranked:
        def __init__(self):
            self.calls = 0

        def refine_with_prompt(self, image, points, prompt):
            self.calls += 1
            return [Sam3Prediction(score=0.5, box=(0, 0, 1, 1), mask=np.ones((1, 1), dtype=bool))]

    ranked = Ranked()
    refinement = Sam3PointRefinementProvider(Sam3PointProvider(loader=broken), fallback=ranked)
    point = (ClickPoint(x=0, y=0, label="positive"),)
    assert refinement.refine_with_prompt(Image.new("RGB", (2, 2)), point, "cars")
    assert refinement.mode == "ranked"
    refinement.refine_with_prompt(Image.new("RGB", (2, 2)), point, "cars")
    assert ranked.calls == 2

    strict = Sam3PointRefinementProvider(Sam3PointProvider(loader=broken))
    with pytest.raises(ModelUnavailableError):
        strict.refine_with_prompt(Image.new("RGB", (2, 2)), point, "cars")
