import numpy as np
import pytest
from PIL import Image
from uuid import uuid4

from agentic_cv.click_refinement import ClickPoint, ClickRefinementService
from agentic_cv.sam3 import Sam3Prediction


class FakeClickProvider:
    def __init__(self):
        self.calls = []

    def refine(self, image, points):
        self.calls.append(points)
        return [Sam3Prediction(score=0.9, box=(0, 0, 2, 2), mask=np.ones((2, 2), dtype=bool))]


def test_click_session_increments_revision_and_calls_provider():
    provider = FakeClickProvider()
    service = ClickRefinementService(provider)
    initial = service.create(image_id="frame-1", image=Image.new("RGB", (10, 10)))

    updated = service.click(
        initial.session_id,
        expected_revision=0,
        point=ClickPoint(x=4, y=5, label="positive"),
    )

    assert updated.revision == 1
    assert updated.points == (ClickPoint(x=4, y=5, label="positive"),)
    assert len(updated.predictions) == 1
    assert provider.calls[0] == updated.points


def test_click_session_rejects_stale_revision_and_out_of_bounds_point():
    service = ClickRefinementService(FakeClickProvider())
    initial = service.create(image_id="frame-1", image=Image.new("RGB", (10, 10)))
    service.click(initial.session_id, expected_revision=0, point=ClickPoint(x=1, y=1, label="negative"))

    with pytest.raises(ValueError, match="current revision"):
        service.click(initial.session_id, expected_revision=0, point=ClickPoint(x=2, y=2, label="positive"))
    with pytest.raises(ValueError, match="inside image"):
        service.click(initial.session_id, expected_revision=1, point=ClickPoint(x=10, y=2, label="positive"))


def test_click_session_validates_identity_and_unknown_session():
    service = ClickRefinementService(FakeClickProvider())
    with pytest.raises(ValueError, match="image_id"):
        service.create(image_id=" ", image=Image.new("RGB", (1, 1)))
    with pytest.raises(KeyError, match="unknown"):
        service.get(uuid4())
