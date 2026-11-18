from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image

from agentic_cv.sam3 import LocalSam3Provider, load_image


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def __float__(self):
        return float(self.value)

    def __iter__(self):
        return iter(self.value)

    def half(self):
        return self

    def tolist(self):
        return self.value

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeInputs(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeTorch:
    @staticmethod
    @contextmanager
    def inference_mode():
        yield

    class cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            pass


class FakeProcessor:
    def __call__(self, **kwargs):
        return FakeInputs(pixel_values=FakeTensor([[1]]), original_sizes=FakeTensor([[2, 3]]))

    def post_process_instance_segmentation(self, outputs, **kwargs):
        return [{
            "masks": [FakeTensor(np.array([[True, False], [False, True]]))],
            "boxes": [FakeTensor([1, 2, 3, 4])],
            "scores": [FakeTensor(0.9)],
        }]


class FakeModel:
    def __call__(self, **inputs):
        return "outputs"


def fake_loader(config):
    return FakeModel(), FakeProcessor(), FakeTorch()


def test_provider_is_lazy_and_maps_predictions():
    provider = LocalSam3Provider(loader=fake_loader)

    assert provider._model is None
    predictions = provider.predict(Image.new("RGB", (3, 2)), "brown bear")

    assert len(predictions) == 1
    assert predictions[0].score == pytest.approx(0.9)
    assert predictions[0].box == (1.0, 2.0, 3.0, 4.0)
    assert predictions[0].mask.shape == (2, 2)


def test_provider_rejects_empty_prompt_and_invalid_threshold():
    provider = LocalSam3Provider(loader=fake_loader)

    with pytest.raises(ValueError, match="prompt"):
        provider.predict(Image.new("RGB", (1, 1)), " ")
    with pytest.raises(ValueError, match="threshold"):
        provider.predict(Image.new("RGB", (1, 1)), "bear", threshold=1.1)


def test_load_image_converts_to_rgb(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGBA", (2, 2), (1, 2, 3, 4)).save(path)

    image = load_image(path)

    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (1, 2, 3)
