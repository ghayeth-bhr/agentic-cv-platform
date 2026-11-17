from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image

from agentic_cv.embeddings import HuggingFaceEmbeddingConfig, HuggingFaceEmbeddingProvider


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeInputs(dict):
    def to(self, device):
        self.device = device
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
        return FakeInputs(**kwargs)


class FakeModel:
    def eval(self):
        return self

    def get_image_features(self, **inputs):
        return FakeTensor([[1, 2, 3]])

    def get_text_features(self, **inputs):
        return FakeTensor([[4, 5, 6]])


def fake_loader(config):
    return FakeModel(), FakeProcessor(), FakeTorch()


def test_provider_lazily_embeds_images_and_text():
    provider = HuggingFaceEmbeddingProvider(loader=fake_loader)

    image_embedding = provider.embed_image(Image.new("RGB", (2, 2)))
    text_embedding = provider.embed_text("brown bear")

    assert image_embedding.tolist() == [1, 2, 3]
    assert text_embedding.tolist() == [4, 5, 6]
    assert provider._model is not None


def test_provider_rejects_empty_text_and_invalid_precision():
    provider = HuggingFaceEmbeddingProvider(loader=fake_loader)
    with pytest.raises(ValueError, match="query"):
        provider.embed_text(" ")
    with pytest.raises(ValueError, match="precision"):
        HuggingFaceEmbeddingConfig(precision="int8")
