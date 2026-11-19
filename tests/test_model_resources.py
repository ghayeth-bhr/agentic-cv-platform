import threading
import time
from contextlib import nullcontext

import numpy as np
from PIL import Image

from agentic_cv.model_resources import ModelResourceManager
from agentic_cv.sam3 import LocalSam3Provider, Sam3Config


class FakeModel:
    def __init__(self, name):
        self.name = name
        self.loaded = False
        self.close_reasons = []

    @property
    def is_loaded(self):
        return self.loaded

    def load(self):
        self.loaded = True

    def close(self):
        self.loaded = False
        self.close_reasons.append("closed")


def test_lease_serializes_inference_and_closes_other_model_under_pressure():
    memory = {"free": 9000.0, "total": 6000.0}
    empties = []
    manager = ModelResourceManager(
        device="cuda",
        headroom_mb=512,
        memory_probe=lambda: (memory["free"], memory["total"]),
        empty_cache=lambda: empties.append(True),
        idle_unload_sec=0,
    )
    embeddings = FakeModel("siglip2")
    sam3 = FakeModel("sam3")
    manager.register("siglip2", embeddings, estimated_mb=2048)
    manager.register("sam3", sam3, estimated_mb=2048)

    with manager.lease("siglip2"):
        embeddings.load()
    status = manager.status()
    assert status.models["siglip2"]["loaded"] is True

    memory["free"] = 1000.0
    with manager.lease("sam3"):
        assert embeddings.is_loaded is False
        sam3.load()
    assert any("make room for sam3" in event for event in manager.status().events)
    assert empties

    active = []
    overlap = []
    lock = threading.Lock()

    def work(name, model):
        with manager.lease(name):
            model.load()
            with lock:
                active.append(name)
                overlap.append(len(active))
            time.sleep(0.05)
            with lock:
                active.remove(name)

    first = threading.Thread(target=work, args=("siglip2", embeddings))
    second = threading.Thread(target=work, args=("sam3", sam3))
    first.start()
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)
    assert overlap == [1, 1]


def test_idle_model_is_unloaded_on_the_next_lease():
    now = {"value": 100.0}
    manager = ModelResourceManager(
        device="cuda",
        headroom_mb=0,
        memory_probe=lambda: (9000.0, 12000.0),
        empty_cache=lambda: None,
        clock=lambda: now["value"],
        idle_unload_sec=10,
    )
    first = FakeModel("first")
    second = FakeModel("second")
    manager.register("first", first, estimated_mb=1024)
    manager.register("second", second, estimated_mb=1024)

    with manager.lease("first"):
        first.load()
    now["value"] += 11
    with manager.lease("second"):
        assert first.is_loaded is False
        second.load()

    assert any("idle TTL" in event for event in manager.status().events)


class FakeTensorDict(dict):
    def to(self, device):
        return self


class FakeMask:
    def cpu(self):
        return self

    def numpy(self):
        return np.ones((8, 8), dtype=bool)


class FakeSam3Processor:
    def __call__(self, *, images, text, return_tensors):
        assert text == "worker"
        return FakeTensorDict(
            pixel_values=object(),
            original_sizes=np.array([[8, 8]]),
        )

    def post_process_instance_segmentation(
        self, outputs, *, threshold, mask_threshold, target_sizes
    ):
        assert target_sizes == [[8, 8]]
        return [
            {
                "masks": [FakeMask()],
                "boxes": [[1, 1, 7, 7]],
                "scores": [0.9],
            }
        ]


class FakeSam3Model:
    def __call__(self, **inputs):
        return object()


class FakeCuda:
    def is_available(self):
        return False


class FakeTorch:
    cuda = FakeCuda()

    def inference_mode(self):
        return nullcontext()


def test_local_sam3_provider_uses_shared_manager_and_reloads_after_unload():
    calls = []

    def loader(config):
        calls.append(config.device)
        return FakeSam3Model(), FakeSam3Processor(), FakeTorch()

    manager = ModelResourceManager(
        device="cuda",
        memory_probe=lambda: (8000.0, 12000.0),
        empty_cache=lambda: None,
    )
    provider = LocalSam3Provider(
        Sam3Config(precision="float32"),
        loader=loader,
        resources=manager,
        resource_name="sam3-test",
        estimated_vram_mb=2048,
    )
    image = Image.new("RGB", (8, 8))

    predictions = provider.predict(image, "worker")
    assert len(predictions) == 1
    assert manager.status().models["sam3-test"]["loaded"] is True
    assert calls == ["cuda"]

    manager.close("sam3-test")
    assert provider.is_loaded is False
    provider.predict(image, "worker")
    assert calls == ["cuda", "cuda"]


def test_cpu_manager_does_not_probe_memory_or_require_torch():
    manager = ModelResourceManager(
        device="cpu",
        memory_probe=lambda: (_ for _ in ()).throw(AssertionError("probe called")),
    )
    model = FakeModel("cpu")
    manager.register("cpu", model)
    with manager.lease("cpu"):
        model.load()

    status = manager.status()
    assert status.free_mb is None
    assert status.total_mb is None
    assert status.models["cpu"]["loaded"] is True
    manager.close_all()
    assert model.is_loaded is False
