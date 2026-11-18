import pytest

from agentic_cv.model_runtime import (
    ModelRuntimeSettings,
    resolve_device,
    resolve_precision,
)


def test_device_defaults_to_cuda_when_available(monkeypatch):
    monkeypatch.delenv("AGENTIC_CV_DEVICE", raising=False)
    monkeypatch.setattr("agentic_cv.model_runtime._cuda_available", lambda: True)

    assert resolve_device() == "cuda"


def test_device_defaults_to_cpu_without_cuda(monkeypatch):
    monkeypatch.delenv("AGENTIC_CV_DEVICE", raising=False)
    monkeypatch.setattr("agentic_cv.model_runtime._cuda_available", lambda: False)

    assert resolve_device() == "cpu"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("cpu", "cpu"),
        ("cuda", "cuda"),
        ("cuda:0", "cuda:0"),
        ("CUDA:1", "cuda:1"),
        ("  cpu  ", "cpu"),
    ],
)
def test_device_accepts_supported_values(value, expected):
    assert resolve_device(value) == expected


@pytest.mark.parametrize("value", ["gpu", "cuda:x", "mps", "cuda:-1"])
def test_device_rejects_unsupported_values(value):
    with pytest.raises(ValueError, match="AGENTIC_CV_DEVICE"):
        resolve_device(value)


def test_cpu_defaults_to_float32(monkeypatch):
    monkeypatch.setenv("AGENTIC_CV_DEVICE", "cpu")
    monkeypatch.delenv("AGENTIC_CV_PRECISION", raising=False)

    settings = ModelRuntimeSettings.from_environment()

    assert settings.device == "cpu"
    assert settings.precision == "float32"
    assert settings.embedding_config().device == "cpu"
    assert settings.embedding_config().precision == "float32"
    assert settings.sam3_config().device == "cpu"
    assert settings.sam3_config().precision == "float32"


def test_gpu_defaults_to_float16(monkeypatch):
    monkeypatch.setenv("AGENTIC_CV_DEVICE", "cuda:0")
    monkeypatch.delenv("AGENTIC_CV_PRECISION", raising=False)

    settings = ModelRuntimeSettings.from_environment()

    assert settings.device == "cuda:0"
    assert settings.precision == "float16"


def test_cpu_rejects_half_precision(monkeypatch):
    monkeypatch.setenv("AGENTIC_CV_DEVICE", "cpu")
    monkeypatch.setenv("AGENTIC_CV_PRECISION", "float16")

    with pytest.raises(ValueError, match="float32"):
        ModelRuntimeSettings.from_environment()


def test_invalid_precision_is_rejected(monkeypatch):
    monkeypatch.setenv("AGENTIC_CV_DEVICE", "cuda")
    monkeypatch.setenv("AGENTIC_CV_PRECISION", "int8")

    with pytest.raises(ValueError, match="AGENTIC_CV_PRECISION"):
        resolve_precision("cuda")
