import json
import socket
import sys
import types
from urllib.error import URLError

from agentic_cv import doctor
from agentic_cv.doctor import CheckResult, doctor_exit_code, format_report
from agentic_cv.model_runtime import ModelRuntimeSettings


def test_python_check_reports_current_interpreter():
    result = doctor.check_python()

    assert result.name == "Python"
    assert result.status == "ok"


def test_format_report_and_exit_code():
    checks = [
        CheckResult("PyTorch", "ok", "torch 2.6.0+cu124"),
        CheckResult("SAM 3", "fail", "access denied", "request access and set HF_TOKEN"),
    ]

    report = format_report(checks)

    assert "[ok] PyTorch: torch 2.6.0+cu124" in report
    assert "[fail] SAM 3: access denied" in report
    assert "fix: request access and set HF_TOKEN" in report
    assert "Summary: 1 ok, 0 warn, 1 fail" in report
    assert doctor_exit_code(checks) == 1
    assert doctor_exit_code(checks[:1]) == 0


def test_torch_cpu_build_with_an_nvidia_gpu_warns_and_prints_the_windows_fix(
    monkeypatch,
):
    fake_torch = types.SimpleNamespace(
        __version__="2.6.0+cpu",
        version=types.SimpleNamespace(cuda=None),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        doctor.importlib.util,
        "find_spec",
        lambda name: object() if name == "torch" else None,
    )
    monkeypatch.setattr(
        doctor, "_nvidia_gpu_name", lambda: "NVIDIA GeForce RTX 4050 Laptop GPU"
    )

    result = doctor.check_torch(ModelRuntimeSettings(device="cpu", precision="float32"))

    assert result.status == "warn"
    assert "CPU-only build" in result.detail
    assert "RTX 4050" in result.detail
    assert result.fix is not None
    assert "uv sync --extra ml --refresh" in result.fix
    assert "pyproject.toml" in result.fix


def test_disk_check_creates_and_probes_data_dir(monkeypatch, tmp_path):
    data_dir = tmp_path / "nested" / "data"
    monkeypatch.setattr(
        doctor.shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": 50 * 1024**3})(),
    )

    result = doctor.check_disk(data_dir)

    assert result.status == "ok"
    assert data_dir.is_dir()
    assert not (data_dir / ".doctor-write-test").exists()


def test_port_check_warns_when_something_is_listening():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        result = doctor.check_port("127.0.0.1", port)
    finally:
        listener.close()

    assert result.status == "warn"
    assert "already accepting connections" in result.detail


def test_port_check_reports_free_port():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()

    result = doctor.check_port("127.0.0.1", port)

    assert result.status == "ok"
    assert "free" in result.detail


def test_ollama_check_reports_unreachable_service(monkeypatch):
    def unreachable(*args, **kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr(doctor, "urlopen", unreachable)

    result = doctor.check_ollama("http://127.0.0.1:1", "qwen3-vl:4b")

    assert result.status == "fail"
    assert "not reachable" in result.detail
    assert "ollama" in (result.fix or "").lower()


def test_ollama_check_reports_missing_model(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"models": [{"name": "gemma3:4b"}]}).encode("utf-8")

    monkeypatch.setattr(doctor, "urlopen", lambda *args, **kwargs: Response())

    result = doctor.check_ollama("http://localhost:11434", "qwen3-vl:4b")

    assert result.status == "fail"
    assert "gemma3:4b" in result.detail
    assert "ollama pull qwen3-vl:4b" in (result.fix or "")


def test_run_doctor_returns_ordered_checks_without_network(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTIC_CV_DEVICE", raising=False)
    monkeypatch.delenv("AGENTIC_CV_PRECISION", raising=False)
    monkeypatch.setattr(doctor, "check_torch", lambda settings: CheckResult("PyTorch", "ok", "fake"))
    monkeypatch.setattr(doctor, "check_package", lambda name, **kwargs: CheckResult(name, "ok", "fake"))
    monkeypatch.setattr(doctor, "check_hf_access", lambda repo, **kwargs: CheckResult("HF access", "ok", "fake"))
    monkeypatch.setattr(doctor, "check_ollama", lambda *args, **kwargs: CheckResult("Ollama", "ok", "fake"))
    monkeypatch.setattr(doctor, "check_disk", lambda path: CheckResult("Disk", "ok", "fake"))
    monkeypatch.setattr(doctor, "check_port", lambda *args, **kwargs: CheckResult("Port", "ok", "fake"))

    checks = doctor.run_doctor(data_dir=tmp_path, network=False)

    assert [check.name for check in checks] == [
        "Python",
        "Model runtime",
        "PyTorch",
        "torchvision",
        "transformers",
        "accelerate",
        "HF access",
        "HF access",
        "Ollama",
        "Disk",
        "Port",
    ]
    assert doctor_exit_code(checks) == 0
