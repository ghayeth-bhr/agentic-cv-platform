"""Environment diagnostics for the local MVP.

The doctor is deliberately dependency-light: it probes optional packages with
``importlib`` and never imports torch/transformers at module import time. Every
check returns an actionable fix instead of a traceback.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import socket
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from .model_runtime import ModelRuntimeSettings

Status = Literal["ok", "warn", "fail"]

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3-vl:4b"
DEFAULT_SAM3_MODEL = "facebook/sam3"
DEFAULT_EMBEDDING_MODEL = "google/siglip2-so400m-patch14-384"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str
    fix: str | None = None


def check_python() -> CheckResult:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info < (3, 11):
        return CheckResult(
            "Python",
            "fail",
            f"Python {version} is too old (requires >=3.11)",
            "Install Python 3.11+ and recreate the virtual environment.",
        )
    return CheckResult("Python", "ok", f"Python {version}")


def check_torch(settings: ModelRuntimeSettings) -> CheckResult:
    if importlib.util.find_spec("torch") is None:
        return CheckResult(
            "PyTorch",
            "fail",
            "torch is not installed",
            "Install the ML extra with `uv sync --extra ml`. On Windows/CUDA, install "
            "torch and torchvision from https://download.pytorch.org/whl/cu124.",
        )
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment-specific
        return CheckResult(
            "PyTorch",
            "fail",
            f"torch failed to import: {exc}",
            "Reinstall the ML extra with `uv sync --extra ml`.",
        )

    version = getattr(torch, "__version__", "unknown")
    if settings.device.startswith("cuda"):
        try:
            available = bool(torch.cuda.is_available())
        except Exception as exc:  # pragma: no cover - environment-specific
            return CheckResult("PyTorch", "fail", f"CUDA check failed: {exc}", None)
        if not available:
            return CheckResult(
                "PyTorch",
                "fail",
                f"torch {version}; CUDA was requested but is not available",
                "Install a CUDA-enabled torch build, or set AGENTIC_CV_DEVICE=cpu "
                "(and AGENTIC_CV_PRECISION=float32) for CPU inference.",
            )
        try:
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            detail = f"torch {version}; CUDA: {name} ({total:.1f} GiB)"
        except Exception:  # pragma: no cover - driver-specific
            detail = f"torch {version}; CUDA available"
        return CheckResult("PyTorch", "ok", detail)
    return CheckResult("PyTorch", "ok", f"torch {version}; CPU mode (AGENTIC_CV_DEVICE=cpu)")


def check_package(name: str, *, min_major: int | None = None) -> CheckResult:
    if importlib.util.find_spec(name) is None:
        return CheckResult(
            name,
            "fail",
            f"{name} is not installed",
            "Install the ML extra with `uv sync --extra ml`.",
        )
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - environment-specific
        return CheckResult(
            name,
            "fail",
            f"{name} failed to import: {exc}",
            "Reinstall the ML extra with `uv sync --extra ml`.",
        )
    version = str(getattr(module, "__version__", "unknown"))
    if min_major is not None:
        try:
            major = int(version.split(".", 1)[0])
        except ValueError:
            major = None
        if major is not None and major < min_major:
            return CheckResult(
                name,
                "warn",
                f"{name} {version} is older than the validated {min_major}.x line",
                f"Upgrade with `uv sync --extra ml` (requires {name}>={min_major}).",
            )
    return CheckResult(name, "ok", f"{name} {version}")


def _cached_model(repo_id: str) -> str | None:
    try:
        from huggingface_hub import scan_cache_dir
    except Exception:
        return None
    try:
        cache = scan_cache_dir()
    except Exception:
        return None
    for repo in getattr(cache, "repos", ()):
        if getattr(repo, "repo_id", None) == repo_id:
            return str(getattr(repo, "repo_path", "cached"))
    return None


def check_hf_access(repo_id: str, *, network: bool = True) -> CheckResult:
    cached = _cached_model(repo_id)
    if cached is not None:
        return CheckResult("HF access", "ok", f"{repo_id} cached at {cached}")

    if importlib.util.find_spec("huggingface_hub") is None:
        return CheckResult(
            "HF access",
            "warn",
            f"{repo_id} is not cached and huggingface_hub is not installed",
            "Install the ML extra with `uv sync --extra ml`, then rerun the doctor.",
        )
    if not network:
        return CheckResult(
            "HF access",
            "warn",
            f"{repo_id} is not in the local cache (network check skipped)",
            "Rerun without --no-network, or start the app once to download the model.",
        )

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except Exception as exc:  # pragma: no cover - import-shape specific
        return CheckResult("HF access", "warn", f"could not import huggingface_hub: {exc}", None)

    api = HfApi()
    try:
        try:
            api.model_info(repo_id, timeout=5)
        except TypeError:
            return CheckResult(
                "HF access",
                "warn",
                f"could not verify {repo_id}: installed huggingface_hub has no timeout argument",
                "Upgrade huggingface_hub with `uv sync --extra ml`.",
            )
    except GatedRepoError:
        return CheckResult(
            "HF access",
            "fail",
            f"{repo_id} is gated and this token does not have access",
            f"Request access at https://huggingface.co/{repo_id}, set HF_TOKEN, then retry.",
        )
    except RepositoryNotFoundError:
        return CheckResult(
            "HF access",
            "fail",
            f"{repo_id} was not found",
            "Check the model id and your Hugging Face token.",
        )
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            return CheckResult(
                "HF access",
                "fail",
                f"{repo_id} access denied (HTTP {status})",
                f"Request access at https://huggingface.co/{repo_id} and set HF_TOKEN.",
            )
        return CheckResult(
            "HF access",
            "warn",
            f"could not verify {repo_id}: {exc}",
            "Check network connectivity and HF_TOKEN.",
        )
    except Exception as exc:
        return CheckResult(
            "HF access",
            "warn",
            f"could not verify {repo_id}: {exc}",
            "Check network connectivity and HF_TOKEN.",
        )
    return CheckResult("HF access", "ok", f"{repo_id} is accessible")


def check_ollama(url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_OLLAMA_MODEL) -> CheckResult:
    endpoint = url.rstrip("/")
    try:
        with urlopen(f"{endpoint}/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return CheckResult(
            "Ollama",
            "fail",
            f"Ollama is not reachable at {endpoint}: {exc}",
            "Start Ollama (if it is already running, do not start a second `ollama serve`) "
            "and rerun the doctor.",
        )
    models = [
        item.get("name", "")
        for item in payload.get("models", [])
        if isinstance(item, dict) and item.get("name")
    ]
    if model in models or f"{model}:latest" in models:
        return CheckResult("Ollama", "ok", f"{endpoint}; model {model} is installed")
    found = ", ".join(models) if models else "none"
    return CheckResult(
        "Ollama",
        "fail",
        f"{endpoint} is reachable but model {model} is missing (found: {found})",
        f"Run `ollama pull {model}`.",
    )


def check_disk(data_dir: Path) -> CheckResult:
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".doctor-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return CheckResult(
            "Data directory",
            "fail",
            f"{data_dir} is not writable: {exc}",
            "Choose a writable --data-dir path.",
        )
    try:
        usage = shutil.disk_usage(data_dir)
    except OSError as exc:  # pragma: no cover - platform-specific
        return CheckResult("Disk", "warn", f"could not read free space: {exc}", None)
    free_gib = usage.free / (1024**3)
    if free_gib < 2:
        return CheckResult(
            "Disk",
            "fail",
            f"{free_gib:.1f} GiB free under {data_dir}",
            "Free at least 2 GiB; model weights and exports need room.",
        )
    if free_gib < 10:
        return CheckResult(
            "Disk",
            "warn",
            f"{free_gib:.1f} GiB free under {data_dir}",
            "Model weights and video frames can fill this quickly; aim for 10 GiB+.",
        )
    return CheckResult("Disk", "ok", f"{free_gib:.1f} GiB free under {data_dir}")


def check_port(host: str = "127.0.0.1", port: int = 8000) -> CheckResult:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            pass
    except OSError:
        return CheckResult("Port", "ok", f"{host}:{port} is free")
    return CheckResult(
        "Port",
        "warn",
        f"{host}:{port} is already accepting connections",
        "If this is the app, it is already running; otherwise stop the process using the port.",
    )


def run_doctor(
    *,
    data_dir: Path,
    network: bool = True,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> list[CheckResult]:
    checks: list[CheckResult] = [check_python()]
    try:
        settings = ModelRuntimeSettings.from_environment()
    except ValueError as exc:
        checks.append(
            CheckResult(
                "Model runtime",
                "fail",
                str(exc),
                "Fix AGENTIC_CV_DEVICE/AGENTIC_CV_PRECISION and rerun the doctor.",
            )
        )
        settings = ModelRuntimeSettings(device="cuda", precision="float16")
    else:
        checks.append(
            CheckResult(
                "Model runtime",
                "ok",
                f"device={settings.device}, precision={settings.precision}",
            )
        )
    checks.append(check_torch(settings))
    checks.append(check_package("torchvision"))
    checks.append(check_package("transformers", min_major=5))
    checks.append(check_package("accelerate"))
    checks.append(check_hf_access(DEFAULT_SAM3_MODEL, network=network))
    checks.append(check_hf_access(DEFAULT_EMBEDDING_MODEL, network=network))
    checks.append(check_ollama(ollama_url, ollama_model))
    checks.append(check_disk(data_dir))
    checks.append(check_port(host, port))
    return checks


def format_report(checks: list[CheckResult]) -> str:
    lines = ["Agentic CV doctor", "================="]
    for check in checks:
        lines.append(f"[{check.status}] {check.name}: {check.detail}")
        if check.fix:
            lines.append(f"       fix: {check.fix}")
    counts = {status: sum(1 for check in checks if check.status == status) for status in ("ok", "warn", "fail")}
    lines.append("")
    lines.append(
        f"Summary: {counts['ok']} ok, {counts['warn']} warn, {counts['fail']} fail"
    )
    return "\n".join(lines)


def doctor_exit_code(checks: list[CheckResult]) -> int:
    return 1 if any(check.status == "fail" for check in checks) else 0


def report_as_json(checks: list[CheckResult]) -> str:
    return json.dumps([asdict(check) for check in checks], indent=2)
