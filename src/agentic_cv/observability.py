"""Runtime, GPU, and logging observability helpers.

Collects lightweight metrics without requiring optional heavy dependencies.
Torch/CUDA metrics are reported only when torch is importable and CUDA is
available; callers can inject a torch-like module for deterministic tests.

Logging helpers configure the ``agentic_cv`` logger namespace (opt-in via env
or explicit arguments) and provide a small JSON formatter for structured logs.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any


def _try_import_torch() -> Any | None:
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def collect_gpu_metrics(torch_module: Any | None = None) -> dict[str, Any]:
    """Return CUDA/GPU metrics when available, otherwise an unavailable report."""
    torch = torch_module if torch_module is not None else _try_import_torch()
    if torch is None:
        return {"available": False, "reason": "torch_not_installed"}
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        reason = "cuda_not_available"
        if cuda is None:
            reason = "cuda_module_missing"
        return {"available": False, "reason": reason}

    device_count = int(cuda.device_count())
    devices = []
    for index in range(device_count):
        try:
            props = cuda.get_device_properties(index)
            total = int(getattr(props, "total_memory", 0))
            name = str(getattr(props, "name", f"cuda:{index}"))
        except Exception:
            total = 0
            name = f"cuda:{index}"
        try:
            allocated = int(cuda.memory_allocated(index))
            reserved = int(cuda.memory_reserved(index))
        except Exception:
            allocated = 0
            reserved = 0
        free = max(total - reserved, 0) if total else None
        devices.append(
            {
                "index": index,
                "name": name,
                "memory_bytes": {
                    "total": total or None,
                    "allocated": allocated,
                    "reserved": reserved,
                    "free": free,
                },
            }
        )

    current_device = None
    try:
        current_device = int(cuda.current_device())
    except Exception:
        current_device = None

    return {
        "available": True,
        "device_count": device_count,
        "current_device": current_device,
        "devices": devices,
    }


def collect_runtime_metrics() -> dict[str, Any]:
    """Return basic process/runtime metrics from the standard library."""
    metrics: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "pid": os.getpid(),
        "time_unix": time.time(),
    }
    try:
        import resource  # Unix-only; unavailable on Windows

        usage = resource.getrusage(resource.RUSAGE_SELF)
        metrics["resource"] = {
            "max_rss_kb": int(getattr(usage, "ru_maxrss", 0)),
            "user_time_s": float(getattr(usage, "ru_utime", 0.0)),
            "system_time_s": float(getattr(usage, "ru_stime", 0.0)),
        }
    except Exception:
        metrics["resource"] = None
    return metrics


def collect_metrics(*, torch_module: Any | None = None) -> dict[str, Any]:
    """Collect runtime and GPU metrics for observability endpoints."""
    return {
        "status": "ok",
        "runtime": collect_runtime_metrics(),
        "gpu": collect_gpu_metrics(torch_module),
    }


class JsonLogFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key in payload or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = str(value)
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _resolve_level(level: str | int | None) -> int:
    if level is None:
        env = os.getenv("AGENTIC_CV_LOG_LEVEL")
        if env:
            level = env
        else:
            return logging.INFO
    if isinstance(level, int):
        return level
    name = str(level).upper()
    return int(getattr(logging, name, logging.INFO))


def configure_logging(level: str | int | None = None, fmt: str | None = None) -> logging.Logger:
    """Configure the ``agentic_cv`` logger namespace.

    Reads ``AGENTIC_CV_LOG_LEVEL`` and ``AGENTIC_CV_LOG_FORMAT`` (``json`` or
    ``text``) when explicit arguments are not provided. Configuration is scoped
    to the ``agentic_cv`` logger to avoid overriding host/uvicorn logging.
    """

    logger = logging.getLogger("agentic_cv")
    resolved_level = _resolve_level(level)
    resolved_fmt = (fmt or os.getenv("AGENTIC_CV_LOG_FORMAT") or "text").lower()

    # Avoid stacking duplicate handlers on repeated calls (e.g. tests/reloads).
    if not any(getattr(handler, "_agentic_cv_logging", False) for handler in logger.handlers):
        handler = logging.StreamHandler(stream=sys.stderr)
        handler._agentic_cv_logging = True  # type: ignore[attr-defined]
        if resolved_fmt == "json":
            handler.setFormatter(JsonLogFormatter())
        else:
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            if getattr(handler, "_agentic_cv_logging", False):
                if resolved_fmt == "json":
                    handler.setFormatter(JsonLogFormatter())
                else:
                    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))

    logger.setLevel(resolved_level)
    logger.propagate = False
    return logger
