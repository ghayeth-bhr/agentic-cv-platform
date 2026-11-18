"""Resolve local model runtime settings (device and precision) from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .embeddings import HuggingFaceEmbeddingConfig
from .sam3 import Sam3Config

_VALID_PRECISIONS = frozenset({"float16", "bfloat16", "float32"})


def _cuda_available() -> bool:
    """Return whether torch can use CUDA, without requiring torch to be installed."""

    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(value: str | None = None) -> str:
    """Return a validated ``cpu`` / ``cuda`` / ``cuda:<index>`` device string."""

    raw = (value if value is not None else os.getenv("AGENTIC_CV_DEVICE", "")).strip().lower()
    if not raw:
        return "cuda" if _cuda_available() else "cpu"
    if raw == "cpu":
        return raw
    if raw == "cuda" or raw.startswith("cuda:"):
        index = raw.split(":", 1)[1] if ":" in raw else ""
        if index and not index.isdigit():
            raise ValueError(
                "AGENTIC_CV_DEVICE must be 'cpu', 'cuda', or 'cuda:<index>'"
            )
        return raw
    raise ValueError("AGENTIC_CV_DEVICE must be 'cpu', 'cuda', or 'cuda:<index>'")


def resolve_precision(device: str, value: str | None = None) -> str:
    """Return a precision compatible with the selected device."""

    raw = (value if value is not None else os.getenv("AGENTIC_CV_PRECISION", "")).strip().lower()
    if not raw:
        raw = "float32" if device == "cpu" else "float16"
    if raw not in _VALID_PRECISIONS:
        raise ValueError("AGENTIC_CV_PRECISION must be float16, bfloat16, or float32")
    if device == "cpu" and raw != "float32":
        raise ValueError("CPU inference requires AGENTIC_CV_PRECISION=float32")
    return raw


@dataclass(frozen=True, kw_only=True)
class ModelRuntimeSettings:
    """Device/precision shared by SigLIP embeddings and SAM 3."""

    device: str
    precision: str

    @classmethod
    def from_environment(cls) -> "ModelRuntimeSettings":
        device = resolve_device()
        return cls(device=device, precision=resolve_precision(device))

    def embedding_config(self) -> HuggingFaceEmbeddingConfig:
        return HuggingFaceEmbeddingConfig(device=self.device, precision=self.precision)

    def sam3_config(self) -> Sam3Config:
        return Sam3Config(device=self.device, precision=self.precision)
