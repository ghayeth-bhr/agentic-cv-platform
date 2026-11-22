"""Choose the calibration validator and render durable VLM overlay artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from uuid import UUID

import numpy as np

from .calibration import CalibrationSample, Validator
from .ollama_vlm import OllamaVlmClient
from .overlay import render_instance_crop, render_prediction_overlay
from .sam3 import Sam3Prediction
from .vlm_validator import OverlayArtifact, VlmCalibrationValidator, VlmValidator

ValidationMode = Literal["vlm", "any_prediction", "custom"]


@dataclass(frozen=True, kw_only=True)
class ValidatorSelection:
    validator: Validator
    mode: ValidationMode
    model: str | None = None
    notice: str | None = None
    notice_provider: Callable[[], str | None] | None = None
    release: Callable[[], bool] | None = None

    def current_notice(self) -> str | None:
        if self.notice_provider is not None:
            return self.notice_provider()
        return self.notice


def at_least_one_prediction(
    sample: CalibrationSample, predictions: list[Sam3Prediction]
) -> bool:
    """The explicitly labeled fallback used when a VLM cannot be reached."""

    return len(predictions) > 0


def any_prediction_selection(*, notice: str | None = None) -> ValidatorSelection:
    return ValidatorSelection(
        validator=at_least_one_prediction,
        mode="any_prediction",
        notice=notice,
    )


def probe_ollama(
    *,
    base_url: str,
    model: str,
    timeout_sec: float = 1.0,
) -> str | None:
    """Return None when the VLM is usable, otherwise a human-readable reason."""

    endpoint = base_url.rstrip("/")
    try:
        with urlopen(f"{endpoint}/api/tags", timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return f"Ollama is not reachable at {endpoint}: {exc}"
    models = {
        str(item.get("name", ""))
        for item in payload.get("models", [])
        if isinstance(item, dict)
    }
    if model not in models and f"{model}:latest" not in models:
        found = ", ".join(sorted(models)) if models else "none"
        return f"Ollama is reachable at {endpoint} but model {model} is missing (found: {found})"
    return None


def default_validator_factory(data_dir: Path):
    """Build the production selector; the factory is lazy so app startup is silent."""

    overlay_root = data_dir / "calibration-overlays"

    def factory(tenant_id: UUID, session_id: UUID) -> ValidatorSelection:
        return default_validator_selection(
            overlay_root=overlay_root,
            tenant_id=tenant_id,
            session_id=session_id,
        )

    return factory


def default_validator_selection(
    *,
    overlay_root: Path,
    tenant_id: UUID,
    session_id: UUID,
) -> ValidatorSelection:
    disabled = os.getenv("AGENTIC_CV_AGENT_VALIDATOR", "").strip().lower()
    if disabled in {"any-prediction", "any_prediction", "rules"}:
        return any_prediction_selection(
            notice=(
                "Vision validation is disabled by AGENTIC_CV_AGENT_VALIDATOR; "
                "calibration used the fallback \"at least one prediction\" check."
            )
        )

    base_url = os.getenv("AGENTIC_CV_OLLAMA_URL", "http://localhost:11434")
    model = (
        os.getenv("AGENTIC_CV_VLM_MODEL")
        or os.getenv("AGENTIC_CV_TEXT_MODEL")
        or "qwen3-vl:4b"
    )
    try:
        probe_timeout = max(
            0.05, float(os.getenv("AGENTIC_CV_VLM_PROBE_TIMEOUT_SEC", "1.0"))
        )
    except ValueError:
        probe_timeout = 1.0
    reason = probe_ollama(base_url=base_url, model=model, timeout_sec=probe_timeout)
    if reason is not None:
        return any_prediction_selection(
            notice=(
                f"Vision validation is unavailable: {reason}. Calibration used the "
                "fallback \"at least one prediction\" check."
            )
        )

    try:
        timeout_sec = max(
            1.0, float(os.getenv("AGENTIC_CV_VLM_TIMEOUT_SEC", "100"))
        )
    except ValueError:
        timeout_sec = 100.0
    try:
        max_calls = max(1, int(os.getenv("AGENTIC_CV_VLM_MAX_CALLS", "50")))
    except ValueError:
        max_calls = 50
    try:
        minimum_confidence = min(
            1.0, max(0.0, float(os.getenv("AGENTIC_CV_VLM_MIN_CONFIDENCE", "0.7")))
        )
    except ValueError:
        minimum_confidence = 0.7
    try:
        minimum_prediction_score = min(
            1.0,
            max(0.0, float(os.getenv("AGENTIC_CV_VLM_MIN_PREDICTION_SCORE", "0.2"))),
        )
    except ValueError:
        minimum_prediction_score = 0.2

    overlay_builder = build_overlay_builder(
        overlay_root=overlay_root,
        tenant_id=tenant_id,
        session_id=session_id,
    )
    # "instance" (default) judges each mask on its own crop; "scene" judges
    # one overlay with every mask drawn, which a small VLM accepts too easily
    # when one of several masks covers the wrong object (see benchmarks/).
    judge = os.getenv("AGENTIC_CV_VLM_JUDGE", "instance").strip().lower()
    instance_builder = (
        build_instance_builder(overlay_root=overlay_root, tenant_id=tenant_id, session_id=session_id)
        if judge != "scene"
        else None
    )
    validator = VlmCalibrationValidator(
        VlmValidator(
            OllamaVlmClient(
                model=model,
                base_url=base_url,
                timeout_sec=timeout_sec,
                max_side=_env_int("AGENTIC_CV_VLM_MAX_SIDE", 768),
            ),
            max_calls=max_calls,
            minimum_confidence=minimum_confidence,
        ),
        overlay_builder,
        model_revision=model,
        fallback=at_least_one_prediction,
        minimum_prediction_score=minimum_prediction_score,
        instance_builder=instance_builder,
    )
    return ValidatorSelection(
        validator=validator,
        mode="vlm",
        model=model,
        notice_provider=lambda: validator.failure_notice,
        release=validator.release,
    )


def build_overlay_builder(
    *,
    overlay_root: Path,
    tenant_id: UUID,
    session_id: UUID,
):
    directory = overlay_root / str(tenant_id) / str(session_id)

    def build(
        sample: CalibrationSample, predictions: list[Sam3Prediction]
    ) -> OverlayArtifact:
        digest = _prediction_digest(sample.image_id, predictions)
        overlay_id = f"{digest}.png"
        path = directory / overlay_id
        render_prediction_overlay(sample.image, predictions, path)
        return OverlayArtifact(uri=str(path), overlay_id=overlay_id)

    return build


def build_instance_builder(
    *,
    overlay_root: Path,
    tenant_id: UUID,
    session_id: UUID,
):
    directory = overlay_root / str(tenant_id) / str(session_id) / "instances"

    def build(sample: CalibrationSample, prediction: Sam3Prediction) -> OverlayArtifact:
        digest = _prediction_digest(sample.image_id, [prediction])
        overlay_id = f"{digest}.png"
        path = directory / overlay_id
        if not path.exists():
            render_instance_crop(sample.image, prediction, path)
        return OverlayArtifact(uri=str(path), overlay_id=overlay_id)

    return build


def _prediction_digest(image_id: str, predictions: list[Sam3Prediction]) -> str:
    digest = hashlib.sha256(image_id.encode("utf-8"))
    for prediction in predictions:
        digest.update(f"{prediction.score:.6f}|{prediction.box}".encode("ascii"))
        if prediction.mask is not None:
            digest.update(np.asarray(prediction.mask, dtype=np.uint8).tobytes())
    return digest.hexdigest()[:24]


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default
