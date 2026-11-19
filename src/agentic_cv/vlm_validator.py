"""Structured, budgeted, cache-aware VLM validation for calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .calibration import CalibrationSample, CalibrationTrace
from .sam3 import Sam3Prediction


class VlmClient(Protocol):
    def judge(self, *, overlay_uri: str, prompt: str) -> dict[str, Any]: ...


class VlmBudgetExceeded(RuntimeError):
    """Raised when a validation run reaches its configured VLM call budget."""


@dataclass(frozen=True, kw_only=True)
class OverlayArtifact:
    """A rendered overlay plus the opaque id used by the local artifact route."""

    uri: str
    overlay_id: str


@dataclass(frozen=True, kw_only=True)
class ValidationJudgment:
    target_present: bool
    fully_detected: bool
    false_positive: bool
    mask_tight: bool
    confidence: float
    reasoning: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ValidationJudgment":
        required = {
            "target_present", "fully_detected", "false_positive",
            "mask_tight", "confidence", "reasoning",
        }
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"VLM response missing fields: {sorted(missing)}")
        confidence = payload["confidence"]
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("VLM confidence must be between 0 and 1")
        if not isinstance(payload["reasoning"], str) or not payload["reasoning"].strip():
            raise ValueError("VLM reasoning must not be empty")
        if any(not isinstance(payload[field], bool) for field in required - {"confidence", "reasoning"}):
            raise ValueError("VLM boolean fields must be booleans")
        return cls(
            target_present=payload["target_present"],
            fully_detected=payload["fully_detected"],
            false_positive=payload["false_positive"],
            mask_tight=payload["mask_tight"],
            confidence=float(confidence),
            reasoning=payload["reasoning"],
        )

    def accepted(self, *, minimum_confidence: float = 0.7) -> bool:
        return (
            self.target_present
            and self.fully_detected
            and not self.false_positive
            and self.mask_tight
            and self.confidence >= minimum_confidence
        )


class VlmValidator:
    def __init__(
        self,
        client: VlmClient,
        *,
        max_calls: int = 20,
        minimum_confidence: float = 0.7,
    ) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1")
        self.client = client
        self.max_calls = max_calls
        self.minimum_confidence = minimum_confidence
        self._calls = 0
        self._cache: dict[tuple[str, str, str], ValidationJudgment] = {}

    @property
    def calls_made(self) -> int:
        return self._calls

    def validate(
        self,
        *,
        overlay_uri: str,
        prompt: str,
        model_revision: str = "default",
    ) -> ValidationJudgment:
        if not overlay_uri.strip() or not prompt.strip():
            raise ValueError("overlay_uri and prompt must not be empty")
        key = (overlay_uri, prompt, model_revision)
        if key in self._cache:
            return self._cache[key]
        if self._calls >= self.max_calls:
            raise VlmBudgetExceeded("VLM validation call budget exceeded")
        judgment = ValidationJudgment.from_payload(
            self.client.judge(overlay_uri=overlay_uri, prompt=prompt)
        )
        self._calls += 1
        self._cache[key] = judgment
        return judgment

    def accepts(self, **kwargs: Any) -> bool:
        return self.validate(**kwargs).accepted(minimum_confidence=self.minimum_confidence)


class VlmCalibrationValidator:
    """Adapt VLM judgments to the callable contract used by calibration."""

    def __init__(
        self,
        validator: VlmValidator,
        overlay_builder: Callable[[CalibrationSample, list[Sam3Prediction]], str | OverlayArtifact],
        *,
        model_revision: str = "default",
        fallback: Validator | None = None,
    ) -> None:
        self.validator = validator
        self.overlay_builder = overlay_builder
        self.model_revision = model_revision
        self.fallback = fallback
        self._failure_notice: str | None = None

    @property
    def failure_notice(self) -> str | None:
        return self._failure_notice

    def validate_with_trace(
        self,
        sample: CalibrationSample,
        predictions: list[Sam3Prediction],
        prompt: str,
        *,
        threshold: float | None = None,
    ) -> CalibrationTrace:
        if not predictions:
            return CalibrationTrace(
                accepted=False,
                confidence=0.0,
                reasoning="SAM 3 returned no masks to validate at this threshold",
            )
        if self._failure_notice is not None:
            return self._fallback_trace(sample, predictions)
        artifact = self._overlay_artifact(sample, predictions)
        try:
            judgment = self.validator.validate(
                overlay_uri=artifact.uri,
                prompt=_validation_instruction(prompt),
                model_revision=self.model_revision,
            )
        except Exception as exc:
            if self.fallback is None:
                raise
            self._failure_notice = (
                "Vision validation failed during calibration "
                f"({type(exc).__name__}: {exc}). Calibration continued with the "
                "fallback \"at least one prediction\" check."
            )
            return self._fallback_trace(sample, predictions)
        return CalibrationTrace(
            accepted=judgment.accepted(minimum_confidence=self.validator.minimum_confidence),
            confidence=judgment.confidence,
            reasoning=judgment.reasoning,
            overlay_id=artifact.overlay_id,
        )

    def _fallback_trace(
        self,
        sample: CalibrationSample,
        predictions: list[Sam3Prediction],
    ) -> CalibrationTrace:
        if self.fallback is None:
            raise RuntimeError("no calibration fallback is configured")
        return CalibrationTrace(
            accepted=bool(self.fallback(sample, predictions)),
            reasoning=self._failure_notice,
        )

    def validate_with_prompt(
        self,
        sample: CalibrationSample,
        predictions: list[Sam3Prediction],
        prompt: str,
    ) -> bool:
        return self.validate_with_trace(sample, predictions, prompt).accepted

    def __call__(
        self,
        sample: CalibrationSample,
        predictions: list[Sam3Prediction],
    ) -> bool:
        return self.validate_with_prompt(sample, predictions, sample.image_id)

    def _overlay_artifact(
        self,
        sample: CalibrationSample,
        predictions: list[Sam3Prediction],
    ) -> OverlayArtifact:
        built = self.overlay_builder(sample, predictions)
        if isinstance(built, OverlayArtifact):
            return built
        if not isinstance(built, str):
            raise TypeError("overlay_builder must return str or OverlayArtifact")
        return OverlayArtifact(uri=built, overlay_id=_fallback_overlay_id(built))


def _fallback_overlay_id(uri: str) -> str:
    from hashlib import sha256
    from pathlib import Path

    name = Path(uri).name
    return name or sha256(uri.encode("utf-8")).hexdigest()[:24]


def _validation_instruction(prompt: str) -> str:
    target = prompt.strip()
    return (
        f"Target object: {target}. The image is a photo with colored segmentation "
        "masks and score boxes drawn on it. Judge the highlighted masks. "
        f"target_present: at least one {target} appears in the photo. "
        f"fully_detected: every visible {target} is covered by a mask. "
        f"false_positive: a mask covers background or something that is not a {target}. "
        "mask_tight: the mask follows the object boundary closely. "
        "Return confidence from 0 to 1 and one short reasoning sentence."
    )
