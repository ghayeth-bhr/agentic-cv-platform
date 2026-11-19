"""Bounded SAM3 threshold calibration over a representative sample set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

from PIL import Image

from .sam3 import Sam3Prediction, Sam3Provider

CalibrationStatus = Literal["calibrated", "max_iterations", "no_acceptable_threshold"]


class CalibrationCancelled(RuntimeError):
    """Raised when a calibration run is cancelled between inference steps."""


@dataclass(frozen=True, kw_only=True)
class CalibrationSample:
    image_id: str
    image: Image.Image


@dataclass(frozen=True, kw_only=True)
class CalibrationTrace:
    """One validator decision, including the evidence shown to the user."""

    accepted: bool
    confidence: float | None = None
    reasoning: str | None = None
    overlay_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class SampleVerdict:
    image_id: str
    accepted: bool
    confidence: float | None = None
    reasoning: str | None = None
    overlay_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class CalibrationAttempt:
    threshold: float
    sample_count: int
    accepted_count: int
    agreement: float
    verdicts: tuple[SampleVerdict, ...] = ()


@dataclass(frozen=True, kw_only=True)
class CalibrationResult:
    status: CalibrationStatus
    prompt: str
    threshold: float | None
    attempts: tuple[CalibrationAttempt, ...]
    reason: str | None = None


class PromptAwareValidator(Protocol):
    def validate_with_prompt(
        self,
        sample: CalibrationSample,
        predictions: list[Sam3Prediction],
        prompt: str,
    ) -> bool: ...


Validator = Callable[[CalibrationSample, list[Sam3Prediction]], bool] | PromptAwareValidator


class Sam3Calibrator:
    def __init__(
        self,
        provider: Sam3Provider,
        validator: Validator,
        *,
        minimum_agreement: float = 0.8,
        max_iterations: int = 5,
    ) -> None:
        if not 0 < minimum_agreement <= 1:
            raise ValueError("minimum_agreement must be between 0 and 1")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self.provider = provider
        self.validator = validator
        self.minimum_agreement = minimum_agreement
        self.max_iterations = max_iterations

    def calibrate(
        self,
        samples: list[CalibrationSample],
        *,
        prompt: str,
        thresholds: list[float],
        should_cancel: Callable[[], bool] | None = None,
    ) -> CalibrationResult:
        if not samples:
            raise ValueError("at least one calibration sample is required")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not thresholds:
            raise ValueError("at least one threshold is required")
        if any(not 0 <= threshold <= 1 for threshold in thresholds):
            raise ValueError("thresholds must be between 0 and 1")

        def check_cancel() -> None:
            if should_cancel is not None and should_cancel():
                raise CalibrationCancelled("Calibration was cancelled")

        attempts: list[CalibrationAttempt] = []
        for threshold in thresholds[: self.max_iterations]:
            check_cancel()
            accepted_count = 0
            verdicts: list[SampleVerdict] = []
            for sample in samples:
                check_cancel()
                predictions = self.provider.predict(sample.image, prompt, threshold=threshold)
                check_cancel()
                trace = self._validate(
                    sample=sample,
                    predictions=predictions,
                    prompt=prompt,
                    threshold=threshold,
                )
                check_cancel()
                accepted_count += int(trace.accepted)
                verdicts.append(
                    SampleVerdict(
                        image_id=sample.image_id,
                        accepted=trace.accepted,
                        confidence=trace.confidence,
                        reasoning=trace.reasoning,
                        overlay_id=trace.overlay_id,
                    )
                )
            agreement = accepted_count / len(samples)
            attempt = CalibrationAttempt(
                threshold=threshold,
                sample_count=len(samples),
                accepted_count=accepted_count,
                agreement=agreement,
                verdicts=tuple(verdicts),
            )
            attempts.append(attempt)
            if agreement >= self.minimum_agreement:
                return CalibrationResult(
                    status="calibrated",
                    prompt=prompt,
                    threshold=threshold,
                    attempts=tuple(attempts),
                )

        status: CalibrationStatus = (
            "max_iterations" if len(thresholds) > self.max_iterations else "no_acceptable_threshold"
        )
        return CalibrationResult(
            status=status,
            prompt=prompt,
            threshold=None,
            attempts=tuple(attempts),
            reason="no threshold met the minimum sample agreement",
        )

    def _validate(
        self,
        *,
        sample: CalibrationSample,
        predictions: list[Sam3Prediction],
        prompt: str,
        threshold: float,
    ) -> CalibrationTrace:
        traced = getattr(self.validator, "validate_with_trace", None)
        if traced is not None:
            trace = traced(
                sample,
                predictions,
                prompt,
                threshold=threshold,
            )
            if isinstance(trace, CalibrationTrace):
                return trace
            if isinstance(trace, bool):
                return CalibrationTrace(accepted=trace)
            if isinstance(trace, dict):
                return CalibrationTrace(
                    accepted=bool(trace.get("accepted")),
                    confidence=_optional_float(trace.get("confidence")),
                    reasoning=_optional_str(trace.get("reasoning")),
                    overlay_id=_optional_str(trace.get("overlay_id")),
                )
            raise TypeError("validate_with_trace must return CalibrationTrace or bool")

        prompt_aware_validator = getattr(self.validator, "validate_with_prompt", None)
        if prompt_aware_validator is None:
            accepted = self.validator(sample, predictions)
        else:
            accepted = prompt_aware_validator(sample, predictions, prompt)
        return CalibrationTrace(accepted=bool(accepted))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
