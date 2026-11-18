"""Bounded SAM3 threshold calibration over a representative sample set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from PIL import Image

from .sam3 import Sam3Prediction, Sam3Provider

CalibrationStatus = Literal["calibrated", "max_iterations", "no_acceptable_threshold"]


@dataclass(frozen=True, kw_only=True)
class CalibrationSample:
    image_id: str
    image: Image.Image


@dataclass(frozen=True, kw_only=True)
class CalibrationAttempt:
    threshold: float
    sample_count: int
    accepted_count: int
    agreement: float


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
    ) -> CalibrationResult:
        if not samples:
            raise ValueError("at least one calibration sample is required")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not thresholds:
            raise ValueError("at least one threshold is required")
        if any(not 0 <= threshold <= 1 for threshold in thresholds):
            raise ValueError("thresholds must be between 0 and 1")

        attempts: list[CalibrationAttempt] = []
        for threshold in thresholds[: self.max_iterations]:
            accepted_count = 0
            for sample in samples:
                predictions = self.provider.predict(sample.image, prompt, threshold=threshold)
                prompt_aware_validator = getattr(self.validator, "validate_with_prompt", None)
                if prompt_aware_validator is None:
                    accepted = self.validator(sample, predictions)
                else:
                    accepted = prompt_aware_validator(sample, predictions, prompt)
                accepted_count += accepted
            agreement = accepted_count / len(samples)
            attempt = CalibrationAttempt(
                threshold=threshold,
                sample_count=len(samples),
                accepted_count=accepted_count,
                agreement=agreement,
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
