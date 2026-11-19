"""SAM 3 threshold calibration over a representative sample set.

Design (Phase 2 of ``docs/OPTIMIZATION_PLAN.md``):

* **One forward pass per sample.** SAM 3's confidence threshold is applied in
  post-processing, so the predictions kept at threshold ``t`` are exactly the
  predictions from the lowest threshold whose score is ``>= t``. The model runs
  once per sample at the lowest threshold; every other threshold is a filter.
* **Every threshold is scored; the best one wins.** The first threshold that
  passed used to win, which favoured the most permissive one. Now the highest
  agreement wins and ties go to the middle of the best run, which is the most
  robust choice. Validators see each distinct prediction set once (the VLM
  validator caches by overlay digest), so scoring every threshold is cheap.
* **Correct negatives count.** A sample where SAM 3 returns nothing is correct
  when the target is absent. The validator reports ``target_present``; a pass
  also needs at least one sample where the target was found.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

from PIL import Image

from .sam3 import Sam3Prediction, Sam3Provider

CalibrationStatus = Literal[
    "calibrated",
    "max_iterations",
    "no_acceptable_threshold",
    "no_target_in_samples",
]


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
    # False when the validator saw no target in the image (a correct empty
    # result is then accepted); None when the validator cannot tell.
    target_present: bool | None = None


@dataclass(frozen=True, kw_only=True)
class SampleVerdict:
    image_id: str
    accepted: bool
    confidence: float | None = None
    reasoning: str | None = None
    overlay_id: str | None = None
    prediction_count: int = 0
    target_present: bool | None = None


@dataclass(frozen=True, kw_only=True)
class CalibrationAttempt:
    threshold: float
    sample_count: int
    accepted_count: int
    agreement: float
    verdicts: tuple[SampleVerdict, ...] = ()

    @property
    def positive_count(self) -> int:
        """Samples where SAM 3 found the target and the validator agreed."""

        return sum(1 for verdict in self.verdicts if verdict.accepted and verdict.prediction_count)


@dataclass(frozen=True, kw_only=True)
class CalibrationResult:
    status: CalibrationStatus
    prompt: str
    threshold: float | None
    attempts: tuple[CalibrationAttempt, ...]
    reason: str | None = None
    forward_passes: int = 0


class PromptAwareValidator(Protocol):
    def validate_with_prompt(
        self,
        sample: CalibrationSample,
        predictions: list[Sam3Prediction],
        prompt: str,
    ) -> bool: ...


Validator = Callable[[CalibrationSample, list[Sam3Prediction]], bool] | PromptAwareValidator
AttemptCallback = Callable[[CalibrationAttempt], None]


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
        on_attempt: AttemptCallback | None = None,
    ) -> CalibrationResult:
        if not samples:
            raise ValueError("at least one calibration sample is required")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not thresholds:
            raise ValueError("at least one threshold is required")
        if any(not 0 <= threshold <= 1 for threshold in thresholds):
            raise ValueError("thresholds must be between 0 and 1")

        raw = self.infer(samples, prompt=prompt, thresholds=thresholds, should_cancel=should_cancel)
        return self.score(
            samples,
            raw,
            prompt=prompt,
            thresholds=thresholds,
            should_cancel=should_cancel,
            on_attempt=on_attempt,
        )

    def _tried(self, thresholds: list[float]) -> list[float]:
        return sorted(dict.fromkeys(float(value) for value in thresholds))[: self.max_iterations]

    def infer(
        self,
        samples: list[CalibrationSample],
        *,
        prompt: str,
        thresholds: list[float],
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[list[Sam3Prediction]]:
        """SAM 3 phase: one forward pass per sample at the lowest threshold.

        Split from :meth:`score` so a caller can unload SAM 3 before the VLM
        runs: on a 6 GB GPU the two together page to system memory and a VLM
        verdict went from ~4 s to ~45 s (measured).
        """

        floor = self._tried(thresholds)[0]
        raw: list[list[Sam3Prediction]] = []
        for sample in samples:
            _check(should_cancel)
            raw.append(list(self.provider.predict(sample.image, prompt, threshold=floor)))
        _check(should_cancel)
        return raw

    def score(
        self,
        samples: list[CalibrationSample],
        raw: list[list[Sam3Prediction]],
        *,
        prompt: str,
        thresholds: list[float],
        should_cancel: Callable[[], bool] | None = None,
        on_attempt: AttemptCallback | None = None,
    ) -> CalibrationResult:
        """Validator phase: judge every threshold's filtered predictions."""

        if len(raw) != len(samples):
            raise ValueError("raw predictions must match the samples")
        attempts: list[CalibrationAttempt] = []
        for threshold in self._tried(thresholds):
            verdicts: list[SampleVerdict] = []
            for sample, predictions in zip(samples, raw):
                _check(should_cancel)
                kept = [item for item in predictions if item.score >= threshold]
                trace = self._validate(
                    sample=sample, predictions=kept, prompt=prompt, threshold=threshold
                )
                verdicts.append(
                    SampleVerdict(
                        image_id=sample.image_id,
                        accepted=trace.accepted,
                        confidence=trace.confidence,
                        reasoning=trace.reasoning,
                        overlay_id=trace.overlay_id,
                        prediction_count=len(kept),
                        target_present=trace.target_present,
                    )
                )
            accepted = sum(1 for verdict in verdicts if verdict.accepted)
            attempt = CalibrationAttempt(
                threshold=threshold,
                sample_count=len(samples),
                accepted_count=accepted,
                agreement=accepted / len(samples),
                verdicts=tuple(verdicts),
            )
            attempts.append(attempt)
            if on_attempt is not None:
                on_attempt(attempt)

        return self._decide(prompt, attempts, len(thresholds), forward_passes=len(samples))

    def _decide(
        self,
        prompt: str,
        attempts: list[CalibrationAttempt],
        requested: int,
        *,
        forward_passes: int,
    ) -> CalibrationResult:
        passing = [
            attempt for attempt in attempts
            if attempt.agreement >= self.minimum_agreement and attempt.positive_count > 0
        ]
        if passing:
            best = max(attempt.agreement for attempt in passing)
            top = [attempt for attempt in passing if attempt.agreement == best]
            chosen = top[(len(top) - 1) // 2]  # middle of the best run (lower middle on ties)
            return CalibrationResult(
                status="calibrated",
                prompt=prompt,
                threshold=chosen.threshold,
                attempts=tuple(attempts),
                forward_passes=forward_passes,
            )
        nobody_has_target = all(
            verdict.target_present is False
            for attempt in attempts
            for verdict in attempt.verdicts
        )
        if nobody_has_target:
            return CalibrationResult(
                status="no_target_in_samples",
                prompt=prompt,
                threshold=None,
                attempts=tuple(attempts),
                reason="none of the sample images contains the target",
                forward_passes=forward_passes,
            )
        status: CalibrationStatus = (
            "max_iterations" if requested > self.max_iterations else "no_acceptable_threshold"
        )
        return CalibrationResult(
            status=status,
            prompt=prompt,
            threshold=None,
            attempts=tuple(attempts),
            reason="no threshold met the minimum sample agreement",
            forward_passes=forward_passes,
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
                present = trace.get("target_present")
                return CalibrationTrace(
                    accepted=bool(trace.get("accepted")),
                    confidence=_optional_float(trace.get("confidence")),
                    reasoning=_optional_str(trace.get("reasoning")),
                    overlay_id=_optional_str(trace.get("overlay_id")),
                    target_present=None if present is None else bool(present),
                )
            raise TypeError("validate_with_trace must return CalibrationTrace or bool")

        prompt_aware_validator = getattr(self.validator, "validate_with_prompt", None)
        if prompt_aware_validator is None:
            accepted = self.validator(sample, predictions)
        else:
            accepted = prompt_aware_validator(sample, predictions, prompt)
        return CalibrationTrace(accepted=bool(accepted))


def _check(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise CalibrationCancelled("Calibration was cancelled")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
