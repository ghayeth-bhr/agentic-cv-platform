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
        self._released = False
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
        self._released = False  # loaded again; a later release() must unload it
        self._cache[key] = judgment
        return judgment

    def accepts(self, **kwargs: Any) -> bool:
        return self.validate(**kwargs).accepted(minimum_confidence=self.minimum_confidence)

    def release(self) -> bool:
        """Unload the VLM after calibration; safe to call more than once."""

        if self._released or self.calls_made == 0:
            return False
        unload = getattr(self.client, "unload", None)
        if not callable(unload):
            return False
        released = bool(unload())
        if released:
            self._released = True
        return released


class VlmCalibrationValidator:
    """Adapt VLM judgments to the callable contract used by calibration."""

    def __init__(
        self,
        validator: VlmValidator,
        overlay_builder: Callable[[CalibrationSample, list[Sam3Prediction]], str | OverlayArtifact],
        *,
        model_revision: str = "default",
        fallback: Validator | None = None,
        minimum_prediction_score: float = 0.0,
        instance_builder: Callable[[CalibrationSample, Sam3Prediction], OverlayArtifact] | None = None,
        max_instances_per_sample: int = 6,
    ) -> None:
        if not 0 <= minimum_prediction_score <= 1:
            raise ValueError("minimum_prediction_score must be between 0 and 1")
        if max_instances_per_sample < 1:
            raise ValueError("max_instances_per_sample must be at least 1")
        self.instance_builder = instance_builder
        self.max_instances_per_sample = max_instances_per_sample
        # Per sample: every mask seen at the lowest threshold, highest score
        # first. Calibration scores thresholds in ascending order, so the first
        # call per sample carries the full set; later calls are subsets.
        self._instances: dict[tuple[str, str], list[Sam3Prediction]] = {}
        self.validator = validator
        self.overlay_builder = overlay_builder
        self.model_revision = model_revision
        self.fallback = fallback
        self.minimum_prediction_score = minimum_prediction_score
        self._failure_notice: str | None = None

    @property
    def failure_notice(self) -> str | None:
        return self._failure_notice

    def release(self) -> bool:
        return self.validator.release()

    def validate_with_trace(
        self,
        sample: CalibrationSample,
        predictions: list[Sam3Prediction],
        prompt: str,
        *,
        threshold: float | None = None,
    ) -> CalibrationTrace:
        if not predictions:
            return self._presence_trace(sample, prompt)
        low_score_count = sum(
            1 for prediction in predictions
            if prediction.score < self.minimum_prediction_score
        )
        if low_score_count:
            return CalibrationTrace(
                accepted=False,
                confidence=0.0,
                reasoning=(
                    f"Rejected {low_score_count} low-score mask(s) below "
                    f"{self.minimum_prediction_score:.2f}; this threshold is too permissive."
                ),
            )
        if self._failure_notice is not None:
            return self._fallback_trace(sample, predictions)
        if self.instance_builder is not None:
            return self._instance_trace(sample, predictions, prompt)
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
            target_present=judgment.target_present,
        )

    def _instance_trace(
        self,
        sample: CalibrationSample,
        predictions: list[Sam3Prediction],
        prompt: str,
    ) -> CalibrationTrace:
        """Judge each mask on its own, then score this threshold.

        A threshold is too low when it keeps a mask the VLM rejected (a false
        positive) and too high when it drops a mask the VLM confirmed (a missed
        target). Each mask is judged once; the cache answers every other
        threshold, so the VLM cost does not grow with the threshold grid.
        """

        target = prompt.strip()
        known = self._instances.setdefault((sample.image_id, target), [])
        for prediction in predictions:
            if not any(prediction is item for item in known):
                known.append(prediction)
        known.sort(key=lambda item: -item.score)
        judged = known[: self.max_instances_per_sample]
        verdicts: list[tuple[Sam3Prediction, bool | None]] = []
        try:
            for prediction in judged:
                artifact = self.instance_builder(sample, prediction)
                judgment = self.validator.validate(
                    overlay_uri=artifact.uri,
                    prompt=_instance_instruction(target),
                    model_revision=self.model_revision,
                )
                if judgment.confidence < 0.5:
                    verdicts.append((prediction, None))  # unsure: counts neither way
                else:
                    verdicts.append((prediction, judgment.target_present and not judgment.false_positive))
        except Exception as exc:
            if self.fallback is None:
                raise
            self._failure_notice = (
                "Vision validation failed during calibration "
                f"({type(exc).__name__}: {exc}). Calibration continued with the "
                "fallback \"at least one prediction\" check."
            )
            return self._fallback_trace(sample, predictions)

        kept_ids = {id(item) for item in predictions}
        wrong_kept = [item for item, verdict in verdicts if verdict is False and id(item) in kept_ids]
        right_dropped = [item for item, verdict in verdicts if verdict is True and id(item) not in kept_ids]
        right_kept = [item for item, verdict in verdicts if verdict is True and id(item) in kept_ids]
        problems = []
        if wrong_kept:
            problems.append(
                f"keeps {len(wrong_kept)} mask(s) that are not a {target} "
                f"(scores {', '.join(f'{item.score:.2f}' for item in wrong_kept)})"
            )
        if right_dropped:
            problems.append(
                f"drops {len(right_dropped)} {target} mask(s) "
                f"(scores {', '.join(f'{item.score:.2f}' for item in right_dropped)})"
            )
        accepted = not problems and bool(right_kept)
        if accepted:
            reasoning = f"All {len(right_kept)} kept mask(s) are a {target}, and none were dropped."
        elif problems:
            reasoning = "This threshold " + " and ".join(problems) + "."
        else:
            reasoning = f"The vision model could not confirm any kept mask as a {target}."
        artifact = self._overlay_artifact(sample, predictions)
        return CalibrationTrace(
            accepted=accepted,
            confidence=1.0 if accepted else 0.0,
            reasoning=reasoning,
            overlay_id=artifact.overlay_id,
            target_present=True if right_kept or right_dropped else None,
        )

    def _presence_trace(self, sample: CalibrationSample, prompt: str) -> CalibrationTrace:
        """SAM 3 found nothing: correct if, and only if, the target is absent.

        The plain image is judged once per sample (its overlay digest does not
        change across thresholds, so the VLM cache answers the rest).
        """

        target = prompt.strip()
        if self._failure_notice is not None:
            return CalibrationTrace(
                accepted=False,
                confidence=0.0,
                reasoning="SAM 3 returned no masks, and the vision check is unavailable.",
            )
        artifact = self._overlay_artifact(sample, [])
        try:
            judgment = self.validator.validate(
                overlay_uri=artifact.uri,
                prompt=_presence_instruction(target),
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
            return CalibrationTrace(
                accepted=False,
                confidence=0.0,
                reasoning="SAM 3 returned no masks, and the vision check is unavailable.",
            )
        # Measured on qwen3-vl:4b: for a plain image it answers
        # target_present=false correctly ("a buffalo, not an elephant") but
        # with a low confidence, because the instruction tells it to leave
        # the mask fields false. Presence is the only question here, so the
        # boolean decides.
        if not judgment.target_present:
            return CalibrationTrace(
                accepted=True,
                confidence=judgment.confidence,
                reasoning=(
                    f"No {target} in this image, so returning no masks is correct. "
                    f"{judgment.reasoning}"
                ),
                overlay_id=artifact.overlay_id,
                target_present=False,
            )
        return CalibrationTrace(
            accepted=False,
            confidence=judgment.confidence,
            reasoning=f"SAM 3 found nothing, but the image shows a {target}. {judgment.reasoning}",
            overlay_id=artifact.overlay_id,
            target_present=bool(judgment.target_present),
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


def _instance_instruction(target: str) -> str:
    return (
        f"Target object: {target}. The image is a crop of a photo. ONE object is "
        "outlined with a red box and a light red mask. Judge only that outlined object. "
        f"target_present: true only if the outlined object is a {target}; false if it "
        "is anything else, such as a different kind of vehicle, a person, or background. "
        f"false_positive: true if the outlined object is not a {target}. "
        "fully_detected: true if the red mask covers the whole object. "
        "mask_tight: true if the mask follows the object's outline closely. "
        "Return confidence from 0 to 1 and one short reasoning sentence."
    )


def _presence_instruction(target: str) -> str:
    return (
        f"Target object: {target}. The image has NO masks or boxes drawn on it. "
        f"Answer one question: does at least one {target} appear anywhere in the photo? "
        f"target_present: true if a {target} is visible, even partly; false otherwise. "
        "Set fully_detected, false_positive and mask_tight to false. "
        "Return confidence from 0 to 1 and one short reasoning sentence."
    )
