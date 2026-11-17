"""Structured, budgeted, cache-aware VLM validation for calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class VlmClient(Protocol):
    def judge(self, *, overlay_uri: str, prompt: str) -> dict[str, Any]: ...


class VlmBudgetExceeded(RuntimeError):
    """Raised when a validation run reaches its configured VLM call budget."""


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
