import numpy as np
import pytest
from PIL import Image

from agentic_cv.calibration import CalibrationSample
from agentic_cv.sam3 import Sam3Prediction
from agentic_cv.vlm_validator import (
    VlmBudgetExceeded,
    VlmCalibrationValidator,
    VlmValidator,
    ValidationJudgment,
)


class FakeVlm:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def judge(self, *, overlay_uri, prompt):
        self.calls += 1
        return self.payload


GOOD = {
    "target_present": True,
    "fully_detected": True,
    "false_positive": False,
    "mask_tight": True,
    "confidence": 0.9,
    "reasoning": "The target is fully covered by the mask.",
}


def test_validator_parses_acceptance_and_caches_results():
    client = FakeVlm(GOOD)
    validator = VlmValidator(client, max_calls=1)

    first = validator.validate(overlay_uri="s3://overlay/1", prompt="bear")
    second = validator.validate(overlay_uri="s3://overlay/1", prompt="bear")

    assert first == second
    assert first.accepted()
    assert client.calls == 1
    assert validator.calls_made == 1


def test_validator_enforces_budget_and_model_cache_key():
    client = FakeVlm(GOOD)
    validator = VlmValidator(client, max_calls=1)
    validator.validate(overlay_uri="overlay", prompt="bear", model_revision="v1")

    with pytest.raises(VlmBudgetExceeded):
        validator.validate(overlay_uri="overlay", prompt="bear", model_revision="v2")


def test_invalid_payload_and_rejection_are_explicit():
    client = FakeVlm({"target_present": True})
    validator = VlmValidator(client)
    with pytest.raises(ValueError, match="missing fields"):
        validator.validate(overlay_uri="overlay", prompt="bear")

    rejected = ValidationJudgment.from_payload({
        **GOOD, "false_positive": True,
    })
    assert not rejected.accepted()


def test_validator_configuration_and_inputs_are_checked():
    with pytest.raises(ValueError, match="max_calls"):
        VlmValidator(FakeVlm(GOOD), max_calls=0)
    validator = VlmValidator(FakeVlm(GOOD))
    with pytest.raises(ValueError, match="overlay_uri"):
        validator.validate(overlay_uri="", prompt="bear")


def test_validator_release_is_idempotent_and_only_after_a_call():
    class UnloadableVlm(FakeVlm):
        def __init__(self):
            super().__init__(GOOD)
            self.unloads = 0

        def unload(self) -> bool:
            self.unloads += 1
            return True

    client = UnloadableVlm()
    validator = VlmValidator(client)

    assert validator.release() is False
    assert client.unloads == 0

    validator.validate(overlay_uri="overlay", prompt="bear")
    assert validator.release() is True
    assert validator.release() is False
    assert client.unloads == 1


def test_low_score_masks_are_rejected_without_a_vlm_call():
    client = FakeVlm(GOOD)
    validator = VlmCalibrationValidator(
        VlmValidator(client),
        lambda sample, predictions: "overlay.png",
        minimum_prediction_score=0.2,
    )
    sample = CalibrationSample(
        image_id="low", image=Image.new("RGB", (4, 4), (0, 0, 0))
    )
    prediction = Sam3Prediction(
        score=0.05, box=(0, 0, 4, 4), mask=np.ones((4, 4), dtype=bool)
    )

    trace = validator.validate_with_trace(sample, [prediction], "target")

    assert trace.accepted is False
    assert "too permissive" in trace.reasoning
    assert client.calls == 0
