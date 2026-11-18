import pytest

from agentic_cv.vlm_validator import VlmBudgetExceeded, VlmValidator, ValidationJudgment


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
