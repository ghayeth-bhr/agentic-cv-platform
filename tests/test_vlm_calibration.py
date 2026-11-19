from PIL import Image

from agentic_cv.calibration import CalibrationSample, Sam3Calibrator
from agentic_cv.sam3 import Sam3Prediction
from agentic_cv.vlm_validator import VlmCalibrationValidator, VlmValidator


class FakeVlmClient:
    def __init__(self):
        self.calls = []

    def judge(self, *, overlay_uri, prompt):
        self.calls.append((overlay_uri, prompt))
        return {
            "target_present": True,
            "fully_detected": True,
            "false_positive": False,
            "mask_tight": True,
            "confidence": 0.9,
            "reasoning": "The target is fully visible.",
        }


class FakeSam3Provider:
    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        return [Sam3Prediction(score=threshold, box=(0, 0, 1, 1), mask=None)]


def test_vlm_calibration_validator_bridges_calibration_and_reuses_cache():
    client = FakeVlmClient()
    validator = VlmCalibrationValidator(
        VlmValidator(client),
        lambda sample, predictions: f"overlay://{sample.image_id}",
        model_revision="vlm-r1",
    )
    sample = CalibrationSample(image_id="image-1", image=Image.new("RGB", (1, 1)))
    predictions = [Sam3Prediction(score=0.8, box=(0, 0, 1, 1), mask=None)]

    assert validator(sample, predictions) is True
    assert validator(sample, predictions) is True
    assert client.calls[0][0] == "overlay://image-1"
    assert "Target object: image-1" in client.calls[0][1]


def test_vlm_calibration_validator_runs_inside_calibration():
    client = FakeVlmClient()
    validator = VlmCalibrationValidator(
        VlmValidator(client),
        lambda sample, predictions: f"overlay://{sample.image_id}/{predictions[0].score}",
    )
    samples = [CalibrationSample(image_id="image-1", image=Image.new("RGB", (1, 1)))]

    result = Sam3Calibrator(
        FakeSam3Provider(), validator, minimum_agreement=1.0
    ).calibrate(samples, prompt="bear", thresholds=[0.6])

    assert result.status == "calibrated"
    assert result.threshold == 0.6
    assert client.calls[0][0] == "overlay://image-1/0.6"
    assert "Target object: bear" in client.calls[0][1]
    verdict = result.attempts[0].verdicts[0]
    assert verdict.accepted is True
    assert verdict.confidence == 0.9
    assert verdict.reasoning == "The target is fully visible."
    assert verdict.overlay_id == "0.6"


def test_vlm_calibration_trace_skips_empty_predictions_without_a_vlm_call():
    client = FakeVlmClient()
    validator = VlmCalibrationValidator(
        VlmValidator(client),
        lambda sample, predictions: f"overlay://{sample.image_id}",
    )
    sample = CalibrationSample(image_id="empty", image=Image.new("RGB", (1, 1)))

    trace = validator.validate_with_trace(sample, [], "bear", threshold=0.9)

    assert trace.accepted is False
    assert trace.overlay_id is None
    assert "no masks" in (trace.reasoning or "").lower()
    assert client.calls == []


def test_vlm_runtime_failure_falls_back_once_and_reports_the_reason():
    class FailingClient:
        def __init__(self):
            self.calls = 0

        def judge(self, *, overlay_uri, prompt):
            self.calls += 1
            raise RuntimeError("Ollama request timed out")

    client = FailingClient()
    validator = VlmCalibrationValidator(
        VlmValidator(client),
        lambda sample, predictions: f"overlay://{sample.image_id}",
        fallback=lambda sample, predictions: len(predictions) > 0,
    )
    sample = CalibrationSample(image_id="image-1", image=Image.new("RGB", (1, 1)))
    predictions = [Sam3Prediction(score=0.8, box=(0, 0, 1, 1), mask=None)]

    first = validator.validate_with_trace(sample, predictions, "bear")
    second = validator.validate_with_trace(sample, predictions, "bear")

    assert first.accepted is True
    assert second.accepted is True
    assert first.overlay_id is None
    assert "timed out" in (validator.failure_notice or "")
    assert "fallback" in (validator.failure_notice or "")
    assert client.calls == 1