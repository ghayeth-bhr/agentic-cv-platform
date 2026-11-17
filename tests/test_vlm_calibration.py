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
    assert client.calls == [("overlay://image-1", "image-1")]


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
    assert client.calls == [("overlay://image-1/0.6", "bear")]