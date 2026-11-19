from PIL import Image
import pytest

from agentic_cv.calibration import (
    CalibrationCancelled,
    CalibrationSample,
    CalibrationTrace,
    Sam3Calibrator,
)
from agentic_cv.sam3 import Sam3Prediction


class FakeProvider:
    def __init__(self):
        self.thresholds = []

    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        self.thresholds.append(threshold)
        return [Sam3Prediction(score=threshold, box=(0, 0, 1, 1), mask=None)]


def samples(count=3):
    return [CalibrationSample(image_id=str(index), image=Image.new("RGB", (1, 1))) for index in range(count)]


def test_calibration_uses_multiple_samples_and_returns_first_acceptable_threshold():
    provider = FakeProvider()

    result = Sam3Calibrator(
        provider,
        lambda sample, predictions: predictions[0].score >= 0.7,
        minimum_agreement=1.0,
    ).calibrate(samples(), prompt="brown bear", thresholds=[0.3, 0.7, 0.9])

    assert result.status == "calibrated"
    assert result.threshold == 0.7
    assert [attempt.agreement for attempt in result.attempts] == [0.0, 1.0]
    assert provider.thresholds == [0.3, 0.3, 0.3, 0.7, 0.7, 0.7]


def test_calibration_reports_max_iterations_without_false_success():
    result = Sam3Calibrator(
        FakeProvider(), lambda sample, predictions: False, max_iterations=2
    ).calibrate(samples(), prompt="bear", thresholds=[0.2, 0.5, 0.8])

    assert result.status == "max_iterations"
    assert result.threshold is None
    assert len(result.attempts) == 2
    assert result.reason is not None


def test_calibration_keeps_per_sample_trace_evidence():
    class TraceValidator:
        def validate_with_trace(self, sample, predictions, prompt, *, threshold=None):
            return CalibrationTrace(
                accepted=sample.image_id != "1",
                confidence=0.91,
                reasoning=f"checked {sample.image_id}",
                overlay_id=f"overlay-{sample.image_id}.png",
            )

    result = Sam3Calibrator(
        FakeProvider(), TraceValidator(), minimum_agreement=0.4
    ).calibrate(samples(2), prompt="bear", thresholds=[0.5])

    verdicts = result.attempts[0].verdicts
    assert [verdict.accepted for verdict in verdicts] == [True, False]
    assert verdicts[0].confidence == 0.91
    assert verdicts[0].reasoning == "checked 0"
    assert verdicts[0].overlay_id == "overlay-0.png"


def test_calibration_stops_between_images_when_cancelled():
    class CountingProvider:
        def __init__(self):
            self.thresholds = []

        def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
            self.thresholds.append(threshold)
            return [Sam3Prediction(score=threshold, box=(0, 0, 1, 1), mask=None)]

    provider = CountingProvider()

    def should_cancel():
        return len(provider.thresholds) >= 1

    with pytest.raises(CalibrationCancelled):
        Sam3Calibrator(provider, lambda sample, predictions: True).calibrate(
            samples(3),
            prompt="bear",
            thresholds=[0.5],
            should_cancel=should_cancel,
        )

    assert provider.thresholds == [0.5]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"samples": [], "prompt": "bear", "thresholds": [0.5]},
        {"samples": samples(1), "prompt": " ", "thresholds": [0.5]},
        {"samples": samples(1), "prompt": "bear", "thresholds": []},
    ],
)
def test_calibration_rejects_missing_inputs(kwargs):
    calibrator = Sam3Calibrator(FakeProvider(), lambda sample, predictions: True)

    with pytest.raises(ValueError):
        calibrator.calibrate(**kwargs)
