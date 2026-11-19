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
    """Like SAM 3: post-processing keeps predictions whose score >= threshold."""

    def __init__(self, scores=(0.9, 0.5)):
        self.thresholds = []
        self.scores = scores

    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        self.thresholds.append(threshold)
        return [
            Sam3Prediction(score=score, box=(0, 0, 1, 1), mask=None)
            for score in self.scores
            if threshold is None or score >= threshold
        ]


def samples(count=3):
    return [CalibrationSample(image_id=str(index), image=Image.new("RGB", (1, 1))) for index in range(count)]


def test_calibration_runs_sam3_once_per_sample_and_filters_by_score():
    provider = FakeProvider()

    result = Sam3Calibrator(
        provider,
        # Accept only when the weak 0.5 prediction has been filtered out.
        lambda sample, predictions: bool(predictions) and min(p.score for p in predictions) >= 0.7,
        minimum_agreement=1.0,
    ).calibrate(samples(), prompt="brown bear", thresholds=[0.9, 0.3, 0.7])

    assert result.status == "calibrated"
    assert [attempt.threshold for attempt in result.attempts] == [0.3, 0.7, 0.9]
    assert [attempt.agreement for attempt in result.attempts] == [0.0, 1.0, 1.0]
    # One forward pass per sample at the lowest threshold, not one per threshold.
    assert provider.thresholds == [0.3, 0.3, 0.3]
    assert result.forward_passes == 3
    # 0.7 and 0.9 tie; the lower middle of the best run is chosen.
    assert result.threshold == 0.7


def test_calibration_picks_the_middle_of_the_best_run_not_the_first_pass():
    result = Sam3Calibrator(
        FakeProvider(scores=(0.95,)), lambda sample, predictions: bool(predictions), minimum_agreement=1.0
    ).calibrate(samples(), prompt="bear", thresholds=[0.2, 0.4, 0.6, 0.8, 0.9])
    assert result.threshold == 0.6


def test_correct_negatives_count_but_a_pass_needs_a_found_target():
    class PresenceValidator:
        """Image "absent" has no bear: an empty result there is correct."""

        def validate_with_trace(self, sample, predictions, prompt, *, threshold=None):
            if sample.image_id == "absent":
                return CalibrationTrace(accepted=not predictions, target_present=False)
            return CalibrationTrace(accepted=bool(predictions), target_present=True)

    class SceneProvider:
        def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
            if image.size == (2, 2):  # the image without a bear
                return []
            return [Sam3Prediction(score=0.9, box=(0, 0, 1, 1), mask=None)]

    scene = [
        CalibrationSample(image_id="bear", image=Image.new("RGB", (1, 1))),
        CalibrationSample(image_id="absent", image=Image.new("RGB", (2, 2))),
    ]
    result = Sam3Calibrator(SceneProvider(), PresenceValidator(), minimum_agreement=1.0).calibrate(
        scene, prompt="bear", thresholds=[0.5]
    )
    assert result.status == "calibrated"
    assert result.attempts[0].accepted_count == 2
    assert result.attempts[0].positive_count == 1

    only_negatives = Sam3Calibrator(SceneProvider(), PresenceValidator(), minimum_agreement=0.5).calibrate(
        [scene[1]], prompt="bear", thresholds=[0.5]
    )
    assert only_negatives.status == "no_target_in_samples"
    assert only_negatives.threshold is None


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
