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


def test_empty_prediction_is_judged_by_target_presence():
    """Measured on a real run: correct empty results were auto-rejected."""

    class PresenceClient:
        def __init__(self, present):
            self.present = present
            self.calls = []

        def judge(self, *, overlay_uri, prompt):
            self.calls.append((overlay_uri, prompt))
            return {
                "target_present": self.present,
                "fully_detected": False,
                "false_positive": False,
                "mask_tight": False,
                "confidence": 0.9,
                "reasoning": "A small antelope, no elephant." if not self.present else "An elephant is visible.",
            }

    sample = CalibrationSample(image_id="frame-0", image=Image.new("RGB", (1, 1)))

    absent = PresenceClient(present=False)
    validator = VlmCalibrationValidator(VlmValidator(absent), lambda sample, predictions: f"overlay://{sample.image_id}/{len(predictions)}")
    trace = validator.validate_with_trace(sample, [], "elephant", threshold=0.9)
    assert trace.accepted is True
    assert trace.target_present is False
    assert "no masks is correct" in trace.reasoning
    assert "NO masks" in absent.calls[0][1]
    # The plain image is judged once; other thresholds hit the cache.
    validator.validate_with_trace(sample, [], "elephant", threshold=0.5)
    assert len(absent.calls) == 1

    present = PresenceClient(present=True)
    validator = VlmCalibrationValidator(VlmValidator(present), lambda sample, predictions: f"overlay://{sample.image_id}")
    missed = validator.validate_with_trace(sample, [], "elephant", threshold=0.9)
    assert missed.accepted is False
    assert missed.target_present is True
    assert "found nothing" in missed.reasoning


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

def test_absent_target_is_accepted_even_with_low_confidence():
    """qwen3-vl:4b answers target_present=false correctly but with low confidence."""

    class LowConfidenceAbsent:
        def judge(self, *, overlay_uri, prompt):
            return {
                "target_present": False,
                "fully_detected": False,
                "false_positive": False,
                "mask_tight": False,
                "confidence": 0.3,
                "reasoning": "The image shows a buffalo, not an elephant.",
            }

    validator = VlmCalibrationValidator(
        VlmValidator(LowConfidenceAbsent()), lambda sample, predictions: f"overlay://{sample.image_id}"
    )
    trace = validator.validate_with_trace(
        CalibrationSample(image_id="buffalo", image=Image.new("RGB", (1, 1))), [], "elephant"
    )
    assert trace.accepted is True
    assert trace.target_present is False
    assert "buffalo" in trace.reasoning


def test_vlm_client_downscales_large_overlays(tmp_path):
    import io

    from agentic_cv.ollama_vlm import _downscale

    path = tmp_path / "overlay.png"
    Image.new("RGB", (720, 1280), (10, 20, 30)).save(path)
    smaller = _downscale(path.read_bytes(), 768)
    with Image.open(io.BytesIO(smaller)) as image:
        assert max(image.size) == 768
        assert image.size == (432, 768)
    small = tmp_path / "small.png"
    Image.new("RGB", (300, 200)).save(small)
    assert _downscale(small.read_bytes(), 768) == small.read_bytes()
    assert _downscale(path.read_bytes(), 0) == path.read_bytes()
