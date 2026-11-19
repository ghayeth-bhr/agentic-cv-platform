from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from agentic_cv import calibration_validation
from agentic_cv.calibration import CalibrationSample
from agentic_cv.calibration_validation import (
    any_prediction_selection,
    build_overlay_builder,
    default_validator_selection,
    probe_ollama,
)
from agentic_cv.sam3 import Sam3Prediction


TENANT_ID = "00000000-0000-0000-0000-000000000001"
SESSION_ID = "00000000-0000-0000-0000-000000000002"


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_probe_ollama_reports_missing_model(monkeypatch):
    monkeypatch.setattr(
        calibration_validation,
        "urlopen",
        lambda *args, **kwargs: _Response({"models": [{"name": "llama3:8b"}]}),
    )

    reason = probe_ollama(
        base_url="http://127.0.0.1:11434",
        model="qwen3-vl:4b",
        timeout_sec=0.1,
    )

    assert reason is not None
    assert "qwen3-vl:4b is missing" in reason
    assert "llama3:8b" in reason


def test_disabled_validation_is_explicitly_labeled(monkeypatch):
    monkeypatch.setenv("AGENTIC_CV_AGENT_VALIDATOR", "any-prediction")

    selection = default_validator_selection(
        overlay_root=Path("unused"),
        tenant_id=TENANT_ID,
        session_id=SESSION_ID,
    )

    assert selection.mode == "any_prediction"
    assert selection.notice is not None
    assert "AGENTIC_CV_AGENT_VALIDATOR" in selection.notice
    assert selection.validator(
        CalibrationSample(image_id="one", image=Image.new("RGB", (1, 1))),
        [Sam3Prediction(score=0.9, box=(0, 0, 1, 1), mask=None)],
    )


def test_unreachable_ollama_falls_back_with_a_notice(monkeypatch):
    monkeypatch.delenv("AGENTIC_CV_AGENT_VALIDATOR", raising=False)
    monkeypatch.setattr(
        calibration_validation,
        "probe_ollama",
        lambda **kwargs: "Ollama is not reachable at http://localhost:11434",
    )

    selection = default_validator_selection(
        overlay_root=Path("unused"),
        tenant_id=TENANT_ID,
        session_id=SESSION_ID,
    )

    assert selection.mode == "any_prediction"
    assert selection.notice is not None
    assert "fallback" in selection.notice
    assert "at least one prediction" in selection.notice


def test_reachable_ollama_selects_the_vlm(monkeypatch):
    monkeypatch.delenv("AGENTIC_CV_AGENT_VALIDATOR", raising=False)
    monkeypatch.setenv("AGENTIC_CV_VLM_MODEL", "qwen3-vl:4b")
    monkeypatch.setattr(calibration_validation, "probe_ollama", lambda **kwargs: None)

    selection = default_validator_selection(
        overlay_root=Path("unused"),
        tenant_id=TENANT_ID,
        session_id=SESSION_ID,
    )

    assert selection.mode == "vlm"
    assert selection.model == "qwen3-vl:4b"
    assert selection.notice is None
    assert selection.current_notice() is None
    assert selection.validator.__class__.__name__ == "VlmCalibrationValidator"


def test_overlay_builder_writes_a_real_png(tmp_path):
    builder = build_overlay_builder(
        overlay_root=tmp_path,
        tenant_id=TENANT_ID,
        session_id=SESSION_ID,
    )
    image = Image.new("RGB", (12, 10), (20, 40, 60))
    mask = np.ones((10, 12), dtype=bool)
    prediction = Sam3Prediction(score=0.88, box=(1, 1, 11, 9), mask=mask)

    artifact = builder(
        CalibrationSample(image_id="frame-1", image=image), [prediction]
    )

    path = Path(artifact.uri)
    assert path.is_file()
    assert path.parent == tmp_path / TENANT_ID / SESSION_ID
    assert artifact.overlay_id.endswith(".png")
    assert len(artifact.overlay_id) == 28
    with Image.open(BytesIO(path.read_bytes())) as saved:
        assert saved.size == (12, 10)
        assert saved.format == "PNG"


def test_any_prediction_selection_is_the_explicit_fallback():
    selection = any_prediction_selection(notice="degraded")

    assert selection.mode == "any_prediction"
    assert selection.notice == "degraded"
    assert selection.model is None
