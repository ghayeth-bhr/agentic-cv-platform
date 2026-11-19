import base64
import json
from pathlib import Path

import pytest

from agentic_cv.ollama_vlm import OllamaVlmClient


class FakeResponse:
    def __init__(self, body):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def test_ollama_client_sends_structured_multimodal_request(monkeypatch, tmp_path: Path):
    overlay = tmp_path / "overlay.jpg"
    overlay.write_bytes(b"overlay-bytes")
    requests = []

    def fake_urlopen(request, *, timeout):
        requests.append((request, timeout))
        return FakeResponse({"message": {"content": json.dumps({"confidence": 0.9})}})

    monkeypatch.setattr("agentic_cv.ollama_vlm.urlopen", fake_urlopen)
    client = OllamaVlmClient(model="qwen3-vl:4b", timeout_sec=12, num_ctx=2048)

    assert client.judge(overlay_uri=str(overlay), prompt="a brown bear") == {"confidence": 0.9}
    request, timeout = requests[0]
    body = json.loads(request.data)
    assert request.full_url == "http://localhost:11434/api/chat"
    assert timeout == 12
    assert body["model"] == "qwen3-vl:4b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"]["num_predict"] == 256
    assert body["messages"][0]["content"] == "a brown bear"
    assert base64.b64decode(body["messages"][0]["images"][0]) == b"overlay-bytes"
    assert body["format"]["required"]


def test_ollama_client_accepts_file_uri_and_rejects_remote_overlay(tmp_path: Path):
    overlay = tmp_path / "overlay.jpg"
    overlay.write_bytes(b"bytes")
    assert overlay.as_uri().startswith("file:")

    with pytest.raises(ValueError, match="local paths"):
        OllamaVlmClient().judge(overlay_uri="https://example.test/overlay.jpg", prompt="bear")


def test_ollama_client_accepts_json_from_the_thinking_field(monkeypatch, tmp_path: Path):
    overlay = tmp_path / "overlay.jpg"
    overlay.write_bytes(b"bytes")
    thinking = json.dumps({"confidence": 0.8, "reasoning": "ok"})
    monkeypatch.setattr(
        "agentic_cv.ollama_vlm.urlopen",
        lambda request, timeout: FakeResponse(
            {"message": {"content": "", "thinking": thinking}}
        ),
    )

    assert OllamaVlmClient().judge(overlay_uri=str(overlay), prompt="bear") == {
        "confidence": 0.8,
        "reasoning": "ok",
    }


def test_ollama_client_rejects_malformed_response(monkeypatch, tmp_path: Path):
    overlay = tmp_path / "overlay.jpg"
    overlay.write_bytes(b"bytes")
    monkeypatch.setattr(
        "agentic_cv.ollama_vlm.urlopen",
        lambda request, timeout: FakeResponse({"message": {"content": "[]"}}),
    )

    with pytest.raises(ValueError, match="object"):
        OllamaVlmClient().judge(overlay_uri=str(overlay), prompt="bear")