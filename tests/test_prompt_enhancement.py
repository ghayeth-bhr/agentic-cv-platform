import json

import pytest

from agentic_cv.prompt_enhancement import OllamaPromptEnhancer, _parse_prompts


class FakeResponse:
    def __init__(self, body):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def test_ollama_prompt_enhancer_sends_structured_text_request(monkeypatch):
    requests = []

    def fake_urlopen(request, *, timeout):
        requests.append((request, timeout))
        return FakeResponse({"message": {"content": json.dumps({"prompts": ["red hard hat", "worker wearing red hard hat"]})}})

    monkeypatch.setattr("agentic_cv.prompt_enhancement.urlopen", fake_urlopen)
    result = OllamaPromptEnhancer(max_prompts=2).enhance("a worker wearing a red hard hat")

    assert result.source_description == "a worker wearing a red hard hat"
    assert result.prompts == ("red hard hat", "worker wearing red hard hat")
    body = json.loads(requests[0][0].data)
    assert body["model"] == "qwen3-vl:4b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"]["num_predict"] == 128
    assert body["format"]["required"] == ["prompts"]
    assert requests[0][1] == 30.0


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "prompts array"),
        ({"prompts": [""]}, "non-empty strings"),
        ({"prompts": ["a red hat."]}, "noun phrases"),
    ],
)
def test_parse_prompts_rejects_invalid_payloads(payload, message):
    with pytest.raises(ValueError, match=message):
        _parse_prompts(payload, max_prompts=3)


def test_prompt_enhancer_rejects_empty_description():
    with pytest.raises(ValueError, match="description"):
        OllamaPromptEnhancer().enhance(" ")


def test_prompt_enhancer_accepts_qwen_json_returned_in_thinking(monkeypatch):
    monkeypatch.setattr(
        "agentic_cv.prompt_enhancement.urlopen",
        lambda request, timeout: FakeResponse({
            "message": {
                "content": "",
                "thinking": '{"prompts": ["yellow helmet"]}',
            }
        }),
    )

    result = OllamaPromptEnhancer().enhance("worker wearing yellow helmet")

    assert result.prompts == ("yellow helmet",)