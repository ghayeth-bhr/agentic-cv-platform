import json

import pytest

from agentic_cv.agent import (
    OllamaTaskPlanner,
    RulesTaskPlanner,
    clean_target_phrase,
    prompt_for_spec,
)
from agentic_cv.contracts import TaskSpec
from agentic_cv.ollama_text import OllamaTextClient


class FakeResponse:
    def __init__(self, body):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class SuccessfulClient:
    def plan_task(self, message):
        return TaskSpec(
            target_objects=("worker",),
            task_type="segmentation",
            visual_attributes=("yellow vest",),
            sample_size=4,
            max_iterations=3,
        )


class FailingClient:
    def plan_task(self, message):
        raise ConnectionError("ollama is down")


def test_rules_planner_strips_instruction_and_infers_segmentation():
    result = RulesTaskPlanner().plan(
        "Create a segmentation dataset of workers wearing yellow vests."
    )

    assert result.mode == "rules"
    assert result.note is None
    assert result.spec.task_type == "segmentation"
    assert result.spec.target_objects == ("workers wearing yellow vests",)
    assert prompt_for_spec(result.spec) == "workers wearing yellow vests"
    assert clean_target_phrase("Segment the workers.") == "workers"


def test_ollama_planner_uses_model_spec_and_applies_explicit_overrides():
    result = OllamaTaskPlanner(client=SuccessfulClient()).plan(
        "find workers",
        task_type="detection",
        sample_size=7,
    )

    assert result.mode == "ollama"
    assert result.spec.task_type == "detection"
    assert result.spec.target_objects == ("worker",)
    assert result.spec.visual_attributes == ("yellow vest",)
    assert result.spec.sample_size == 7
    assert prompt_for_spec(result.spec) == "worker with yellow vest"


def test_ollama_planner_falls_back_to_rules_when_model_is_down():
    result = OllamaTaskPlanner(client=FailingClient()).plan(
        "Create a detection dataset of hard hats."
    )

    assert result.mode == "rules"
    assert "local rule-based planner" in (result.note or "")
    assert result.spec.target_objects == ("hard hats",)


def test_prompt_for_spec_does_not_repeat_attributes_already_in_the_object():
    spec = TaskSpec(
        target_objects=("yellow vests",),
        task_type="segmentation",
        visual_attributes=("yellow",),
    )

    assert prompt_for_spec(spec) == "yellow vests"


def test_ollama_text_client_requests_and_parses_task_spec(monkeypatch):
    requests = []

    def fake_urlopen(request, *, timeout):
        requests.append((request, timeout))
        return FakeResponse(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "target_objects": ["worker"],
                            "task_type": "segmentation",
                            "visual_attributes": ["yellow vest"],
                            "sample_size": 5,
                            "max_iterations": 4,
                        }
                    )
                }
            }
        )

    monkeypatch.setattr("agentic_cv.ollama_text.urlopen", fake_urlopen)
    client = OllamaTextClient(
        model="qwen3-vl:4b", base_url="http://ollama.test:11434", timeout_sec=9
    )

    spec = client.plan_task("Create a segmentation dataset of workers in yellow vests.")

    request, timeout = requests[0]
    body = json.loads(request.data)
    assert request.full_url == "http://ollama.test:11434/api/chat"
    assert timeout == 9
    assert body["format"]["required"]
    assert body["stream"] is False
    assert body["think"] is False
    assert spec.target_objects == ("worker",)
    assert spec.visual_attributes == ("yellow vest",)
    assert spec.max_iterations == 4


def test_ollama_text_client_rejects_invalid_task_spec(monkeypatch):
    monkeypatch.setattr(
        "agentic_cv.ollama_text.urlopen",
        lambda request, *, timeout: FakeResponse(
            {"message": {"content": json.dumps({"target_objects": []})}}
        ),
    )

    with pytest.raises(ValueError, match="target_objects"):
        OllamaTextClient().plan_task("label everything")


def test_ollama_text_client_accepts_qwen_thinking_fallback(monkeypatch):
    monkeypatch.setattr(
        "agentic_cv.ollama_text.urlopen",
        lambda request, *, timeout: FakeResponse(
            {
                "message": {
                    "content": "",
                    "thinking": json.dumps(
                        {
                            "target_objects": ["worker"],
                            "task_type": "segmentation",
                            "visual_attributes": ["yellow vest"],
                            "sample_size": 5,
                            "max_iterations": 4,
                        }
                    ),
                }
            }
        ),
    )

    spec = OllamaTextClient().plan_task("workers in yellow vests")

    assert spec.target_objects == ("worker",)
    assert spec.visual_attributes == ("yellow vest",)
