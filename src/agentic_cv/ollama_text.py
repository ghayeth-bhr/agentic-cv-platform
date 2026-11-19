"""Structured local Ollama client for turning a chat message into a TaskSpec."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .contracts import TaskSpec

TASK_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_objects": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
        "task_type": {"type": "string", "enum": ["detection", "segmentation"]},
        "visual_attributes": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "sample_size": {"type": "integer", "minimum": 1, "maximum": 20},
        "max_iterations": {"type": "integer", "minimum": 1, "maximum": 10},
    },
    "required": [
        "target_objects",
        "task_type",
        "visual_attributes",
        "sample_size",
        "max_iterations",
    ],
}

_SYSTEM_PROMPT = (
    "You plan computer-vision labeling tasks for a promptable segmentation model. "
    "Return only JSON matching the provided schema. target_objects must be short noun "
    "phrases describing what to label, not the user's whole instruction. Use "
    "segmentation when the user asks for masks, polygons, or a segmentation dataset; "
    "otherwise use detection. visual_attributes lists colors, clothing, or states that "
    "are not already part of the object phrase. sample_size is the number of images to "
    "calibrate on (1-20). max_iterations is the number of threshold attempts (1-10)."
)


@dataclass(frozen=True, kw_only=True)
class OllamaTextClient:
    """Ask a locally hosted Ollama model for a structured TaskSpec."""

    model: str = "qwen3-vl:4b"
    base_url: str = "http://localhost:11434"
    timeout_sec: float = 30.0
    num_ctx: int = 4096
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be greater than 0")
        if self.num_ctx < 1:
            raise ValueError("num_ctx must be at least 1")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")

    def plan_task(self, message: str) -> TaskSpec:
        if not message.strip():
            raise ValueError("message must not be empty")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": message.strip()},
            ],
            "format": TASK_SPEC_SCHEMA,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "num_predict": 512,
            },
        }
        request = Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                response_body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Ollama request failed at {self.base_url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError("Ollama response was not valid JSON") from exc

        try:
            message = response_body["message"]
            content = message["content"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Ollama response is missing message.content") from exc
        if not content:
            # Qwen3-VL via Ollama can place the structured answer in
            # message.thinking while returning an empty content field, even
            # with think=false. Accept that field as a narrow fallback.
            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                content = thinking
        if isinstance(content, str):
            try:
                specification = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError("Ollama message.content was not valid JSON") from exc
        elif isinstance(content, dict):
            specification = content
        else:
            raise ValueError("Ollama message.content must be a JSON object or string")
        if not isinstance(specification, dict):
            raise ValueError("Ollama message.content must decode to an object")
        return _spec_from_payload(specification)

    def extract_draft(self, message: str, context: dict[str, Any]) -> dict[str, Any]:
        """Structured slot extraction for one chat turn (see ``agent_dialogue``)."""

        if not message.strip():
            raise ValueError("message must not be empty")
        return _draft_request(self, message, context)


def _spec_from_payload(value: dict[str, Any]) -> TaskSpec:
    target_objects = value.get("target_objects")
    if not isinstance(target_objects, list) or not target_objects:
        raise ValueError("target_objects must be a non-empty array")
    cleaned_objects = tuple(
        item.strip() for item in target_objects if isinstance(item, str) and item.strip()
    )
    if not cleaned_objects:
        raise ValueError("target_objects must contain non-empty strings")
    task_type = value.get("task_type")
    if task_type not in ("detection", "segmentation"):
        raise ValueError("task_type must be detection or segmentation")
    visual_attributes = value.get("visual_attributes", [])
    if not isinstance(visual_attributes, list):
        raise ValueError("visual_attributes must be an array")
    cleaned_attributes = tuple(
        item.strip() for item in visual_attributes if isinstance(item, str) and item.strip()
    )
    sample_size = value.get("sample_size", 5)
    max_iterations = value.get("max_iterations", 5)
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise ValueError("sample_size must be an integer")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise ValueError("max_iterations must be an integer")
    return TaskSpec(
        target_objects=cleaned_objects,
        task_type=task_type,
        visual_attributes=cleaned_attributes,
        sample_size=max(1, min(20, sample_size)),
        max_iterations=max(1, min(10, max_iterations)),
    )


DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_type": {"type": "string", "enum": ["detection", "segmentation", "unknown"]},
        "classes_changed": {"type": "boolean"},
        "classes": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "attributes": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                },
                "required": ["name", "attributes"],
            },
        },
        "export_format": {"type": "string", "enum": ["coco", "yolo", "unknown"]},
    },
    "required": ["task_type", "classes_changed", "classes", "export_format"],
}

_DRAFT_SYSTEM_PROMPT = (
    "You extract a computer-vision labeling task from a chat turn. Return only JSON "
    "matching the schema. Never invent information the user did not give.\n"
    "- task_type: 'segmentation' if the user wants masks/polygons/segmentation, "
    "'detection' if they want boxes/detection, else 'unknown'.\n"
    "- classes: the COMPLETE updated list of object classes to label, each a short "
    "noun phrase (e.g. 'hard hat'), with colors/states in attributes (e.g. ['yellow']). "
    "Start from the current classes and apply the user's change (add, remove, replace).\n"
    "- classes_changed: false when this turn does not mention objects to label.\n"
    "- export_format: 'coco' or 'yolo' only if the user names it, else 'unknown'.\n"
    "If the agent just asked a question, the user's message is probably its answer."
)


def _draft_request(client: "OllamaTextClient", message: str, context: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": client.model,
        "messages": [
            {"role": "system", "content": _DRAFT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Current task: {json.dumps(context, ensure_ascii=False)}\n"
                    f"User message: {message.strip()}"
                ),
            },
        ],
        "format": DRAFT_SCHEMA,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {"num_ctx": client.num_ctx, "temperature": client.temperature, "num_predict": 512},
    }
    request = Request(
        f"{client.base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=client.timeout_sec) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Ollama request failed at {client.base_url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("Ollama response was not valid JSON") from exc
    try:
        message_body = body["message"]
        content = message_body["content"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Ollama response is missing message.content") from exc
    if not content and isinstance(message_body.get("thinking"), str):
        content = message_body["thinking"]
    value = json.loads(content) if isinstance(content, str) else content
    if not isinstance(value, dict):
        raise ValueError("Ollama draft must be a JSON object")
    return value
