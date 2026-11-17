"""Provider-neutral SAM3 prompt enhancement with a local Ollama adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class PromptEnhancer(Protocol):
    def enhance(self, description: str) -> "PromptEnhancement": ...


@dataclass(frozen=True, kw_only=True)
class PromptEnhancement:
    source_description: str
    prompts: tuple[str, ...]


PROMPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompts": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }
    },
    "required": ["prompts"],
}


@dataclass(frozen=True, kw_only=True)
class OllamaPromptEnhancer:
    model: str = "qwen3-vl:4b"
    base_url: str = "http://localhost:11434"
    timeout_sec: float = 30.0
    max_prompts: int = 3

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be greater than 0")
        if self.max_prompts < 1:
            raise ValueError("max_prompts must be at least 1")

    def enhance(self, description: str) -> PromptEnhancement:
        if not description.strip():
            raise ValueError("description must not be empty")
        instruction = (
            "Rewrite the user's computer-vision target into one to three short, "
            "concrete noun phrases suitable as SAM3 text prompts. Preserve the "
            "object and important visual attributes. Do not write questions, "
            "sentences, explanations, or actions. Return only the requested JSON.\n\n"
            f"User target: {description.strip()}"
        )
        request = Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": instruction}],
                    "format": PROMPT_SCHEMA,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.0, "num_predict": 128},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Ollama prompt enhancement failed at {self.base_url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError("Ollama response was not valid JSON") from exc

        try:
            message = envelope["message"]
            content = message["content"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Ollama response is missing message.content") from exc
        if not content and isinstance(message.get("thinking"), str):
            content = message["thinking"]
        if not isinstance(content, str):
            raise ValueError("Ollama message.content must be a JSON string")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Ollama prompt response was not valid JSON") from exc
        prompts = _parse_prompts(payload, max_prompts=self.max_prompts)
        return PromptEnhancement(source_description=description.strip(), prompts=prompts)


def _parse_prompts(payload: Any, *, max_prompts: int) -> tuple[str, ...]:
    if not isinstance(payload, dict) or not isinstance(payload.get("prompts"), list):
        raise ValueError("prompt response must contain a prompts array")
    prompts: list[str] = []
    for prompt in payload["prompts"][:max_prompts]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt response entries must be non-empty strings")
        normalized = " ".join(prompt.split())
        if normalized[-1:] in ".?!":
            raise ValueError("SAM3 prompts must be noun phrases, not sentences")
        prompts.append(normalized)
    if not prompts:
        raise ValueError("prompt response must contain at least one prompt")
    return tuple(prompts)