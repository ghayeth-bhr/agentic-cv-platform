"""Local Ollama VLM client for structured calibration judgments."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


VALIDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_present": {"type": "boolean"},
        "fully_detected": {"type": "boolean"},
        "false_positive": {"type": "boolean"},
        "mask_tight": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"},
    },
    "required": [
        "target_present",
        "fully_detected",
        "false_positive",
        "mask_tight",
        "confidence",
        "reasoning",
    ],
}


@dataclass(frozen=True, kw_only=True)
class OllamaVlmClient:
    """Call a locally hosted Ollama multimodal model over its HTTP API."""

    model: str = "qwen3-vl:4b"
    base_url: str = "http://localhost:11434"
    timeout_sec: float = 60.0
    num_ctx: int = 4096
    temperature: float = 0.0
    keep_alive: str = "10m"

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

    def judge(self, *, overlay_uri: str, prompt: str) -> dict[str, Any]:
        image_bytes = _read_overlay(overlay_uri)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                }
            ],
            "format": VALIDATION_SCHEMA,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "num_predict": 256,
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
        if not isinstance(content, str):
            raise ValueError("Ollama message.content must be a JSON string")

        candidates = [content]
        thinking = message.get("thinking") if isinstance(message, dict) else None
        if isinstance(thinking, str) and thinking.strip():
            candidates.append(thinking)
        judgment = None
        saw_json = False
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            saw_json = True
            if isinstance(parsed, dict):
                judgment = parsed
                break
        if judgment is None:
            if saw_json:
                raise ValueError("Ollama message.content must decode to an object")
            raise ValueError("Ollama message.content was not valid JSON")
        return judgment


def _read_overlay(overlay_uri: str) -> bytes:
    if not overlay_uri.strip():
        raise ValueError("overlay_uri must not be empty")
    parsed = urlparse(overlay_uri)
    is_windows_path = len(parsed.scheme) == 1 and overlay_uri[1:2] == ":"
    if parsed.scheme not in ("", "file") and not is_windows_path:
        raise ValueError("Ollama overlays must be local paths or file:// URIs")
    if parsed.scheme == "file":
        path_text = unquote(parsed.path)
        if os.name == "nt" and path_text.startswith("/") and path_text[2:3] == ":":
            path_text = path_text[1:]
        path = Path(path_text)
    else:
        path = Path(overlay_uri)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read overlay image: {path}") from exc