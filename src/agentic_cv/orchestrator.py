"""Chat-to-TaskSpec orchestration.

Provides a deterministic, dependency-light way to turn a user message into a
`TaskSpec`. Prompt enhancement for SAM3 noun phrases remains a separate concern
(`prompt_enhancement.PromptEnhancer`); this module focuses on extracting the
task specification (target objects, task type, visual attributes).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .contracts import TaskSpec, TaskType

_SEGMENTATION_HINTS = (
    "segment", "segmentation", "mask", "outline", "contour", "pixel", "polygon",
)
_DETECTION_HINTS = (
    "detect", "detection", "box", "boxes", "bounding", "bbox", "find", "locate", "count",
)


def _infer_task_type(message: str, explicit: TaskType | None = None) -> TaskType:
    if explicit is not None:
        return explicit
    lowered = message.lower()
    if any(hint in lowered for hint in _SEGMENTATION_HINTS):
        return "segmentation"
    if any(hint in lowered for hint in _DETECTION_HINTS):
        return "detection"
    # Default to detection for generic object-finding requests.
    return "detection"


def _normalize_objects(values: list[str] | tuple[str, ...] | None, message: str) -> tuple[str, ...]:
    if values:
        cleaned = tuple(v.strip() for v in values if v and v.strip())
        if cleaned:
            return cleaned
    # Fallback: use the message itself as the target description.
    text = re.sub(r"\s+", " ", message).strip()
    return (text,) if text else ("object",)


@dataclass(frozen=True, kw_only=True)
class TaskOrchestrationRequest:
    message: str
    task_type: TaskType | None = None
    target_objects: tuple[str, ...] = ()
    visual_attributes: tuple[str, ...] = ()
    sample_size: int = 5
    max_iterations: int = 5


@dataclass(frozen=True, kw_only=True)
class TaskOrchestrator:
    """Deterministic chat → TaskSpec orchestrator.

    Callers may override any inferred field explicitly. The orchestrator keeps
    behavior predictable and side-effect free so it can be used in the API and
    in tests without external services.
    """

    default_sample_size: int = 5
    default_max_iterations: int = 5

    def orchestrate(self, request: TaskOrchestrationRequest) -> TaskSpec:
        message = (request.message or "").strip()
        if not message and not request.target_objects:
            raise ValueError("message or target_objects must be provided")
        task_type = _infer_task_type(message or " ".join(request.target_objects), request.task_type)
        target_objects = _normalize_objects(request.target_objects, message)
        visual_attributes = tuple(v.strip() for v in request.visual_attributes if v and v.strip())
        sample_size = request.sample_size if request.sample_size >= 1 else self.default_sample_size
        max_iterations = request.max_iterations if request.max_iterations >= 1 else self.default_max_iterations
        return TaskSpec(
            target_objects=target_objects,
            task_type=task_type,
            visual_attributes=visual_attributes,
            sample_size=sample_size,
            max_iterations=max_iterations,
        )
