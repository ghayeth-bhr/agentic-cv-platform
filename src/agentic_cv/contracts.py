"""Versioned control-plane contracts for the platform."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

SCHEMA_VERSION = 1
TaskType = Literal["detection", "segmentation"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_text(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


@dataclass(frozen=True, kw_only=True)
class TaskSpec:
    target_objects: tuple[str, ...]
    task_type: TaskType
    visual_attributes: tuple[str, ...] = ()
    sample_size: int = 5
    max_iterations: int = 5

    def __post_init__(self) -> None:
        if not self.target_objects or any(not item.strip() for item in self.target_objects):
            raise ValueError("target_objects must contain at least one non-empty value")
        if self.task_type not in ("detection", "segmentation"):
            raise ValueError("task_type must be detection or segmentation")
        if self.sample_size < 1 or self.max_iterations < 1:
            raise ValueError("sample_size and max_iterations must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskSpec":
        return cls(
            target_objects=tuple(value["target_objects"]),
            task_type=value["task_type"],
            visual_attributes=tuple(value.get("visual_attributes", ())),
            sample_size=value.get("sample_size", 5),
            max_iterations=value.get("max_iterations", 5),
        )


@dataclass(frozen=True, kw_only=True)
class ArtifactManifest:
    tenant_id: UUID
    asset_id: UUID
    artifact_id: UUID
    kind: str
    storage_uri: str
    content_sha256: str
    model_revision: str | None = None
    parent_artifact_id: UUID | None = None
    schema_version: int = SCHEMA_VERSION
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        _require_text(self.kind, "kind")
        _require_text(self.storage_uri, "storage_uri")
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_sha256):
            raise ValueError("content_sha256 must be a lowercase SHA-256 hex digest")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported artifact schema version: {self.schema_version}")

    @classmethod
    def from_bytes(
        cls,
        *,
        tenant_id: UUID,
        asset_id: UUID,
        kind: str,
        storage_uri: str,
        content: bytes,
        model_revision: str | None = None,
        artifact_id: UUID | None = None,
    ) -> "ArtifactManifest":
        return cls(
            tenant_id=tenant_id,
            asset_id=asset_id,
            artifact_id=artifact_id or uuid4(),
            kind=kind,
            storage_uri=storage_uri,
            content_sha256=hashlib.sha256(content).hexdigest(),
            model_revision=model_revision,
        )

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        for key in ("tenant_id", "asset_id", "artifact_id", "parent_artifact_id"):
            if value[key] is not None:
                value[key] = str(value[key])
        value["created_at"] = self.created_at.isoformat()
        return value


@dataclass(frozen=True, kw_only=True)
class EventEnvelope:
    event_type: str
    tenant_id: UUID
    aggregate_type: str
    aggregate_id: UUID
    payload: dict[str, Any]
    event_id: UUID = field(default_factory=uuid4)
    correlation_id: UUID = field(default_factory=uuid4)
    causation_id: UUID | None = None
    idempotency_key: str | None = None
    occurred_at: datetime = field(default_factory=_utc_now)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.event_type, "event_type")
        _require_text(self.aggregate_type, "aggregate_type")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported event schema version: {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        for key in ("event_id", "tenant_id", "aggregate_id", "correlation_id", "causation_id"):
            if value[key] is not None:
                value[key] = str(value[key])
        value["occurred_at"] = self.occurred_at.isoformat()
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
