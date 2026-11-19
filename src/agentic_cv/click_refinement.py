"""Revisioned interactive point-click refinement sessions."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID, uuid4

import numpy as np
from PIL import Image

from .sam3 import Sam3Prediction, Sam3Provider

PointLabel = Literal["positive", "negative"]


@dataclass(frozen=True, kw_only=True)
class ClickPoint:
    x: float
    y: float
    label: PointLabel

    def validate(self, image: Image.Image) -> None:
        if self.label not in ("positive", "negative"):
            raise ValueError("point label must be positive or negative")
        if not 0 <= self.x < image.width or not 0 <= self.y < image.height:
            raise ValueError("point must be inside image bounds")


@dataclass(frozen=True, kw_only=True)
class RefinementTarget:
    image: Image.Image
    prompt: str
    category_id: int


@dataclass(frozen=True, kw_only=True)
class RefinementSnapshot:
    session_id: UUID
    image_id: str
    revision: int
    points: tuple[ClickPoint, ...]
    predictions: tuple[Sam3Prediction, ...]
    tenant_id: UUID | None = None
    version_id: UUID | None = None
    prompt: str = ""
    category_id: int = 0


class ClickRefinementProvider(Protocol):
    def refine(self, image: Image.Image, points: tuple[ClickPoint, ...]) -> list[Sam3Prediction]: ...


class PromptAwareClickRefinementProvider(Protocol):
    def refine_with_prompt(
        self,
        image: Image.Image,
        points: tuple[ClickPoint, ...],
        prompt: str,
    ) -> list[Sam3Prediction]: ...


class Sam3ClickRefinementProvider:
    """Select the best SAM 3 mask for a category using click points.

    This is deliberately a selection layer rather than a point-conditioned
    segmentation model: SAM 3 is prompted with the dataset category, and the
    positive/negative clicks rank the returned masks. The UI and docs call it
    "click-guided correction", not prompt-free segmentation.
    """

    def __init__(self, provider: Sam3Provider, *, threshold: float = 0.3) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.provider = provider
        self.threshold = threshold

    def refine(
        self, image: Image.Image, points: tuple[ClickPoint, ...]
    ) -> list[Sam3Prediction]:
        return self.refine_with_prompt(image, points, "object")

    def refine_with_prompt(
        self,
        image: Image.Image,
        points: tuple[ClickPoint, ...],
        prompt: str,
    ) -> list[Sam3Prediction]:
        predictions = self.provider.predict(
            image, prompt.strip() or "object", threshold=self.threshold
        )
        if not predictions:
            return []

        def rank(prediction: Sam3Prediction) -> tuple[float, float]:
            mask = np.asarray(prediction.mask, dtype=bool)
            height, width = mask.shape[:2]
            score = 0.0
            for point in points:
                x = min(max(int(point.x), 0), width - 1)
                y = min(max(int(point.y), 0), height - 1)
                inside = bool(mask[y, x])
                if point.label == "positive":
                    score += 1.0 if inside else -0.25
                else:
                    score += -2.0 if inside else 0.1
            return score, float(prediction.score)

        return [max(predictions, key=rank)]


@dataclass
class _Session:
    session_id: UUID
    image_id: str
    image: Image.Image
    points: list[ClickPoint]
    revision: int
    predictions: list[Sam3Prediction]
    tenant_id: UUID | None
    version_id: UUID | None
    prompt: str
    category_id: int


class ClickRefinementService:
    def __init__(self, provider: ClickRefinementProvider) -> None:
        self.provider = provider
        self._sessions: dict[UUID, _Session] = {}
        self._lock = threading.RLock()

    def create(
        self,
        *,
        image_id: str,
        image: Image.Image,
        tenant_id: UUID | None = None,
        version_id: UUID | None = None,
        prompt: str = "",
        category_id: int = 0,
    ) -> RefinementSnapshot:
        if not image_id.strip():
            raise ValueError("image_id must not be empty")
        if category_id < 0:
            raise ValueError("category_id must not be negative")
        session = _Session(
            session_id=uuid4(),
            image_id=image_id,
            image=image.convert("RGB"),
            points=[],
            revision=0,
            predictions=[],
            tenant_id=tenant_id,
            version_id=version_id,
            prompt=prompt.strip(),
            category_id=category_id,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return self._snapshot(session)

    def click(
        self,
        session_id: UUID,
        *,
        expected_revision: int,
        point: ClickPoint,
    ) -> RefinementSnapshot:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"unknown refinement session: {session_id}")
            if session.revision != expected_revision:
                raise ValueError(
                    f"expected revision {expected_revision}, current revision is {session.revision}"
                )
            point.validate(session.image)
            next_points = tuple((*session.points, point))
            prompt_aware = getattr(self.provider, "refine_with_prompt", None)
            if callable(prompt_aware):
                predictions = prompt_aware(session.image, next_points, session.prompt)
            else:
                predictions = self.provider.refine(session.image, next_points)
            session.points.append(point)
            session.predictions = list(predictions)
            session.revision += 1
            return self._snapshot(session)

    def get(self, session_id: UUID) -> RefinementSnapshot:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"unknown refinement session: {session_id}")
            return self._snapshot(session)

    @staticmethod
    def _snapshot(session: _Session) -> RefinementSnapshot:
        return RefinementSnapshot(
            session_id=session.session_id,
            image_id=session.image_id,
            revision=session.revision,
            points=tuple(session.points),
            predictions=tuple(session.predictions),
            tenant_id=session.tenant_id,
            version_id=session.version_id,
            prompt=session.prompt,
            category_id=session.category_id,
        )
