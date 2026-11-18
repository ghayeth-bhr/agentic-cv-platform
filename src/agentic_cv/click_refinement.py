"""Revisioned interactive point-click refinement sessions."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID, uuid4

from PIL import Image

from .sam3 import Sam3Prediction

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
class RefinementSnapshot:
    session_id: UUID
    image_id: str
    revision: int
    points: tuple[ClickPoint, ...]
    predictions: tuple[Sam3Prediction, ...]


class ClickRefinementProvider(Protocol):
    def refine(self, image: Image.Image, points: tuple[ClickPoint, ...]) -> list[Sam3Prediction]: ...


@dataclass
class _Session:
    session_id: UUID
    image_id: str
    image: Image.Image
    points: list[ClickPoint]
    revision: int
    predictions: list[Sam3Prediction]


class ClickRefinementService:
    def __init__(self, provider: ClickRefinementProvider) -> None:
        self.provider = provider
        self._sessions: dict[UUID, _Session] = {}
        self._lock = threading.RLock()

    def create(self, *, image_id: str, image: Image.Image) -> RefinementSnapshot:
        if not image_id.strip():
            raise ValueError("image_id must not be empty")
        session = _Session(
            session_id=uuid4(), image_id=image_id, image=image.convert("RGB"),
            points=[], revision=0, predictions=[],
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
            session_id=session.session_id, image_id=session.image_id,
            revision=session.revision, points=tuple(session.points),
            predictions=tuple(session.predictions),
        )
