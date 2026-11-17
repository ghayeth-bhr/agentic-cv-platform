"""Provider-neutral local semantic image index with exact cosine search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True, kw_only=True)
class ImageRecord:
    record_id: str
    asset_id: str
    image_path: Path


@dataclass(frozen=True, kw_only=True)
class SearchResult:
    record_id: str
    asset_id: str
    image_path: Path
    score: float


class EmbeddingProvider(Protocol):
    def embed_image(self, image: Image.Image) -> Sequence[float]: ...

    def embed_text(self, query: str) -> Sequence[float]: ...


def _normalize(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("embedding must be a non-empty one-dimensional vector")
    norm = np.linalg.norm(array)
    if norm == 0:
        raise ValueError("embedding must not be a zero vector")
    return array / norm


class LocalVectorIndex:
    """Small exact-search index for local mode; replaceable by pgvector/Qdrant later."""

    def __init__(self) -> None:
        self._records: dict[str, tuple[ImageRecord, np.ndarray]] = {}
        self._dimension: int | None = None

    def upsert(self, record: ImageRecord, embedding: Sequence[float]) -> None:
        vector = _normalize(embedding)
        if self._dimension is None:
            self._dimension = vector.size
        if vector.size != self._dimension:
            raise ValueError("embedding dimension does not match the index")
        self._records[record.record_id] = (record, vector)

    def remove(self, record_id: str) -> None:
        self._records.pop(record_id, None)

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        asset_ids: set[str] | None = None,
        limit: int = 20,
    ) -> list[SearchResult]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        query = _normalize(query_embedding)
        if self._dimension is not None and query.size != self._dimension:
            raise ValueError("query embedding dimension does not match the index")
        scored = []
        for record, embedding in self._records.values():
            if asset_ids is not None and record.asset_id not in asset_ids:
                continue
            scored.append(
                SearchResult(
                    record_id=record.record_id,
                    asset_id=record.asset_id,
                    image_path=record.image_path,
                    score=float(np.dot(query, embedding)),
                )
            )
        return sorted(scored, key=lambda result: (-result.score, result.record_id))[:limit]


class SemanticSearchService:
    def __init__(self, provider: EmbeddingProvider, index: LocalVectorIndex) -> None:
        self.provider = provider
        self.index = index

    def index_images(self, records: list[ImageRecord]) -> None:
        for record in records:
            with Image.open(record.image_path) as image:
                embedding = self.provider.embed_image(image.convert("RGB"))
            self.index.upsert(record, embedding)

    def search(
        self,
        query: str,
        *,
        asset_ids: set[str] | None = None,
        limit: int = 20,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        return self.index.search(self.provider.embed_text(query), asset_ids=asset_ids, limit=limit)
