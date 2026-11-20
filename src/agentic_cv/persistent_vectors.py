"""SQLite-backed exact vector index for durable local semantic search."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

from .semantic_search import ImageRecord, SearchResult, _normalize
from .sqlite_utils import connect_sqlite


class SqliteVectorIndex:
    def __init__(self, path: str) -> None:
        self._connection = connect_sqlite(path)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_records (
                record_id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                embedding BLOB NOT NULL
            )
            """
        )
        self._connection.commit()
        self._cache = None
        self._cache_key = None
        self._writes = 0

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def upsert(self, record: ImageRecord, embedding: Sequence[float]) -> None:
        vector = _normalize(embedding)
        with self._lock:
            existing = self._connection.execute(
                "SELECT dimension FROM vector_records LIMIT 1"
            ).fetchone()
            if existing is not None and existing["dimension"] != vector.size:
                raise ValueError("embedding dimension does not match the index")
            self._connection.execute(
                """
                INSERT INTO vector_records (record_id, asset_id, image_path, dimension, embedding)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(record_id) DO UPDATE SET
                    asset_id = excluded.asset_id,
                    image_path = excluded.image_path,
                    dimension = excluded.dimension,
                    embedding = excluded.embedding
                """,
                (
                    record.record_id, record.asset_id, str(record.image_path), vector.size,
                    sqlite3.Binary(vector.tobytes()),
                ),
            )
            self._connection.commit()
            self._writes += 1

    def remove(self, record_id: str) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM vector_records WHERE record_id = ?", (record_id,))
            self._connection.commit()
            self._writes += 1

    def get_record(self, record_id: str) -> ImageRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT record_id, asset_id, image_path FROM vector_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        return ImageRecord(record_id=row["record_id"], asset_id=row["asset_id"], image_path=Path(row["image_path"]))

    def get_embedding(self, record_id: str) -> np.ndarray | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT dimension, embedding FROM vector_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row["embedding"], dtype=np.float32).copy()

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        asset_ids: set[str] | None = None,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Exact cosine search over an in-memory matrix of every vector.

        The matrix is rebuilt only when the index changes: on this
        connection's writes, and when another process commits (SQLite's
        ``data_version``). Measured: ~160 ms per query at 20k vectors when
        every row was read and scored in Python on each search.
        """

        if limit < 1:
            raise ValueError("limit must be at least 1")
        query = _normalize(query_embedding)
        cache = self._matrix()
        if cache is None:
            return []
        record_ids, assets, paths, matrix = cache
        if matrix.shape[1] != query.size:
            raise ValueError("query embedding dimension does not match the index")
        if asset_ids is not None:
            rows = np.flatnonzero(np.isin(assets, list(asset_ids)))
            if rows.size == 0:
                return []
            scores = matrix[rows] @ query
        else:
            rows = None
            scores = matrix @ query
        k = min(limit, scores.size)
        if k < scores.size:
            top = np.argpartition(-scores, k - 1)[:k]
            # Keep ties at the cut so ordering by record id stays exact.
            candidates = np.flatnonzero(scores >= scores[top].min())
        else:
            candidates = np.arange(scores.size)
        results = []
        for position in candidates:
            row = int(rows[position]) if rows is not None else int(position)
            results.append(
                SearchResult(
                    record_id=record_ids[row],
                    asset_id=str(assets[row]),
                    image_path=Path(paths[row]),
                    score=float(scores[position]),
                )
            )
        return sorted(results, key=lambda result: (-result.score, result.record_id))[:limit]

    def _matrix(self):
        with self._lock:
            version = self._connection.execute("PRAGMA data_version").fetchone()[0]
            key = (version, self._writes)
            if self._cache is not None and self._cache_key == key:
                return self._cache
            rows = self._connection.execute(
                "SELECT record_id, asset_id, image_path, dimension, embedding FROM vector_records "
                "ORDER BY record_id"
            ).fetchall()
            if not rows:
                self._cache, self._cache_key = None, key
                return None
            dimensions = {row["dimension"] for row in rows}
            if len(dimensions) != 1:
                raise ValueError("the index holds embeddings of different dimensions")
            dimension = dimensions.pop()
            matrix = np.frombuffer(
                b"".join(row["embedding"] for row in rows), dtype=np.float32
            ).reshape(len(rows), dimension)
            self._cache = (
                [row["record_id"] for row in rows],
                np.array([row["asset_id"] for row in rows], dtype=object),
                [row["image_path"] for row in rows],
                matrix,
            )
            self._cache_key = key
            return self._cache
