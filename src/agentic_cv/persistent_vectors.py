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

    def remove(self, record_id: str) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM vector_records WHERE record_id = ?", (record_id,))
            self._connection.commit()

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
        if limit < 1:
            raise ValueError("limit must be at least 1")
        query = _normalize(query_embedding)
        with self._lock:
            rows = self._connection.execute("SELECT * FROM vector_records").fetchall()
        results: list[SearchResult] = []
        for row in rows:
            if asset_ids is not None and row["asset_id"] not in asset_ids:
                continue
            if row["dimension"] != query.size:
                raise ValueError("query embedding dimension does not match the index")
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            results.append(
                SearchResult(
                    record_id=row["record_id"], asset_id=row["asset_id"],
                    image_path=Path(row["image_path"]), score=float(np.dot(query, vector)),
                )
            )
        return sorted(results, key=lambda result: (-result.score, result.record_id))[:limit]
