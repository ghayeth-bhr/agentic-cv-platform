"""PostgreSQL/pgvector-backed semantic vector index.

Mirrors :class:`agentic_cv.semantic_search.LocalVectorIndex` semantics (exact
cosine similarity over normalized vectors) but persists vectors in PostgreSQL
using the pgvector extension. SQLite/local in-memory index remains the default;
this index is opt-in via ``AGENTIC_CV_POSTGRES_DSN``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID

import numpy as np

from .postgres import PostgresConfig, apply_migrations, connect
from .semantic_search import ImageRecord, SearchResult, _normalize


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(format(float(value), ".17g") for value in vector.tolist()) + "]"


def open_postgres_vector_index(config: PostgresConfig) -> "PgVectorIndex":
    from psycopg.rows import dict_row

    connection = connect(config, row_factory=dict_row)
    try:
        return PgVectorIndex(connection, schema=config.schema)
    except Exception:
        connection.close()
        raise


class PgVectorIndex:
    """Durable exact-search index backed by pgvector cosine distance."""

    def __init__(self, connection: Any, *, schema: str = "public") -> None:
        self._connection = connection
        self._lock = threading.RLock()
        apply_migrations(connection, schema=schema)
        row = self._connection.execute(
            "SELECT dimension FROM vector_records LIMIT 1"
        ).fetchone()
        self._dimension: int | None = int(row["dimension"]) if row else None

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def upsert(self, record: ImageRecord, embedding: Sequence[float]) -> None:
        vector = _normalize(embedding)
        with self._lock:
            if self._dimension is None:
                self._dimension = int(vector.size)
            if vector.size != self._dimension:
                raise ValueError("embedding dimension does not match the index")
            try:
                self._connection.execute(
                    """
                    INSERT INTO vector_records (record_id, asset_id, image_path, dimension, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT (record_id) DO UPDATE SET
                        asset_id = EXCLUDED.asset_id,
                        image_path = EXCLUDED.image_path,
                        dimension = EXCLUDED.dimension,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        record.record_id,
                        str(record.asset_id),
                        str(record.image_path),
                        int(vector.size),
                        _vector_literal(vector),
                    ),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def remove(self, record_id: str) -> None:
        with self._lock:
            try:
                self._connection.execute(
                    "DELETE FROM vector_records WHERE record_id = %s", (record_id,)
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

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
            if self._dimension is not None and query.size != self._dimension:
                raise ValueError("query embedding dimension does not match the index")
            literal = _vector_literal(query)
            params: list[Any] = [literal]
            where = ""
            if asset_ids is not None:
                where = " WHERE asset_id::text = ANY(%s)"
                params.append(list(asset_ids))
            params.extend([literal, limit])
            rows = self._connection.execute(
                f"""
                SELECT record_id,
                       asset_id::text AS asset_id,
                       image_path,
                       1 - (embedding <=> %s::vector) AS score
                FROM vector_records
                {where}
                ORDER BY embedding <=> %s::vector, record_id
                LIMIT %s
                """,
                params,
            ).fetchall()
        return [
            SearchResult(
                record_id=row["record_id"],
                asset_id=str(row["asset_id"]),
                image_path=Path(row["image_path"]),
                score=float(row["score"]),
            )
            for row in rows
        ]
