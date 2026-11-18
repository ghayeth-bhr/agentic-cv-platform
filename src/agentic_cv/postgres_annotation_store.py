"""PostgreSQL implementation of immutable annotation revisions.

Mirrors :class:`agentic_cv.annotation_store.AnnotationRevisionStore` with the
same optimistic-concurrency contract. SQLite remains the default; this store is
opt-in via ``AGENTIC_CV_POSTGRES_DSN``.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from .annotation_store import AnnotationRevision, RevisionConflictError
from .dataset import Annotation
from .postgres import PostgresConfig, apply_migrations, connect


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _jsonb(value: Any) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(value)


def _annotation_dict(annotation: Annotation) -> dict:
    return {
        "annotation_id": annotation.annotation_id,
        "image_id": annotation.image_id,
        "category_id": annotation.category_id,
        "bbox": list(annotation.bbox),
        "segmentation": [list(polygon) for polygon in annotation.segmentation],
        "score": annotation.score,
    }


def open_postgres_annotation_store(config: PostgresConfig) -> "PostgresAnnotationRevisionStore":
    from psycopg.rows import dict_row

    connection = connect(config, row_factory=dict_row)
    try:
        return PostgresAnnotationRevisionStore(connection, schema=config.schema)
    except Exception:
        connection.close()
        raise


class PostgresAnnotationRevisionStore:
    def __init__(self, connection: Any, *, schema: str = "public") -> None:
        self._connection = connection
        self._lock = threading.RLock()
        apply_migrations(connection, schema=schema)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def save(
        self,
        *,
        tenant_id: UUID,
        annotation: Annotation,
        author_id: str,
        expected_revision: int | None = 0,
    ) -> AnnotationRevision:
        if not author_id.strip():
            raise ValueError("author_id must not be empty")
        if expected_revision is not None and expected_revision < 0:
            raise ValueError("expected_revision must not be negative")
        from psycopg.errors import UniqueViolation

        with self._lock:
            row = self._connection.execute(
                """
                SELECT revision FROM annotation_revisions
                WHERE tenant_id = %s AND image_id = %s AND annotation_id = %s
                ORDER BY revision DESC LIMIT 1
                """,
                (tenant_id, annotation.image_id, annotation.annotation_id),
            ).fetchone()
            current_revision = int(row["revision"]) if row is not None else 0
            if expected_revision is not None and current_revision != expected_revision:
                raise RevisionConflictError(
                    f"expected revision {expected_revision}, current revision is {current_revision}"
                )
            revision = AnnotationRevision(
                revision_id=uuid4(),
                tenant_id=tenant_id,
                image_id=annotation.image_id,
                annotation=annotation,
                revision=current_revision + 1,
                author_id=author_id,
                created_at=_iso(_utc_now()),
            )
            try:
                self._connection.execute(
                    """
                    INSERT INTO annotation_revisions (
                        revision_id, tenant_id, image_id, annotation_id, revision,
                        annotation, author_id, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        revision.revision_id,
                        revision.tenant_id,
                        revision.image_id,
                        revision.annotation.annotation_id,
                        revision.revision,
                        _jsonb(_annotation_dict(revision.annotation)),
                        revision.author_id,
                        _utc_now(),
                    ),
                )
                self._connection.commit()
            except UniqueViolation:
                self._connection.rollback()
                raise RevisionConflictError(
                    f"annotation revision conflict for {annotation.annotation_id}"
                )
            return revision

    def latest(
        self, *, tenant_id: UUID, image_id: str, annotation_id: str
    ) -> AnnotationRevision:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM annotation_revisions
                WHERE tenant_id = %s AND image_id = %s AND annotation_id = %s
                ORDER BY revision DESC LIMIT 1
                """,
                (tenant_id, image_id, annotation_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown annotation: {annotation_id}")
        return self._record(row)

    def list_latest(
        self,
        *,
        tenant_id: UUID,
        image_ids: list[str] | None = None,
    ) -> list[AnnotationRevision]:
        if image_ids is not None and not image_ids:
            return []
        image_clause = " AND image_id = ANY(%s)" if image_ids is not None else ""
        values: tuple[Any, ...] = (tenant_id,)
        if image_ids is not None:
            values = (tenant_id, image_ids)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT DISTINCT ON (image_id, annotation_id) *
                FROM annotation_revisions
                WHERE tenant_id = %s{image_clause}
                ORDER BY image_id, annotation_id, revision DESC
                """,
                values,
            ).fetchall()
        return [self._record(row) for row in rows]

    def history(
        self, *, tenant_id: UUID, image_id: str, annotation_id: str
    ) -> list[AnnotationRevision]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM annotation_revisions
                WHERE tenant_id = %s AND image_id = %s AND annotation_id = %s
                ORDER BY revision
                """,
                (tenant_id, image_id, annotation_id),
            ).fetchall()
        return [self._record(row) for row in rows]

    @staticmethod
    def _record(row: Any) -> AnnotationRevision:
        value = row["annotation"]
        if not isinstance(value, dict):
            value = json.loads(value)
        annotation = Annotation(
            annotation_id=value["annotation_id"],
            image_id=value["image_id"],
            category_id=value["category_id"],
            bbox=tuple(value["bbox"]),
            segmentation=tuple(tuple(polygon) for polygon in value["segmentation"]),
            score=value["score"],
        )
        return AnnotationRevision(
            revision_id=UUID(str(row["revision_id"])),
            tenant_id=UUID(str(row["tenant_id"])),
            image_id=row["image_id"],
            annotation=annotation,
            revision=int(row["revision"]),
            author_id=row["author_id"],
            created_at=_iso(row["created_at"]),
        )
