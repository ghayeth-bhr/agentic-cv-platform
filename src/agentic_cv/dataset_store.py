"""Tenant-scoped SQLite persistence for immutable dataset versions."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import UUID, uuid4

from .dataset import Annotation, DatasetCategory, DatasetImage, DatasetVersion
from .sqlite_utils import connect_sqlite


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, kw_only=True)
class DatasetVersionRecord:
    version_id: UUID
    tenant_id: UUID
    name: str
    dataset: DatasetVersion
    idempotency_key: str
    created_at: str


class DatasetVersionBackend(Protocol):
    def close(self) -> None: ...

    def create(
        self,
        *,
        tenant_id: UUID,
        name: str,
        dataset: DatasetVersion,
        idempotency_key: str,
    ) -> DatasetVersionRecord: ...

    def get(self, *, tenant_id: UUID, version_id: UUID) -> DatasetVersionRecord: ...

    def list(self, *, tenant_id: UUID) -> list[DatasetVersionRecord]: ...


class DatasetVersionStore:
    def __init__(self, path: str) -> None:
        self._connection = connect_sqlite(path)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_versions (
                version_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL,
                tenant_id TEXT NOT NULL,
                name TEXT NOT NULL,
                dataset TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (tenant_id, idempotency_key)
            )
            """
        )
        columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(dataset_versions)")
        }
        if "sequence" not in columns:
            self._connection.execute(
                "ALTER TABLE dataset_versions ADD COLUMN sequence INTEGER"
            )
            self._connection.execute("UPDATE dataset_versions SET sequence = rowid")
        self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create(
        self,
        *,
        tenant_id: UUID,
        name: str,
        dataset: DatasetVersion,
        idempotency_key: str,
    ) -> DatasetVersionRecord:
        if not name.strip() or not idempotency_key.strip():
            raise ValueError("name and idempotency_key must not be empty")
        encoded = json.dumps(_dataset_dict(dataset), sort_keys=True)
        with self._lock:
            existing = self._connection.execute(
                "SELECT * FROM dataset_versions WHERE tenant_id = ? AND idempotency_key = ?",
                (str(tenant_id), idempotency_key),
            ).fetchone()
            if existing is not None:
                if existing["name"] != name or existing["dataset"] != encoded:
                    raise ValueError("idempotency key already identifies a different version")
                return self._record(existing)
            version_id = uuid4()
            sequence = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM dataset_versions"
            ).fetchone()[0]
            self._connection.execute(
                """
                INSERT INTO dataset_versions (
                    version_id, sequence, tenant_id, name, dataset, idempotency_key,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(version_id), sequence, str(tenant_id), name, encoded,
                    idempotency_key, _utc_now(),
                ),
            )
            self._connection.commit()
            return self.get(tenant_id=tenant_id, version_id=version_id)

    def get(self, *, tenant_id: UUID, version_id: UUID) -> DatasetVersionRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM dataset_versions WHERE tenant_id = ? AND version_id = ?",
                (str(tenant_id), str(version_id)),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown dataset version: {version_id}")
        return self._record(row)

    def list(self, *, tenant_id: UUID) -> list[DatasetVersionRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM dataset_versions WHERE tenant_id = ? ORDER BY created_at, sequence",
                (str(tenant_id),),
            ).fetchall()
        return [self._record(row) for row in rows]

    @classmethod
    def _record(cls, row: sqlite3.Row) -> DatasetVersionRecord:
        return DatasetVersionRecord(
            version_id=UUID(row["version_id"]), tenant_id=UUID(row["tenant_id"]),
            name=row["name"], dataset=_dataset_from_dict(json.loads(row["dataset"])),
            idempotency_key=row["idempotency_key"], created_at=row["created_at"],
        )


def _dataset_dict(dataset: DatasetVersion) -> dict:
    value = {
        "images": [
            {"image_id": image.image_id, "file_name": image.file_name,
             "width": image.width, "height": image.height,
             **({"split": image.split} if image.split is not None else {})}
            for image in dataset.images
        ],
        "categories": [
            {"category_id": category.category_id, "name": category.name}
            for category in dataset.categories
        ],
        "annotations": [
            {"annotation_id": annotation.annotation_id, "image_id": annotation.image_id,
             "category_id": annotation.category_id, "bbox": list(annotation.bbox),
             "segmentation": [list(polygon) for polygon in annotation.segmentation],
             "score": annotation.score}
            for annotation in dataset.annotations
        ],
    }
    # Optional keys are omitted when unset so versions created before they
    # existed keep the same encoding (idempotency compares encodings).
    if dataset.annotation_set is not None:
        value["annotation_set"] = dataset.annotation_set
    if dataset.source_asset_ids:
        value["source_asset_ids"] = list(dataset.source_asset_ids)
    return value


def _dataset_from_dict(value: dict) -> DatasetVersion:
    return DatasetVersion(
        images=tuple(DatasetImage(**image) for image in value["images"]),
        categories=tuple(DatasetCategory(**category) for category in value["categories"]),
        annotations=tuple(
            Annotation(
                annotation_id=annotation["annotation_id"], image_id=annotation["image_id"],
                category_id=annotation["category_id"], bbox=tuple(annotation["bbox"]),
                segmentation=tuple(tuple(polygon) for polygon in annotation["segmentation"]),
                score=annotation["score"],
            )
            for annotation in value["annotations"]
        ),
        annotation_set=value.get("annotation_set"),
        source_asset_ids=tuple(value.get("source_asset_ids", ())),
    )