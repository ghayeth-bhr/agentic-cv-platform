"""PostgreSQL implementation of the tenant-scoped dataset version store.

Mirrors :class:`agentic_cv.dataset_store.DatasetVersionStore`. SQLite remains
the default; this store is opt-in via ``AGENTIC_CV_POSTGRES_DSN``.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from .dataset_store import DatasetVersionRecord, _dataset_dict, _dataset_from_dict
from .dataset import DatasetVersion
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


def open_postgres_dataset_version_store(config: PostgresConfig) -> "PostgresDatasetVersionStore":
    from psycopg.rows import dict_row

    connection = connect(config, row_factory=dict_row)
    try:
        return PostgresDatasetVersionStore(connection, schema=config.schema)
    except Exception:
        connection.close()
        raise


class PostgresDatasetVersionStore:
    def __init__(self, connection: Any, *, schema: str = "public") -> None:
        self._connection = connection
        self._lock = threading.RLock()
        apply_migrations(connection, schema=schema)

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
            row = self._connection.execute(
                """
                SELECT * FROM dataset_versions
                WHERE tenant_id = %s AND idempotency_key = %s
                """,
                (tenant_id, idempotency_key),
            ).fetchone()
            if row is not None:
                existing = self._record(row)
                existing_encoded = json.dumps(_dataset_dict(existing.dataset), sort_keys=True)
                if existing.name != name or existing_encoded != encoded:
                    raise ValueError("idempotency key already identifies a different version")
                return existing
            version_id = uuid4()
            created_at = _utc_now()
            self._connection.execute(
                """
                INSERT INTO dataset_versions (
                    version_id, tenant_id, name, dataset, idempotency_key, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    version_id,
                    tenant_id,
                    name,
                    _jsonb(_dataset_dict(dataset)),
                    idempotency_key,
                    created_at,
                ),
            )
            self._connection.commit()
            return self.get(tenant_id=tenant_id, version_id=version_id)

    def get(self, *, tenant_id: UUID, version_id: UUID) -> DatasetVersionRecord:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM dataset_versions
                WHERE tenant_id = %s AND version_id = %s
                """,
                (tenant_id, version_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown dataset version: {version_id}")
        return self._record(row)

    def list(self, *, tenant_id: UUID) -> list[DatasetVersionRecord]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM dataset_versions
                WHERE tenant_id = %s
                ORDER BY created_at, sequence
                """,
                (tenant_id,),
            ).fetchall()
        return [self._record(row) for row in rows]

    @staticmethod
    def _record(row: Any) -> DatasetVersionRecord:
        dataset_value = row["dataset"]
        if not isinstance(dataset_value, dict):
            dataset_value = json.loads(dataset_value)
        return DatasetVersionRecord(
            version_id=UUID(str(row["version_id"])),
            tenant_id=UUID(str(row["tenant_id"])),
            name=row["name"],
            dataset=_dataset_from_dict(dataset_value),
            idempotency_key=row["idempotency_key"],
            created_at=_iso(row["created_at"]),
        )
