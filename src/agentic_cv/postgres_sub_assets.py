"""PostgreSQL implementation of the durable sub-asset store.

Mirrors :class:`agentic_cv.sub_assets.SubAssetStore`. SQLite remains the
default; this store is opt-in via ``AGENTIC_CV_POSTGRES_DSN``.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import UUID, uuid4

from .postgres import PostgresConfig, apply_migrations, connect
from .semantic_search import SearchResult
from .sub_assets import SubAssetRecord


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def open_postgres_sub_asset_store(config: PostgresConfig) -> "PostgresSubAssetStore":
    from psycopg.rows import dict_row

    connection = connect(config, row_factory=dict_row)
    try:
        return PostgresSubAssetStore(connection, schema=config.schema)
    except Exception:
        connection.close()
        raise


class PostgresSubAssetStore:
    def __init__(self, connection: Any, *, schema: str = "public") -> None:
        self._connection = connection
        self._lock = threading.RLock()
        apply_migrations(connection, schema=schema)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create_from_results(
        self,
        *,
        tenant_id: UUID,
        query: str,
        results: Iterable[SearchResult],
    ) -> SubAssetRecord:
        if not query.strip():
            raise ValueError("query must not be empty")
        ordered_results = tuple(results)
        if not ordered_results:
            raise ValueError("at least one search result is required")
        result_ids = tuple(result.record_id for result in ordered_results)
        if len(set(result_ids)) != len(result_ids):
            raise ValueError("search result IDs must be unique")
        parent_asset_ids = tuple(dict.fromkeys(result.asset_id for result in ordered_results))
        record = SubAssetRecord(
            sub_asset_id=uuid4(),
            tenant_id=tenant_id,
            parent_asset_ids=parent_asset_ids,
            query=query,
            result_ids=result_ids,
            created_at=_iso(_utc_now()),
        )
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO sub_assets
                    (sub_asset_id, tenant_id, parent_asset_ids, query, result_ids,
                     created_at, asset_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    record.sub_asset_id,
                    record.tenant_id,
                    list(record.parent_asset_ids),
                    record.query,
                    list(record.result_ids),
                    _utc_now(),
                    None,
                ),
            )
            self._connection.commit()
        return record

    def get(self, sub_asset_id: UUID) -> SubAssetRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM sub_assets WHERE sub_asset_id = %s", (sub_asset_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown sub-asset: {sub_asset_id}")
        return self._record(row)

    def get_for_tenant(self, tenant_id: UUID, sub_asset_id: UUID) -> SubAssetRecord:
        record = self.get(sub_asset_id)
        if record.tenant_id != tenant_id:
            raise PermissionError("sub-asset does not belong to tenant")
        return record

    def link_asset(self, sub_asset_id: UUID, *, asset_id: UUID) -> SubAssetRecord:
        with self._lock:
            record = self.get(sub_asset_id)
            self._connection.execute(
                "UPDATE sub_assets SET asset_id = %s WHERE sub_asset_id = %s",
                (asset_id, sub_asset_id),
            )
            self._connection.commit()
        return SubAssetRecord(
            sub_asset_id=record.sub_asset_id,
            tenant_id=record.tenant_id,
            parent_asset_ids=record.parent_asset_ids,
            query=record.query,
            result_ids=record.result_ids,
            created_at=record.created_at,
            asset_id=asset_id,
        )

    def list_for_tenant(self, tenant_id: UUID) -> list[SubAssetRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM sub_assets WHERE tenant_id = %s ORDER BY created_at",
                (tenant_id,),
            ).fetchall()
        return [self._record(row) for row in rows]

    def find_by_asset(self, tenant_id: UUID, asset_id: UUID) -> SubAssetRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM sub_assets WHERE tenant_id = %s AND asset_id = %s",
                (tenant_id, asset_id),
            ).fetchone()
        return self._record(row) if row is not None else None

    @staticmethod
    def _record(row: Any) -> SubAssetRecord:
        parent = row["parent_asset_ids"]
        result_ids = row["result_ids"]
        if isinstance(parent, str):
            parent = json.loads(parent)
        if isinstance(result_ids, str):
            result_ids = json.loads(result_ids)
        asset_id = row["asset_id"]
        return SubAssetRecord(
            sub_asset_id=UUID(str(row["sub_asset_id"])),
            tenant_id=UUID(str(row["tenant_id"])),
            parent_asset_ids=tuple(parent),
            query=row["query"],
            result_ids=tuple(result_ids),
            created_at=_iso(row["created_at"]),
            asset_id=UUID(str(asset_id)) if asset_id else None,
        )
