"""PostgreSQL implementation of the asset registry contract.

Mirrors :class:`agentic_cv.asset_registry.AssetRegistry` method for method so the
upload service can switch backends without changes. SQLite remains the default;
this store is only constructed when ``AGENTIC_CV_POSTGRES_DSN`` is set.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from .asset_registry import AssetRecord, SourceRecord, UploadRecord
from .postgres import PostgresConfig, apply_migrations, connect


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def open_postgres_asset_registry(config: PostgresConfig) -> "PostgresAssetRegistry":
    """Connect, apply migrations, and return a registry scoped to the config schema."""
    from psycopg.rows import dict_row

    connection = connect(config, row_factory=dict_row)
    try:
        return PostgresAssetRegistry(connection, schema=config.schema)
    except Exception:
        connection.close()
        raise


class PostgresAssetRegistry:
    def __init__(self, connection: Any, *, schema: str = "public") -> None:
        self._connection = connection
        self._lock = threading.RLock()
        apply_migrations(connection, schema=schema)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create_asset(self, *, tenant_id: UUID, name: str) -> AssetRecord:
        if not name.strip():
            raise ValueError("asset name must not be empty")
        record = AssetRecord(
            asset_id=uuid4(), tenant_id=tenant_id, name=name, created_at=_utc_now()
        )
        with self._lock:
            self._connection.execute(
                "INSERT INTO assets (asset_id, tenant_id, name, created_at) VALUES (%s, %s, %s, %s)",
                (record.asset_id, record.tenant_id, record.name, record.created_at),
            )
            self._connection.commit()
        return record

    def start_upload(
        self,
        *,
        tenant_id: UUID,
        asset_id: UUID,
        filename: str,
        media_type: str,
        expected_sha256: str,
    ) -> UploadRecord:
        if not filename.strip() or not media_type.strip() or len(expected_sha256) != 64:
            raise ValueError("filename, media_type, and a SHA-256 digest are required")
        with self._lock:
            asset = self._connection.execute(
                "SELECT tenant_id FROM assets WHERE asset_id = %s", (asset_id,)
            ).fetchone()
            if asset is None:
                raise KeyError(f"unknown asset: {asset_id}")
            if asset["tenant_id"] != tenant_id:
                raise PermissionError("asset does not belong to tenant")
            record = UploadRecord(
                upload_id=uuid4(),
                tenant_id=tenant_id,
                asset_id=asset_id,
                filename=filename,
                media_type=media_type,
                expected_sha256=expected_sha256,
                object_key=f"tenants/{tenant_id}/assets/{asset_id}/uploads/{uuid4()}/{filename}",
                status="active",
                created_at=_utc_now(),
            )
            self._connection.execute(
                """
                INSERT INTO uploads (
                    upload_id, tenant_id, asset_id, filename, media_type,
                    expected_sha256, object_key, status, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    record.upload_id, record.tenant_id, record.asset_id, record.filename,
                    record.media_type, record.expected_sha256, record.object_key,
                    record.status, record.created_at,
                ),
            )
            self._connection.commit()
            return record

    def get_upload(self, upload_id: UUID) -> UploadRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM uploads WHERE upload_id = %s", (upload_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown upload: {upload_id}")
        return self._upload(row)

    def complete_upload(self, upload_id: UUID, *, object_uri: str) -> SourceRecord:
        with self._lock:
            upload = self.get_upload(upload_id)
            if upload.status == "completed":
                row = self._connection.execute(
                    "SELECT * FROM sources WHERE asset_id = %s AND content_sha256 = %s",
                    (upload.asset_id, upload.expected_sha256),
                ).fetchone()
                if row is not None:
                    return self._source(row)
            if upload.status != "active":
                raise ValueError(f"upload is not active: {upload.status}")
            existing = self._connection.execute(
                "SELECT * FROM sources WHERE asset_id = %s AND content_sha256 = %s",
                (upload.asset_id, upload.expected_sha256),
            ).fetchone()
            if existing is not None:
                self._connection.execute(
                    "UPDATE uploads SET status = 'completed' WHERE upload_id = %s",
                    (upload_id,),
                )
                self._connection.commit()
                return self._source(existing)
            source = SourceRecord(
                source_id=uuid4(),
                tenant_id=upload.tenant_id,
                asset_id=upload.asset_id,
                filename=upload.filename,
                media_type=upload.media_type,
                object_uri=object_uri,
                content_sha256=upload.expected_sha256,
                created_at=_utc_now(),
            )
            self._connection.execute(
                """
                INSERT INTO sources (
                    source_id, tenant_id, asset_id, filename, media_type,
                    object_uri, content_sha256, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    source.source_id, source.tenant_id, source.asset_id, source.filename,
                    source.media_type, source.object_uri, source.content_sha256,
                    source.created_at,
                ),
            )
            self._connection.execute(
                "UPDATE uploads SET status = 'completed' WHERE upload_id = %s", (upload_id,)
            )
            self._connection.commit()
            return source

    @staticmethod
    def _upload(row: Any) -> UploadRecord:
        return UploadRecord(
            upload_id=UUID(str(row["upload_id"])),
            tenant_id=UUID(str(row["tenant_id"])),
            asset_id=UUID(str(row["asset_id"])),
            filename=row["filename"],
            media_type=row["media_type"],
            expected_sha256=row["expected_sha256"],
            object_key=row["object_key"],
            status=row["status"],
            created_at=_iso(row["created_at"]),
        )

    @staticmethod
    def _source(row: Any) -> SourceRecord:
        return SourceRecord(
            source_id=UUID(str(row["source_id"])),
            tenant_id=UUID(str(row["tenant_id"])),
            asset_id=UUID(str(row["asset_id"])),
            filename=row["filename"],
            media_type=row["media_type"],
            object_uri=row["object_uri"],
            content_sha256=row["content_sha256"],
            created_at=_iso(row["created_at"]),
        )
