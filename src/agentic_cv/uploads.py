"""Application service joining resumable storage to asset metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from .asset_registry import AssetRegistry, SourceRecord, UploadRecord
from .object_storage import ObjectStorage


@dataclass(frozen=True, kw_only=True)
class CompletedUpload:
    source: SourceRecord
    object_path: Path | str


class ResumableUploadService:
    def __init__(self, registry: AssetRegistry, storage: ObjectStorage) -> None:
        self.registry = registry
        self.storage = storage

    def start(
        self,
        *,
        tenant_id: UUID,
        asset_id: UUID,
        filename: str,
        media_type: str,
        expected_sha256: str,
    ) -> UploadRecord:
        return self.registry.start_upload(
            tenant_id=tenant_id, asset_id=asset_id, filename=filename,
            media_type=media_type, expected_sha256=expected_sha256,
        )

    def put_part(self, upload_id: UUID, part_number: int, content: bytes) -> None:
        upload = self.registry.get_upload(upload_id)
        if upload.status != "active":
            raise ValueError(f"upload is not active: {upload.status}")
        self.storage.write_part(upload_id, part_number, content)

    def complete(self, upload_id: UUID) -> CompletedUpload:
        upload = self.registry.get_upload(upload_id)
        object_path = self.storage.complete(
            upload_id, object_key=upload.object_key, expected_sha256=upload.expected_sha256
        )
        object_uri = (
            object_path.resolve().as_uri() if isinstance(object_path, Path) else object_path
        )
        source = self.registry.complete_upload(upload_id, object_uri=object_uri)
        return CompletedUpload(source=source, object_path=object_path)