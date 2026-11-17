"""Filesystem implementation of resumable object storage for local mode."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from uuid import UUID


class ObjectStorage(Protocol):
    def write_part(self, upload_id: UUID, part_number: int, content: bytes) -> None: ...

    def complete(
        self,
        upload_id: UUID,
        *,
        object_key: str,
        expected_sha256: str,
    ) -> Path | str: ...


class LocalObjectStorage:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.parts_root = self.root / ".uploads"
        self.objects_root = self.root / "objects"
        self.parts_root.mkdir(parents=True, exist_ok=True)
        self.objects_root.mkdir(parents=True, exist_ok=True)

    def write_part(self, upload_id: UUID, part_number: int, content: bytes) -> None:
        if part_number < 1:
            raise ValueError("part_number must be at least 1")
        destination = self.parts_root / str(upload_id) / f"{part_number:08d}.part"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(content)
        os.replace(temporary, destination)

    def list_parts(self, upload_id: UUID) -> tuple[int, ...]:
        directory = self.parts_root / str(upload_id)
        if not directory.exists():
            return ()
        return tuple(sorted(int(path.stem) for path in directory.glob("*.part")))

    def complete(
        self,
        upload_id: UUID,
        *,
        object_key: str,
        expected_sha256: str,
    ) -> Path:
        self._validate_key(object_key)
        parts = self.list_parts(upload_id)
        if not parts or parts != tuple(range(1, len(parts) + 1)):
            raise ValueError("upload parts must be contiguous starting at part 1")
        destination = self.objects_root / Path(*PurePosixPath(object_key).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        digest = hashlib.sha256()
        with temporary.open("wb") as output:
            for part_number in parts:
                content = (self.parts_root / str(upload_id) / f"{part_number:08d}.part").read_bytes()
                digest.update(content)
                output.write(content)
        if digest.hexdigest() != expected_sha256:
            temporary.unlink(missing_ok=True)
            raise ValueError("assembled upload checksum does not match expected_sha256")
        os.replace(temporary, destination)
        return destination

    @staticmethod
    def _validate_key(object_key: str) -> None:
        path = PurePosixPath(object_key)
        if path.is_absolute() or ".." in path.parts or not object_key.strip():
            raise ValueError("object_key must be a non-empty relative path")


class S3ObjectStorage:
    """S3/MinIO adapter using temporary objects for resumable upload parts."""

    def __init__(self, *, bucket: str, client: Any, endpoint_url: str | None = None) -> None:
        if not bucket.strip():
            raise ValueError("bucket must not be empty")
        self.bucket = bucket
        self.client = client
        self.endpoint_url = endpoint_url.rstrip("/") if endpoint_url else None

    @classmethod
    def from_environment(
        cls,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        region_name: str | None = None,
    ) -> "S3ObjectStorage":
        import boto3

        return cls(
            bucket=bucket,
            client=boto3.client("s3", endpoint_url=endpoint_url, region_name=region_name),
            endpoint_url=endpoint_url,
        )

    def write_part(self, upload_id: UUID, part_number: int, content: bytes) -> None:
        if part_number < 1:
            raise ValueError("part_number must be at least 1")
        self.client.put_object(
            Bucket=self.bucket, Key=self._part_key(upload_id, part_number), Body=content
        )

    def complete(
        self,
        upload_id: UUID,
        *,
        object_key: str,
        expected_sha256: str,
    ) -> str:
        LocalObjectStorage._validate_key(object_key)
        prefix = f".uploads/{upload_id}/"
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        keys = sorted(
            item["Key"] for item in response.get("Contents", []) if item["Key"].startswith(prefix)
        )
        expected_keys = [self._part_key(upload_id, index) for index in range(1, len(keys) + 1)]
        if not keys or keys != expected_keys:
            raise ValueError("upload parts must be contiguous starting at part 1")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        for key in keys:
            content = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            digest.update(content)
            chunks.append(content)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("assembled upload checksum does not match expected_sha256")
        self.client.put_object(Bucket=self.bucket, Key=object_key, Body=b"".join(chunks))
        for key in keys:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        return self._object_uri(object_key)

    @staticmethod
    def _part_key(upload_id: UUID, part_number: int) -> str:
        return f".uploads/{upload_id}/{part_number:08d}.part"

    def _object_uri(self, object_key: str) -> str:
        if self.endpoint_url:
            return f"{self.endpoint_url}/{self.bucket}/{object_key}"
        return f"s3://{self.bucket}/{object_key}"