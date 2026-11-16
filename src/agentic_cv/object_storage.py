"""Filesystem implementation of resumable object storage for local mode."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from uuid import UUID


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