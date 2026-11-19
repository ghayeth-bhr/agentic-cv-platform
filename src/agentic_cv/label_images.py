"""Resolve asset sources into the image list consumed by batch labeling."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from .asset_registry import AssetRegistryBackend
from .batch_labeling import LabelImage
from .semantic_indexing import path_from_file_uri
from .video_frames import extract_frames_cached, frames_dir_for


_IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


class LabelImageResolver:
    """Turn one tenant asset into deterministic image/frame records.

    Image IDs are stable across runs so annotation revisions and dataset builds
    refer to the same source even after a worker restart.
    """

    def __init__(
        self,
        *,
        registry: AssetRegistryBackend,
        data_dir: Path,
    ) -> None:
        self.registry = registry
        self.data_dir = data_dir

    def resolve(self, *, tenant_id: UUID, asset_id: str) -> list[LabelImage]:
        try:
            parsed_asset_id = UUID(asset_id)
        except ValueError as exc:
            raise ValueError(f"asset_id must be a UUID: {asset_id}") from exc

        summary = next(
            (
                item
                for item in self.registry.list_assets(tenant_id=tenant_id)
                if item.asset_id == parsed_asset_id
            ),
            None,
        )
        if summary is None:
            raise KeyError(f"unknown asset: {asset_id}")
        if not summary.sources:
            raise ValueError("asset has no uploaded sources")

        images: list[LabelImage] = []
        for source in summary.sources:
            path = path_from_file_uri(source.object_uri)
            if path is None:
                raise ValueError(f"source is not a local file URI: {source.source_id}")
            if not path.exists():
                raise ValueError(f"source file is missing: {path}")
            media_type = (source.media_type or "").lower()
            asset_key = str(summary.asset_id)
            source_key = str(source.source_id)
            if media_type.startswith("image/"):
                suffix = path.suffix.lower() or _IMAGE_EXTENSIONS.get(media_type, ".img")
                images.append(
                    LabelImage(
                        image_id=f"{asset_key}:{source_key}:image",
                        path=path,
                        file_name=f"{source_key}{suffix}",
                    )
                )
                continue
            if media_type.startswith("video/"):
                frames = extract_frames_cached(
                    path, frames_dir_for(self.data_dir, asset_key, source_key)
                )
                if not frames:
                    raise ValueError(f"video source produced no frames: {path}")
                images.extend(
                    LabelImage(
                        image_id=f"{asset_key}:{source_key}:frame:{frame.frame_index}",
                        path=frame.image_path,
                        file_name=f"{source_key}_frame_{frame.frame_index:08d}.jpg",
                    )
                    for frame in frames
                )
                continue
            raise ValueError(
                f"unsupported source media type for labeling: {source.media_type or 'unknown'}"
            )

        if not images:
            raise ValueError("asset has no indexable image or video sources")
        return images

    def resolve_many(self, *, tenant_id: UUID, asset_ids: list[str] | tuple[str, ...]) -> list[LabelImage]:
        """Images of several assets in order, without duplicates."""

        seen: set[str] = set()
        images: list[LabelImage] = []
        for asset_id in asset_ids:
            for image in self.resolve(tenant_id=tenant_id, asset_id=str(asset_id)):
                if image.image_id not in seen:
                    seen.add(image.image_id)
                    images.append(image)
        return images
