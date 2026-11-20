"""Index asset sources (images and extracted video frames) for semantic search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname
from uuid import UUID

from .asset_registry import AssetRegistryBackend, SourceRecord
from .semantic_search import ImageRecord, SemanticSearchService
from .video_frames import extract_frames_cached, frames_dir_for


def path_from_file_uri(uri: str) -> Path | None:
    try:
        parsed = urlparse(uri)
    except Exception:
        return None
    if parsed.scheme != "file":
        return None
    return Path(url2pathname(unquote(parsed.path)))


@dataclass(frozen=True, kw_only=True)
class AssetIndexResult:
    asset_id: UUID
    indexed_records: int
    sources_indexed: int
    detail: str


class AssetSemanticIndexer:
    def __init__(
        self,
        *,
        registry: AssetRegistryBackend,
        search: SemanticSearchService,
        data_dir: Path,
    ) -> None:
        self.registry = registry
        self.search = search
        self.data_dir = data_dir

    def index_asset(self, *, tenant_id: UUID, asset_id: UUID) -> AssetIndexResult:
        summaries = self.registry.list_assets(tenant_id=tenant_id)
        summary = next((item for item in summaries if item.asset_id == asset_id), None)
        if summary is None:
            raise KeyError(f"unknown asset: {asset_id}")
        if not summary.sources:
            return AssetIndexResult(asset_id=asset_id, indexed_records=0, sources_indexed=0, detail="no sources to index")

        records: list[ImageRecord] = []
        sources_indexed = 0
        for source in summary.sources:
            source_records = self._records_for_source(asset_id=asset_id, source=source)
            if source_records:
                records.extend(source_records)
                sources_indexed += 1
        if not records:
            return AssetIndexResult(asset_id=asset_id, indexed_records=0, sources_indexed=0, detail="no indexable image/video sources")

        self.search.index_images(records)
        return AssetIndexResult(
            asset_id=asset_id,
            indexed_records=len(records),
            sources_indexed=sources_indexed,
            detail=f"indexed {len(records)} frame(s)/image(s) from {sources_indexed} source(s)",
        )

    def _records_for_source(self, *, asset_id: UUID, source: SourceRecord) -> list[ImageRecord]:
        path = path_from_file_uri(source.object_uri)
        if path is None or not path.exists():
            return []
        media_type = (source.media_type or "").lower()
        asset_key = str(asset_id)
        source_key = str(source.source_id)
        if media_type.startswith("image/"):
            return [ImageRecord(record_id=f"{asset_key}:{source_key}:image", asset_id=asset_key, image_path=path)]
        if media_type.startswith("video/"):
            frames = extract_frames_cached(
                path, frames_dir_for(self.data_dir, asset_key, source_key)
            )
            return [
                ImageRecord(
                    record_id=f"{asset_key}:{source_key}:frame:{frame.frame_index}",
                    asset_id=asset_key,
                    image_path=frame.image_path,
                )
                for frame in frames
            ]
        return []
