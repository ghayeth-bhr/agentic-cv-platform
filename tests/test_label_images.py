import hashlib
from uuid import uuid4

import pytest
from PIL import Image

from agentic_cv.asset_registry import AssetRegistry
from agentic_cv.label_images import LabelImageResolver


def add_image_source(registry, *, tenant_id, asset_id, path, media_type="image/png"):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return registry.add_source(
        tenant_id=tenant_id,
        asset_id=asset_id,
        filename=path.name,
        media_type=media_type,
        object_uri=path.resolve().as_uri(),
        content_sha256=digest,
    )


def test_resolver_turns_asset_image_into_stable_label_image(tmp_path):
    registry = AssetRegistry(str(tmp_path / "assets.sqlite3"))
    tenant_id = uuid4()
    asset = registry.create_asset(tenant_id=tenant_id, name="photos")
    path = tmp_path / "frame.png"
    Image.new("RGB", (8, 6), (1, 2, 3)).save(path)
    source = add_image_source(
        registry, tenant_id=tenant_id, asset_id=asset.asset_id, path=path
    )
    resolver = LabelImageResolver(registry=registry, data_dir=tmp_path / "data")

    images = resolver.resolve(tenant_id=tenant_id, asset_id=str(asset.asset_id))

    assert len(images) == 1
    assert images[0].image_id == f"{asset.asset_id}:{source.source_id}:image"
    assert images[0].path == path.resolve()
    assert images[0].file_name == f"{source.source_id}.png"
    registry.close()


def test_resolver_rejects_unknown_asset_and_empty_asset(tmp_path):
    registry = AssetRegistry(str(tmp_path / "assets.sqlite3"))
    tenant_id = uuid4()
    asset = registry.create_asset(tenant_id=tenant_id, name="empty")
    resolver = LabelImageResolver(registry=registry, data_dir=tmp_path / "data")

    with pytest.raises(KeyError, match="unknown asset"):
        resolver.resolve(tenant_id=tenant_id, asset_id=str(uuid4()))
    with pytest.raises(ValueError, match="no uploaded sources"):
        resolver.resolve(tenant_id=tenant_id, asset_id=str(asset.asset_id))
    registry.close()
