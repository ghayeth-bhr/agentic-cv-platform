import hashlib
from uuid import uuid4

import pytest

from agentic_cv.asset_registry import AssetRegistry
from agentic_cv.object_storage import LocalObjectStorage
from agentic_cv.uploads import ResumableUploadService


def make_service(tmp_path):
    registry = AssetRegistry(tmp_path / "metadata.sqlite3")
    storage = LocalObjectStorage(tmp_path / "objects")
    return registry, storage, ResumableUploadService(registry, storage)


def test_upload_can_resume_out_of_order_and_register_source(tmp_path):
    registry, storage, service = make_service(tmp_path)
    tenant_id = uuid4()
    asset = registry.create_asset(tenant_id=tenant_id, name="Wildlife")
    content = b"part-one|part-two|part-three"
    upload = service.start(
        tenant_id=tenant_id,
        asset_id=asset.asset_id,
        filename="clip.mp4",
        media_type="video/mp4",
        expected_sha256=hashlib.sha256(content).hexdigest(),
    )

    service.put_part(upload.upload_id, 2, b"part-two|")
    service.put_part(upload.upload_id, 1, b"part-one|")
    service.put_part(upload.upload_id, 3, b"part-three")
    result = service.complete(upload.upload_id)

    assert result.object_path.read_bytes() == content
    assert result.source.asset_id == asset.asset_id
    assert registry.get_upload(upload.upload_id).status == "completed"
    registry.close()


def test_upload_rejects_missing_part_and_bad_checksum(tmp_path):
    registry, storage, service = make_service(tmp_path)
    asset = registry.create_asset(tenant_id=uuid4(), name="Images")
    upload = service.start(
        tenant_id=asset.tenant_id,
        asset_id=asset.asset_id,
        filename="frame.png",
        media_type="image/png",
        expected_sha256="a" * 64,
    )
    service.put_part(upload.upload_id, 2, b"only-part-two")

    with pytest.raises(ValueError, match="contiguous"):
        service.complete(upload.upload_id)

    service.put_part(upload.upload_id, 1, b"part-one")
    with pytest.raises(ValueError, match="checksum"):
        service.complete(upload.upload_id)
    registry.close()


def test_upload_enforces_tenant_ownership(tmp_path):
    registry, storage, service = make_service(tmp_path)
    asset = registry.create_asset(tenant_id=uuid4(), name="Private")

    with pytest.raises(PermissionError):
        service.start(
            tenant_id=uuid4(), asset_id=asset.asset_id, filename="x.jpg",
            media_type="image/jpeg", expected_sha256="b" * 64,
        )
    registry.close()