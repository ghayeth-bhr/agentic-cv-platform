import hashlib
from io import BytesIO
from uuid import uuid4

import pytest

from agentic_cv.object_storage import S3ObjectStorage


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, *, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = Body

    def list_objects_v2(self, *, Bucket, Prefix):
        return {
            "Contents": [
                {"Key": key} for bucket, key in self.objects
                if bucket == Bucket and key.startswith(Prefix)
            ]
        }

    def get_object(self, *, Bucket, Key):
        return {"Body": BytesIO(self.objects[(Bucket, Key)])}

    def delete_object(self, *, Bucket, Key):
        del self.objects[(Bucket, Key)]


def test_s3_storage_assembles_parts_and_returns_minio_uri():
    client = FakeS3()
    storage = S3ObjectStorage(bucket="agentic", client=client, endpoint_url="http://minio:9000")
    upload_id = uuid4()
    content = b"first-second"
    storage.write_part(upload_id, 1, b"first-")
    storage.write_part(upload_id, 2, b"second")

    result = storage.complete(
        upload_id, object_key="tenant/frame.jpg", expected_sha256=hashlib.sha256(content).hexdigest()
    )

    assert result == "http://minio:9000/agentic/tenant/frame.jpg"
    assert client.objects[("agentic", "tenant/frame.jpg")] == content
    assert not any(key.startswith(f".uploads/{upload_id}/") for _, key in client.objects)


def test_s3_storage_rejects_checksum_mismatch_without_final_object():
    client = FakeS3()
    storage = S3ObjectStorage(bucket="agentic", client=client)
    upload_id = uuid4()
    storage.write_part(upload_id, 1, b"content")

    with pytest.raises(ValueError, match="checksum"):
        storage.complete(upload_id, object_key="frame.jpg", expected_sha256="0" * 64)
    assert ("agentic", "frame.jpg") not in client.objects
