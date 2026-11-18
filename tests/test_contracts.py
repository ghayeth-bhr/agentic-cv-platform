import json
from uuid import uuid4

import pytest

from agentic_cv.contracts import ArtifactManifest, EventEnvelope, TaskSpec


def test_task_spec_round_trip():
    spec = TaskSpec(
        target_objects=("brown bear",),
        task_type="segmentation",
        visual_attributes=("full body",),
        sample_size=7,
        max_iterations=3,
    )

    assert TaskSpec.from_dict(spec.to_dict()) == spec


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_objects": (), "task_type": "detection"},
        {"target_objects": ("bear",), "task_type": "classification"},
        {"target_objects": ("bear",), "task_type": "detection", "sample_size": 0},
    ],
)
def test_task_spec_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TaskSpec(**kwargs)


def test_artifact_manifest_hashes_content():
    tenant_id = uuid4()
    asset_id = uuid4()
    manifest = ArtifactManifest.from_bytes(
        tenant_id=tenant_id,
        asset_id=asset_id,
        kind="frame",
        storage_uri="s3://bucket/frame.jpg",
        content=b"frame-bytes",
    )

    value = manifest.to_dict()

    assert value["tenant_id"] == str(tenant_id)
    assert value["asset_id"] == str(asset_id)
    assert len(value["content_sha256"]) == 64


def test_event_envelope_serializes_idempotency_metadata():
    envelope = EventEnvelope(
        event_type="MediaVerified",
        tenant_id=uuid4(),
        aggregate_type="source",
        aggregate_id=uuid4(),
        payload={"content_sha256": "abc"},
        idempotency_key="verify:source-1:v1",
    )

    value = json.loads(envelope.to_json())

    assert value["event_type"] == "MediaVerified"
    assert value["idempotency_key"] == "verify:source-1:v1"
    assert value["schema_version"] == 1
