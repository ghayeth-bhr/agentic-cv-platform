import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from agentic_cv.api import ApiDependencies, RefinementApiDependencies, create_app
from agentic_cv.asset_registry import AssetRegistry
from agentic_cv.click_refinement import ClickRefinementService
from agentic_cv.object_storage import LocalObjectStorage
from agentic_cv.outbox import Outbox
from agentic_cv.sam3 import Sam3Prediction
from agentic_cv.uploads import ResumableUploadService


class FakeClickProvider:
    def refine(self, image, points):
        return [Sam3Prediction(score=0.9, box=(1, 1, 3, 3), mask=np.ones((4, 4), dtype=bool))]


def make_client(tmp_path):
    registry = AssetRegistry(str(tmp_path / "metadata.sqlite3"))
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    refinement = RefinementApiDependencies(
        service=ClickRefinementService(FakeClickProvider()),
        image_loader=lambda image_id: Image.new("RGB", (4, 4)),
    )
    dependencies = ApiDependencies(
        registry=registry,
        uploads=ResumableUploadService(registry, LocalObjectStorage(tmp_path / "objects")),
        outbox=outbox,
        refinement=refinement,
    )
    return TestClient(create_app(dependencies)), registry, outbox


def test_refinement_websocket_replays_snapshots_and_clicks(tmp_path):
    client, registry, outbox = make_client(tmp_path)
    with client.websocket_connect("/v1/refinement/frame-1") as socket:
        initial = socket.receive_json()
        assert initial["type"] == "snapshot"
        assert initial["revision"] == 0
        socket.send_json({
            "type": "click",
            "expected_revision": 0,
            "x": 2,
            "y": 1,
            "label": "positive",
        })
        updated = socket.receive_json()

    assert updated["revision"] == 1
    assert updated["points"] == [{"x": 2, "y": 1, "label": "positive"}]
    assert updated["predictions"][0]["mask"] == [[True] * 4 for _ in range(4)]
    registry.close()
    outbox.close()


def test_refinement_websocket_reports_stale_revision(tmp_path):
    client, registry, outbox = make_client(tmp_path)
    with client.websocket_connect("/v1/refinement/frame-1") as socket:
        socket.receive_json()
        socket.send_json({"type": "click", "expected_revision": 4, "x": 1, "y": 1, "label": "positive"})
        error = socket.receive_json()

    assert error == {
        "type": "error",
        "detail": "expected revision 4, current revision is 0",
    }
    registry.close()
    outbox.close()