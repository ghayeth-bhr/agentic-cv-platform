import hashlib
import sys
import types
from io import BytesIO
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from agentic_cv.api import ApiDependencies, create_app
from agentic_cv.asset_registry import AssetRegistry
from agentic_cv.embeddings import HuggingFaceEmbeddingProvider
from agentic_cv.errors import ModelUnavailableError, install_model_unavailable_handler
from agentic_cv.object_storage import LocalObjectStorage
from agentic_cv.outbox import Outbox
from agentic_cv.sam3 import LocalSam3Provider, Sam3Config
from agentic_cv.semantic_api import SemanticApiDependencies, create_app as create_semantic_app
from agentic_cv.semantic_search import LocalVectorIndex, SemanticSearchService
from agentic_cv.uploads import ResumableUploadService


class UnavailableEmbeddingProvider:
    def embed_image(self, image):
        raise ModelUnavailableError("install the ml extra")

    def embed_text(self, query):
        raise ModelUnavailableError("install the ml extra")


class UnavailableSam3Provider:
    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        raise ModelUnavailableError("install the ml extra")


def test_exception_handler_returns_503_with_code():
    app = FastAPI()
    install_model_unavailable_handler(app)

    @app.get("/boom")
    def boom():
        raise ModelUnavailableError("install the ml extra")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/boom")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "install the ml extra",
        "code": "model_unavailable",
    }


def test_search_returns_503_instead_of_500_when_embeddings_are_missing():
    app = create_semantic_app(
        SemanticApiDependencies(
            search=SemanticSearchService(UnavailableEmbeddingProvider(), LocalVectorIndex()),
            sub_assets=None,
        )
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/search", json={"query": "bear"})

    assert response.status_code == 503
    assert response.json()["code"] == "model_unavailable"


def test_asset_predict_returns_503_instead_of_500_when_model_is_missing(tmp_path):
    registry = AssetRegistry(str(tmp_path / "metadata.sqlite3"))
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    uploads = ResumableUploadService(registry, LocalObjectStorage(tmp_path / "objects"))
    app = create_app(
        ApiDependencies(
            registry=registry,
            uploads=uploads,
            outbox=outbox,
            sam3=UnavailableSam3Provider(),
        )
    )
    tenant_id = uuid4()
    headers = {"X-Tenant-ID": str(tenant_id)}
    image = BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(image, format="PNG")
    payload = image.getvalue()
    digest = hashlib.sha256(payload).hexdigest()

    with TestClient(app, raise_server_exceptions=False) as client:
        asset = client.post("/v1/assets", headers=headers, json={"name": "test"}).json()
        upload = client.post(
            f"/v1/assets/{asset['asset_id']}/uploads",
            headers=headers,
            json={"filename": "frame.png", "media_type": "image/png", "expected_sha256": digest},
        ).json()
        client.put(
            f"/v1/uploads/{upload['upload_id']}/parts/1",
            files={"part": ("frame.png", BytesIO(payload), "image/png")},
        )
        client.post(f"/v1/uploads/{upload['upload_id']}/complete")
        response = client.post(
            f"/v1/assets/{asset['asset_id']}/predict",
            headers=headers,
            json={"prompt": "bear"},
        )

    registry.close()
    outbox.close()
    assert response.status_code == 503
    assert response.json()["code"] == "model_unavailable"


def test_embedding_provider_reports_missing_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ModelUnavailableError, match="uv sync --extra ml"):
        HuggingFaceEmbeddingProvider().embed_text("bear")


def test_sam3_provider_reports_missing_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ModelUnavailableError, match="uv sync --extra ml"):
        LocalSam3Provider().predict(Image.new("RGB", (2, 2)), "bear")


def test_sam3_provider_reports_cuda_unavailable(monkeypatch):
    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    transformers = types.ModuleType("transformers")
    transformers.Sam3Model = object
    transformers.Sam3Processor = object

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    with pytest.raises(ModelUnavailableError, match="AGENTIC_CV_DEVICE=cpu"):
        LocalSam3Provider().predict(Image.new("RGB", (2, 2)), "bear")


def test_sam3_provider_can_load_on_cpu(monkeypatch):
    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    transformers = types.ModuleType("transformers")
    transformers.Sam3Model = FakeModel
    transformers.Sam3Processor = FakeProcessor

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    provider = LocalSam3Provider(Sam3Config(device="cpu", precision="float32"))
    provider._load()

    assert provider._model is not None
    assert provider._model.device == "cpu"
