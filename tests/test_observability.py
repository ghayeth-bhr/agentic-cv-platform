from fastapi.testclient import TestClient

from agentic_cv.application import LocalAppConfig, create_local_app
from agentic_cv.observability import collect_gpu_metrics, collect_metrics


class _FakeProps:
    def __init__(self, name="Fake GPU", total_memory=8 * 1024**3):
        self.name = name
        self.total_memory = total_memory


class _FakeCuda:
    def __init__(self, available=True):
        self._available = available

    def is_available(self):
        return self._available

    def device_count(self):
        return 1

    def get_device_properties(self, index):
        assert index == 0
        return _FakeProps()

    def memory_allocated(self, index):
        return 1024

    def memory_reserved(self, index):
        return 2048

    def current_device(self):
        return 0


class _FakeTorch:
    def __init__(self, cuda):
        self.cuda = cuda


def test_collect_gpu_metrics_without_torch():
    metrics = collect_gpu_metrics(torch_module=None)
    # In environments where torch is installed, this path is not forced; test the
    # unavailable branch explicitly by passing a module without cuda.
    class NoCuda:
        pass

    unavailable = collect_gpu_metrics(torch_module=NoCuda())
    assert unavailable["available"] is False
    assert "reason" in unavailable
    assert "available" in metrics


def test_collect_gpu_metrics_with_cuda():
    torch = _FakeTorch(_FakeCuda(available=True))
    metrics = collect_gpu_metrics(torch_module=torch)

    assert metrics["available"] is True
    assert metrics["device_count"] == 1
    assert metrics["current_device"] == 0
    assert metrics["devices"][0]["name"] == "Fake GPU"
    assert metrics["devices"][0]["memory_bytes"]["allocated"] == 1024
    assert metrics["devices"][0]["memory_bytes"]["reserved"] == 2048


def test_collect_metrics_shape():
    payload = collect_metrics(torch_module=_FakeTorch(_FakeCuda(available=False)))

    assert payload["status"] == "ok"
    assert "python_version" in payload["runtime"]
    assert payload["gpu"]["available"] is False


def test_metrics_endpoint_and_api_info(tmp_path):
    app = create_local_app(LocalAppConfig(data_dir=tmp_path))
    with TestClient(app) as client:
        info = client.get("/api-info")
        assert info.status_code == 200
        assert info.json()["observability"]["metrics"] == "/v1/metrics"

        response = client.get("/v1/metrics")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "runtime" in body
        assert "gpu" in body
        assert isinstance(body["gpu"].get("available"), bool)
