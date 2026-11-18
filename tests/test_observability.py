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

import json
import logging

from agentic_cv.observability import JsonLogFormatter, configure_logging


def _cleanup_agentic_cv_logger():
    logger = logging.getLogger("agentic_cv")
    logger.handlers = [h for h in logger.handlers if not getattr(h, "_agentic_cv_logging", False)]
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def test_json_log_formatter_includes_structured_fields():
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="agentic_cv.http",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request completed",
        args=(),
        exc_info=None,
    )
    record.method = "GET"
    record.path = "/health/ready"
    record.status_code = 200
    record.duration_ms = 1.23

    payload = json.loads(formatter.format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "agentic_cv.http"
    assert payload["message"] == "request completed"
    assert payload["method"] == "GET"
    assert payload["path"] == "/health/ready"
    assert payload["status_code"] == 200
    assert payload["duration_ms"] == 1.23
    assert "timestamp" in payload


def test_configure_logging_scopes_agentic_cv_logger_and_supports_json(monkeypatch):
    monkeypatch.delenv("AGENTIC_CV_LOG_LEVEL", raising=False)
    monkeypatch.delenv("AGENTIC_CV_LOG_FORMAT", raising=False)
    try:
        logger = configure_logging(level="DEBUG", fmt="json")
        assert logger.name == "agentic_cv"
        assert logger.level == logging.DEBUG
        assert logger.propagate is False
        handlers = [h for h in logger.handlers if getattr(h, "_agentic_cv_logging", False)]
        assert len(handlers) == 1
        assert isinstance(handlers[0].formatter, JsonLogFormatter)

        # repeated configuration should not stack handlers
        configure_logging(level="INFO", fmt="json")
        handlers = [h for h in logger.handlers if getattr(h, "_agentic_cv_logging", False)]
        assert len(handlers) == 1
        assert logger.level == logging.INFO
    finally:
        _cleanup_agentic_cv_logger()


def test_request_middleware_logs_completion(tmp_path, caplog):
    app = create_local_app(LocalAppConfig(data_dir=tmp_path))
    with caplog.at_level(logging.INFO, logger="agentic_cv.http"):
        with TestClient(app) as client:
            response = client.get("/health/ready")
            assert response.status_code == 200

    records = [r for r in caplog.records if r.name == "agentic_cv.http" and r.message == "request completed"]
    assert records, "expected request completion log"
    record = records[-1]
    assert getattr(record, "method", None) == "GET"
    assert getattr(record, "path", None) == "/health/ready"
    assert getattr(record, "status_code", None) == 200
    assert isinstance(getattr(record, "duration_ms", None), (int, float))
