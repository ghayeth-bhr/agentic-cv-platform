from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from agentic_cv.sam3 import Sam3Prediction
from agentic_cv.sam3_service import Sam3ServiceConfig, create_app


class FakeProvider:
    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        assert image.mode == "RGB"
        return [Sam3Prediction(
            score=0.91,
            box=(1, 2, 10, 12),
            mask=np.array([[True, False], [False, True]]),
        )]


def image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (2, 2), (1, 2, 3, 4)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_and_prediction_contract():
    client = TestClient(create_app(FakeProvider()))

    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").json() == {"status": "ready"}
    response = client.post(
        "/v1/predict",
        files={"image": ("frame.png", image_bytes(), "image/png")},
        data={"prompt": "brown bear", "threshold": "0.6"},
    )

    assert response.status_code == 200
    assert response.json()["predictions"][0]["box"] == [1.0, 2.0, 10.0, 12.0]
    assert response.json()["predictions"][0]["mask"] == [[True, False], [False, True]]


def test_prediction_rejects_invalid_and_oversized_images():
    client = TestClient(create_app(FakeProvider(), Sam3ServiceConfig(max_image_bytes=100)))
    invalid = client.post(
        "/v1/predict",
        files={"image": ("frame.bin", b"not an image", "application/octet-stream")},
        data={"prompt": "bear"},
    )
    limited_client = TestClient(create_app(FakeProvider(), Sam3ServiceConfig(max_image_bytes=4)))
    oversized = limited_client.post(
        "/v1/predict",
        files={"image": ("frame.bin", b"12345", "application/octet-stream")},
        data={"prompt": "bear"},
    )

    assert invalid.status_code == 400
    assert oversized.status_code == 413
