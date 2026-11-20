"""Bounded HTTP boundary for local SAM 3 inference."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .errors import install_model_unavailable_handler
from .sam3 import Sam3Prediction, Sam3Provider


@dataclass(frozen=True, kw_only=True)
class Sam3ServiceConfig:
    max_image_bytes: int = 20 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.max_image_bytes < 1:
            raise ValueError("max_image_bytes must be positive")


class PredictionResponse(BaseModel):
    score: float
    box: tuple[float, float, float, float]
    mask: list[list[bool]]


class PredictResponse(BaseModel):
    prompt: str
    predictions: list[PredictionResponse]


def _serialize_prediction(prediction: Sam3Prediction) -> PredictionResponse:
    return PredictionResponse(
        score=prediction.score,
        box=prediction.box,
        mask=prediction.mask.astype(bool).tolist(),
    )


def create_app(
    provider: Sam3Provider,
    config: Sam3ServiceConfig | None = None,
) -> FastAPI:
    service_config = config or Sam3ServiceConfig()
    app = FastAPI(title="BotFlow SAM3 service", version="1")
    install_model_unavailable_handler(app)

    @app.get("/health/live")
    def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def readiness() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/v1/predict", response_model=PredictResponse)
    async def predict(
        image: UploadFile = File(...),
        prompt: str = Form(...),
        threshold: float | None = Form(default=None),
        mask_threshold: float | None = Form(default=None),
    ) -> PredictResponse:
        content = await image.read(service_config.max_image_bytes + 1)
        if len(content) > service_config.max_image_bytes:
            raise HTTPException(status_code=413, detail="image exceeds configured size limit")
        try:
            source_image = Image.open(BytesIO(content)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail="uploaded file is not a valid image") from exc
        try:
            predictions = provider.predict(
                source_image,
                prompt,
                threshold=threshold,
                mask_threshold=mask_threshold,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictResponse(
            prompt=prompt,
            predictions=[_serialize_prediction(item) for item in predictions],
        )

    return app
