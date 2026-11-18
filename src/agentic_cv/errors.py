"""Typed runtime errors that are safe to surface at the HTTP boundary."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ModelUnavailableError(RuntimeError):
    """Raised when an optional local model runtime is unavailable.

    This covers missing ML dependencies, a missing CUDA device, gated model
    weights, and model-load failures. API handlers map it to HTTP 503 with an
    actionable message instead of leaking a 500 traceback.
    """


MODEL_UNAVAILABLE_CODE = "model_unavailable"


def install_model_unavailable_handler(app: FastAPI) -> None:
    """Register the 503 response used for :class:`ModelUnavailableError`."""

    @app.exception_handler(ModelUnavailableError)
    async def _model_unavailable(_request: Request, exc: ModelUnavailableError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "code": MODEL_UNAVAILABLE_CODE},
        )
