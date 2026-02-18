"""FastAPI application exposing OpenAI-compatible chat completion and model endpoints."""

from __future__ import annotations

import time
import uuid

from dcc_backend_common.logger import get_logger
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from bentoml_ocr.ocr_proxy.container import Container
from bentoml_ocr.ocr_proxy.models import (
    MODEL_NAME,
    ChatCompletionRequest,
    ModelCard,
    ModelListResponse,
    OCRBackend,
)

logger = get_logger(__name__)

app = FastAPI(title="GLM-OCR Docling-Compatible Proxy")


# ---------------------------------------------------------------------------
# Middleware: X-Request-ID correlation
# ---------------------------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Middleware: Request body size limit
# ---------------------------------------------------------------------------


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        max_bytes: int = request.app.state.max_body_size_bytes
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body too large. Maximum allowed size is {max_bytes} bytes."},
            )
        return await call_next(request)


# Register middleware once at import time (outermost first).
# Configuration is read from app.state at request time.
app.add_middleware(BodySizeLimitMiddleware)  # type: ignore[arg-type]
app.add_middleware(RequestIDMiddleware)  # type: ignore[arg-type]

# Defaults -- overridden by configure_app_state() during service startup.
app.state.max_body_size_bytes = 50 * 1024 * 1024


def configure_app_state(max_body_size: int = 50 * 1024 * 1024) -> None:
    """Set runtime configuration on the app. Called once during service init."""
    app.state.max_body_size_bytes = max_body_size


# ---------------------------------------------------------------------------
# Health / readiness endpoints
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
@inject
async def readyz(
    backend: OCRBackend = Depends(Provide[Container.backend]),  # noqa: B008
) -> dict[str, str]:
    """Readiness check that verifies the vLLM backend is reachable."""
    _ = backend
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """List all models supported by this proxy."""
    return ModelListResponse(
        data=[ModelCard(id=MODEL_NAME)],
    )


@app.post("/v1/chat/completions")
@inject
async def chat_completions(
    request: ChatCompletionRequest,
    backend: OCRBackend = Depends(Provide[Container.backend]),  # noqa: B008
) -> dict[str, object]:
    """Process an OpenAI-compatible chat completion request through the OCR pipeline."""
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported by this proxy.")

    if request.model != MODEL_NAME:
        logger.warning("Unsupported model requested", model=request.model)
        detail = f"Unsupported model '{request.model}'. Supported: {MODEL_NAME}"
        raise HTTPException(status_code=400, detail=detail)

    start_time = time.perf_counter()
    response = await backend.process_full(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info("Full OCR pipeline completed", model=request.model, duration_ms=round(duration_ms, 2))
    return response.model_dump()
