"""FastAPI application exposing OpenAI-compatible chat completion and model endpoints."""

from __future__ import annotations

import time
from typing import Any

from dcc_backend_common.logger import get_logger
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, FastAPI, HTTPException

from bentoml_ocr.ocr_proxy.backend import extract_image_data_uri
from bentoml_ocr.ocr_proxy.constants import FULL_MODEL_NAME, RAW_MODEL_NAME
from bentoml_ocr.ocr_proxy.container import Container
from bentoml_ocr.ocr_proxy.models import (
    ChatCompletionRequest,
    ModelCard,
    ModelListResponse,
    OCRBackend,
)

logger = get_logger(__name__)

app = FastAPI(title="GLM-OCR Docling-Compatible Proxy")


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """List all models supported by this proxy."""
    return ModelListResponse(
        data=[ModelCard(id=FULL_MODEL_NAME), ModelCard(id=RAW_MODEL_NAME)],
    )


@app.post("/v1/chat/completions")
@inject
async def chat_completions(
    request: ChatCompletionRequest,
    backend: OCRBackend = Depends(Provide[Container.backend]),  # noqa: B008
) -> dict[str, Any]:
    """Process an OpenAI-compatible chat completion request through the OCR pipeline."""
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported by this proxy.")

    if request.model not in {FULL_MODEL_NAME, RAW_MODEL_NAME}:
        logger.warning("Unsupported model requested", model=request.model)
        detail = f"Unsupported model '{request.model}'. Supported: {FULL_MODEL_NAME}, {RAW_MODEL_NAME}"
        raise HTTPException(status_code=400, detail=detail)

    start_time = time.perf_counter()

    if request.model == FULL_MODEL_NAME:
        extract_image_data_uri(request.messages)
        response = await backend.process_full(request)
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Full OCR pipeline completed", model=request.model, duration_ms=round(duration_ms, 2))
        return response.model_dump()

    result = await backend.process_raw(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info("Raw passthrough completed", model=request.model, duration_ms=round(duration_ms, 2))
    return result
