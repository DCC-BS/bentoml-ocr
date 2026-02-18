from __future__ import annotations

import time
from typing import Any

from dcc_backend_common.logger import get_logger
from fastapi import FastAPI, HTTPException

from bentoml_ocr.ocr_proxy.backend import DefaultOCRBackend, extract_image_data_uri
from bentoml_ocr.ocr_proxy.config import AppConfig
from bentoml_ocr.ocr_proxy.constants import FULL_MODEL_NAME, RAW_MODEL_NAME
from bentoml_ocr.ocr_proxy.models import (
    ChatCompletionRequest,
    ModelCard,
    ModelListResponse,
    OCRBackend,
)

logger = get_logger(__name__)

app = FastAPI(title="GLM-OCR Docling-Compatible Proxy")

_backend: OCRBackend | None = None


def set_backend_for_tests(backend: OCRBackend | None) -> None:
    global _backend
    _backend = backend


def get_backend() -> OCRBackend:
    global _backend
    if _backend is None:
        _backend = DefaultOCRBackend(AppConfig.from_env())
    return _backend


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    return ModelListResponse(
        data=[ModelCard(id=FULL_MODEL_NAME), ModelCard(id=RAW_MODEL_NAME)],
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported by this proxy.")

    if request.model not in {FULL_MODEL_NAME, RAW_MODEL_NAME}:
        logger.warning("Unsupported model requested", model=request.model)
        detail = f"Unsupported model '{request.model}'. Supported: {FULL_MODEL_NAME}, {RAW_MODEL_NAME}"
        raise HTTPException(status_code=400, detail=detail)

    backend = get_backend()
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
