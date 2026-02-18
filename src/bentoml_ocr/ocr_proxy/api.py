from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from ocr_proxy.backend import DefaultOCRBackend, extract_image_data_uri
from ocr_proxy.config import load_runtime_config
from ocr_proxy.constants import FULL_MODEL_NAME, RAW_MODEL_NAME
from ocr_proxy.models import (
    ChatCompletionRequest,
    ModelCard,
    ModelListResponse,
    OCRBackend,
)

app = FastAPI(title="GLM-OCR Docling-Compatible Proxy")

_backend: OCRBackend | None = None


def set_backend_for_tests(backend: OCRBackend | None) -> None:
    global _backend
    _backend = backend


def get_backend() -> OCRBackend:
    global _backend
    if _backend is None:
        _backend = DefaultOCRBackend(load_runtime_config())
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
        detail = f"Unsupported model '{request.model}'. Supported: {FULL_MODEL_NAME}, {RAW_MODEL_NAME}"
        raise HTTPException(status_code=400, detail=detail)

    backend = get_backend()
    if request.model == FULL_MODEL_NAME:
        extract_image_data_uri(request.messages)
        response = await backend.process_full(request)
        return response.model_dump()
    return await backend.process_raw(request)
