from __future__ import annotations

from typing import Any

import pytest

from bentoml_ocr.ocr_proxy.api import set_backend_for_tests
from bentoml_ocr.ocr_proxy.backend import build_openai_response
from bentoml_ocr.ocr_proxy.constants import FULL_MODEL_NAME, RAW_MODEL_NAME
from bentoml_ocr.ocr_proxy.models import ChatCompletionRequest, ChatCompletionResponse


class FakeBackend:
    def __init__(self) -> None:
        self.full_calls = 0

    async def process_full(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.full_calls += 1
        return build_openai_response("# OCR\n\nhello world", request.model)

    async def process_raw(self, request: ChatCompletionRequest) -> dict[str, Any]:
        return {"model": request.model, "choices": []}

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_extracts_valid_image_data_uri(app_client: Any, sample_openai_request: dict[str, Any]) -> None:
    backend = FakeBackend()
    set_backend_for_tests(backend)

    response = await app_client.post("/v1/chat/completions", json=sample_openai_request)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "glm-ocr"
    assert data["choices"][0]["message"]["content"].startswith("# OCR")
    assert backend.full_calls == 1


@pytest.mark.asyncio
async def test_rejects_missing_image_content(app_client: Any) -> None:
    backend = FakeBackend()
    set_backend_for_tests(backend)

    payload = {"model": "glm-ocr", "messages": [{"role": "user", "content": "no image"}]}
    response = await app_client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert "image_url content item" in response.text


@pytest.mark.asyncio
async def test_rejects_unknown_model(app_client: Any, sample_openai_request: dict[str, Any]) -> None:
    sample_openai_request["model"] = "does-not-exist"
    response = await app_client.post("/v1/chat/completions", json=sample_openai_request)

    assert response.status_code == 400
    assert "Unsupported model" in response.text


@pytest.mark.asyncio
async def test_list_models_returns_supported_models(app_client: Any) -> None:
    response = await app_client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    model_ids = {model["id"] for model in data["data"]}
    assert FULL_MODEL_NAME in model_ids
    assert RAW_MODEL_NAME in model_ids
