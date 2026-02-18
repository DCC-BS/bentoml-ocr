from __future__ import annotations

from typing import Any

import pytest

from bentoml_ocr.ocr_proxy.constants import FULL_MODEL_NAME, RAW_MODEL_NAME
from bentoml_ocr.ocr_proxy.container import Container
from tests.conftest import FakeBackend


@pytest.mark.asyncio
async def test_extracts_valid_image_data_uri(
    wire_container: Container,
    app_client: Any,
    sample_openai_request: dict[str, Any],
) -> None:
    backend = FakeBackend()
    with wire_container.backend.override(backend):
        response = await app_client.post("/v1/chat/completions", json=sample_openai_request)

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "glm-ocr"
    assert data["choices"][0]["message"]["content"].startswith("# OCR")
    assert backend.full_calls == 1


@pytest.mark.asyncio
async def test_rejects_missing_image_content(wire_container: Container, app_client: Any) -> None:
    backend = FakeBackend()
    with wire_container.backend.override(backend):
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


@pytest.mark.asyncio
async def test_streaming_request_returns_400(
    app_client: Any,
    sample_openai_request: dict[str, Any],
) -> None:
    sample_openai_request["stream"] = True
    response = await app_client.post("/v1/chat/completions", json=sample_openai_request)
    assert response.status_code == 400
    assert "Streaming is not supported" in response.text


@pytest.mark.asyncio
async def test_missing_messages_field_returns_422(app_client: Any) -> None:
    response = await app_client.post("/v1/chat/completions", json={"model": "glm-ocr"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_response_object_field_is_chat_completion(
    wire_container: Container,
    app_client: Any,
    sample_openai_request: dict[str, Any],
) -> None:
    with wire_container.backend.override(FakeBackend()):
        response = await app_client.post("/v1/chat/completions", json=sample_openai_request)
    data = response.json()
    assert data["object"] == "chat.completion"


@pytest.mark.asyncio
async def test_response_choices_finish_reason_is_stop(
    wire_container: Container,
    app_client: Any,
    sample_openai_request: dict[str, Any],
) -> None:
    with wire_container.backend.override(FakeBackend()):
        response = await app_client.post("/v1/chat/completions", json=sample_openai_request)
    assert response.json()["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_http_url_image_raises_400(
    wire_container: Container,
    app_client: Any,
    sample_openai_request: dict[str, Any],
) -> None:
    sample_openai_request["messages"][0]["content"][1]["image_url"]["url"] = "https://example.com/img.png"
    with wire_container.backend.override(FakeBackend()):
        response = await app_client.post("/v1/chat/completions", json=sample_openai_request)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_post_to_models_endpoint_returns_405(app_client: Any) -> None:
    response = await app_client.post("/v1/models", json={})
    assert response.status_code == 405


@pytest.mark.asyncio
async def test_models_response_contains_owned_by_field(app_client: Any) -> None:
    response = await app_client.get("/v1/models")
    for model in response.json()["data"]:
        assert model["object"] == "model"
        assert model["owned_by"] == "bentoml-ocr"


@pytest.mark.asyncio
async def test_raw_backend_called_once_for_raw_model(
    wire_container: Container,
    app_client: Any,
    sample_openai_request: dict[str, Any],
) -> None:
    backend = FakeBackend()
    sample_openai_request["model"] = "glm-ocr-raw"
    with wire_container.backend.override(backend):
        await app_client.post("/v1/chat/completions", json=sample_openai_request)
    assert backend.raw_calls == 1
    assert backend.full_calls == 0
