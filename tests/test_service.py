from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import respx
from httpx import ReadTimeout
from pytest import MonkeyPatch

from bentoml_ocr import service
from bentoml_ocr.ocr_proxy.backend import DefaultOCRBackend
from bentoml_ocr.ocr_proxy.config import AppConfig


@dataclass
class FakePipelineResult:
    markdown_result: str


class FakeGlmOcrParser:
    def parse(self, _image_uri: str, **_kwargs: Any) -> FakePipelineResult:
        return FakePipelineResult(markdown_result="## Parsed\n\ncontent")


def _test_config() -> AppConfig:
    return AppConfig(
        vllm_api_url="http://vllm.local/v1/chat/completions",
        vllm_model_name="glm-ocr",
        request_timeout_seconds=10,
        enable_layout=True,
        max_workers=16,
        log_level="INFO",
        config_path=None,
    )


@pytest.mark.asyncio
async def test_full_pipeline_response_with_fake_parser(
    monkeypatch: MonkeyPatch,
    app_client: Any,
    sample_openai_request: dict[str, Any],
) -> None:
    monkeypatch.setattr(DefaultOCRBackend, "_init_glmocr_parser", lambda self, config: FakeGlmOcrParser())
    backend = DefaultOCRBackend(_test_config())
    service.set_backend_for_tests(backend)

    response = await app_client.post("/v1/chat/completions", json=sample_openai_request)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "glm-ocr"
    assert "## Parsed" in body["choices"][0]["message"]["content"]

    await backend.close()


@pytest.mark.asyncio
async def test_raw_model_passthrough_hits_vllm(
    monkeypatch: MonkeyPatch,
    app_client: Any,
    sample_openai_request: dict[str, Any],
) -> None:
    monkeypatch.setattr(DefaultOCRBackend, "_init_glmocr_parser", lambda self, config: FakeGlmOcrParser())
    backend = DefaultOCRBackend(_test_config())
    service.set_backend_for_tests(backend)
    sample_openai_request["model"] = "glm-ocr-raw"

    with respx.mock(assert_all_called=True) as mocked:
        route = mocked.post("http://vllm.local/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-raw",
                "object": "chat.completion",
                "created": 123,
                "model": "glm-ocr",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "raw data"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
        response = await app_client.post("/v1/chat/completions", json=sample_openai_request)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "raw data"
    assert route.called

    await backend.close()


@pytest.mark.asyncio
async def test_raw_timeout_returns_504(
    monkeypatch: MonkeyPatch,
    app_client: Any,
    sample_openai_request: dict[str, Any],
) -> None:
    monkeypatch.setattr(DefaultOCRBackend, "_init_glmocr_parser", lambda self, config: FakeGlmOcrParser())
    backend = DefaultOCRBackend(_test_config())
    service.set_backend_for_tests(backend)
    sample_openai_request["model"] = "glm-ocr-raw"

    with respx.mock(assert_all_called=True) as mocked:
        mocked.post("http://vllm.local/v1/chat/completions").mock(side_effect=ReadTimeout("timeout"))
        response = await app_client.post("/v1/chat/completions", json=sample_openai_request)

    assert response.status_code == 504
    assert "Timed out calling external vLLM" in response.text
    await backend.close()


@pytest.mark.asyncio
async def test_raw_error_propagates_status_code(
    monkeypatch: MonkeyPatch,
    app_client: Any,
    sample_openai_request: dict[str, Any],
) -> None:
    monkeypatch.setattr(DefaultOCRBackend, "_init_glmocr_parser", lambda self, config: FakeGlmOcrParser())
    backend = DefaultOCRBackend(_test_config())
    service.set_backend_for_tests(backend)
    sample_openai_request["model"] = "glm-ocr-raw"

    with respx.mock(assert_all_called=True) as mocked:
        mocked.post("http://vllm.local/v1/chat/completions").respond(
            500,
            text="vLLM internal error",
        )
        response = await app_client.post("/v1/chat/completions", json=sample_openai_request)

    assert response.status_code == 500
    assert "vLLM internal error" in response.text
    await backend.close()
