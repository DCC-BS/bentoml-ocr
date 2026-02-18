from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pytest import MonkeyPatch

from bentoml_ocr.ocr_proxy.backend import DefaultOCRBackend
from bentoml_ocr.ocr_proxy.config import AppConfig
from bentoml_ocr.ocr_proxy.container import Container


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


def _make_backend(monkeypatch: MonkeyPatch) -> DefaultOCRBackend:
    monkeypatch.setattr(DefaultOCRBackend, "_init_glmocr_parser", staticmethod(lambda cfg: FakeGlmOcrParser()))
    return DefaultOCRBackend(_test_config())


@pytest.mark.asyncio
async def test_full_pipeline_response_with_fake_parser(
    monkeypatch: MonkeyPatch,
    wire_container: Container,
    app_client: Any,
    sample_openai_request: dict[str, Any],
) -> None:
    backend = _make_backend(monkeypatch)
    with wire_container.backend.override(backend):
        response = await app_client.post("/v1/chat/completions", json=sample_openai_request)

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "glm-ocr"
    assert "## Parsed" in body["choices"][0]["message"]["content"]

    await backend.close()
