from __future__ import annotations

import base64
from collections.abc import AsyncGenerator
from io import BytesIO
from typing import Any

import httpx
import pytest
import pytest_asyncio
from PIL import Image

from bentoml_ocr import service
from bentoml_ocr.ocr_proxy.backend import build_openai_response
from bentoml_ocr.ocr_proxy.models import ChatCompletionRequest, ChatCompletionResponse


class FakeBackend:
    def __init__(self) -> None:
        self.full_calls = 0
        self.raw_calls = 0
        self.last_request: ChatCompletionRequest | None = None
        self.raw_response: dict[str, Any] = {
            "id": "chatcmpl-raw",
            "object": "chat.completion",
            "created": 0,
            "model": "glm-ocr-raw",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "raw ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def process_full(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.full_calls += 1
        self.last_request = request
        return build_openai_response("# OCR\n\nhello world", request.model)

    async def process_raw(self, request: ChatCompletionRequest) -> dict[str, Any]:
        self.raw_calls += 1
        self.last_request = request
        return self.raw_response

    async def close(self) -> None:
        return None


@pytest.fixture
def sample_image_b64() -> str:
    image = Image.new("RGB", (8, 8), color=(255, 255, 255))
    output = BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("utf-8")


@pytest.fixture
def sample_openai_request(sample_image_b64: str) -> dict[str, Any]:
    return {
        "model": "glm-ocr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please OCR this page."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{sample_image_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0.0,
    }


@pytest_asyncio.fixture
async def app_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    transport = httpx.ASGITransport(app=service.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture(autouse=True)
def reset_backend() -> None:
    service.set_backend_for_tests(None)
