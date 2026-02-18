from __future__ import annotations

import asyncio
import base64
import time
import uuid
from typing import Any, cast

import httpx
from fastapi import HTTPException

from ocr_proxy.config import RuntimeConfig
from ocr_proxy.constants import FULL_MODEL_NAME
from ocr_proxy.models import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,
    ContentImagePart,
    ContentTextPart,
    ResponseMessage,
)


def extract_image_data_uri(messages: list[ChatMessage]) -> str:
    for message in messages:
        if not isinstance(message.content, list):
            continue
        for part in message.content:
            if isinstance(part, ContentImagePart):
                url = part.image_url.url.strip()
                if url.startswith("data:image/") and ";base64," in url:
                    return url
                if looks_like_base64(url):
                    return f"data:image/png;base64,{url}"
    raise HTTPException(
        status_code=400,
        detail="Request must include at least one image_url content item with base64 data URI.",
    )


def extract_text_prompt(messages: list[ChatMessage]) -> str:
    prompts: list[str] = []
    for message in messages:
        if isinstance(message.content, str):
            prompts.append(message.content)
            continue
        for part in message.content:
            if isinstance(part, ContentTextPart):
                prompts.append(part.text)
    return "\n".join(p.strip() for p in prompts if p.strip())


def looks_like_base64(value: str) -> bool:
    try:
        base64.b64decode(value, validate=True)
    except Exception:
        return False
    return True


def build_openai_response(content: str, model: str) -> ChatCompletionResponse:
    completion_tokens = max(1, len(content) // 4)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model,
        choices=[ChatChoice(message=ResponseMessage(content=content))],
        usage=ChatUsage(
            prompt_tokens=0,
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens,
        ),
    )


def extract_markdown_from_glmocr_result(result: Any) -> str:
    if isinstance(result, list):
        markdown_parts = [getattr(item, "markdown_result", "") for item in result]
        markdown = "\n\n".join(part for part in markdown_parts if part)
    else:
        markdown = cast(str, getattr(result, "markdown_result", ""))
    return markdown or ""


class DefaultOCRBackend:
    def __init__(self, config: RuntimeConfig):
        self._config = config
        self._raw_http = httpx.AsyncClient(timeout=self._config.request_timeout_seconds)
        self._parser = self._init_glmocr_parser(config)

    @staticmethod
    def _init_glmocr_parser(config: RuntimeConfig) -> Any:
        try:
            from glmocr import GlmOcr
        except Exception as exc:  # pragma: no cover - import is environment-dependent
            raise RuntimeError("glmocr package is not available. Install dependencies with `uv sync`.") from exc

        kwargs: dict[str, Any] = {
            "mode": "selfhosted",
            "timeout": config.request_timeout_seconds,
            "enable_layout": config.enable_layout,
            "log_level": config.log_level,
        }
        if config.config_path:
            kwargs["config_path"] = config.config_path
        return GlmOcr(**kwargs)

    async def process_full(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        image_uri = extract_image_data_uri(request.messages)
        extract_text_prompt(request.messages)

        def _run_parse() -> Any:
            kwargs: dict[str, Any] = {
                "api_url": self._config.vllm_api_url,
                "model": self._config.vllm_model_name,
                "timeout": self._config.request_timeout_seconds,
                "enable_layout": self._config.enable_layout,
                "save_layout_visualization": False,
            }
            return self._parser.parse(image_uri, **kwargs)

        result = await asyncio.to_thread(_run_parse)
        markdown = extract_markdown_from_glmocr_result(result)
        if not markdown:
            markdown = "No OCR content produced by GLM-OCR."
        return build_openai_response(markdown, FULL_MODEL_NAME)

    async def process_raw(self, request: ChatCompletionRequest) -> dict[str, Any]:
        payload = request.model_dump(exclude_none=True)
        if not payload.get("model"):
            payload["model"] = self._config.vllm_model_name
        try:
            response = await self._raw_http.post(self._config.vllm_api_url, json=payload)
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail="Timed out calling external vLLM.") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"vLLM request failed: {exc}") from exc
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return cast(dict[str, Any], response.json())

    async def close(self) -> None:
        await self._raw_http.aclose()
        parser_close = getattr(self._parser, "close", None)
        if callable(parser_close):
            await asyncio.to_thread(parser_close)
