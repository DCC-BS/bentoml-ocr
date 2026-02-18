"""OCR backend implementation using GLM-OCR SDK and vLLM passthrough."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import httpx
from dcc_backend_common.logger import get_logger
from fastapi import HTTPException
from PIL import Image
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from bentoml_ocr.ocr_proxy.config import AppConfig
from bentoml_ocr.ocr_proxy.constants import FULL_MODEL_NAME
from bentoml_ocr.ocr_proxy.models import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,
    ContentImagePart,
    ContentTextPart,
    ResponseMessage,
)

logger = get_logger(__name__)


def validate_image_data_uri(data_uri: str) -> None:
    """Verify that the base64 payload in a data URI decodes to a valid image.

    Raises:
        HTTPException: 422 if the data cannot be decoded as an image.
    """
    try:
        raw_b64 = data_uri.split(";base64,", 1)[1]
        image_bytes = base64.b64decode(raw_b64)
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="The provided image data is corrupted or not a valid image.",
        ) from exc


def extract_image_data_uri(messages: list[ChatMessage]) -> str:
    """Extract the first base64 image data URI from chat messages.

    Raises:
        HTTPException: If no image_url content part with base64 data is found,
            or if the image data is corrupted.
    """
    for message in messages:
        if not isinstance(message.content, list):
            continue
        for part in message.content:
            if isinstance(part, ContentImagePart):
                url = part.image_url.url.strip()
                if url.startswith("data:image/") and ";base64," in url:
                    validate_image_data_uri(url)
                    return url
                if looks_like_base64(url):
                    uri = f"data:image/png;base64,{url}"
                    validate_image_data_uri(uri)
                    return uri
    raise HTTPException(
        status_code=400,
        detail="Request must include at least one image_url content item with base64 data URI.",
    )


def extract_text_prompt(messages: list[ChatMessage]) -> str:
    """Concatenate all text content from chat messages into a single prompt string."""
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
    """Check whether a string is valid base64-encoded data."""
    try:
        base64.b64decode(value, validate=True)
    except Exception:
        return False
    return True


def build_openai_response(content: str, model: str) -> ChatCompletionResponse:
    """Wrap OCR output text in an OpenAI-compatible chat completion response."""
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
    """Extract concatenated markdown text from a GLM-OCR parse result."""
    if isinstance(result, list):
        markdown_parts = [getattr(item, "markdown_result", "") for item in result]
        markdown = "\n\n".join(part for part in markdown_parts if part)
    else:
        markdown = cast(str, getattr(result, "markdown_result", ""))
    return markdown or ""


_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _is_retryable_httpx_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in _RETRYABLE_STATUS_CODES:
        return True
    return isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError))


class DefaultOCRBackend:
    """OCR backend that delegates to the GLM-OCR SDK for full parsing and to vLLM for raw passthrough."""

    def __init__(self, config: AppConfig) -> None:
        """Initialize the backend with an HTTP client for vLLM and a GLM-OCR parser.

        Args:
            config: Application configuration containing vLLM and GLM-OCR settings.
        """
        self._config = config
        headers = {"Authorization": f"Bearer {config.vllm_api_key}"} if config.vllm_api_key else {}
        self._raw_http = httpx.AsyncClient(
            timeout=self._config.request_timeout_seconds,
            headers=headers,
            limits=httpx.Limits(
                max_connections=config.max_http_connections,
                max_keepalive_connections=config.max_keepalive_connections,
            ),
        )
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._parser = self._init_glmocr_parser(config)
        logger.info(
            "OCR backend initialized",
            vllm_url=config.vllm_api_url,
            model=config.vllm_model_name,
            enable_layout=config.enable_layout,
            timeout_seconds=config.request_timeout_seconds,
            max_workers=config.max_workers,
            max_http_connections=config.max_http_connections,
        )

    @staticmethod
    def _init_glmocr_parser(config: AppConfig) -> Any:
        try:
            from glmocr import GlmOcr
        except Exception as exc:
            raise RuntimeError("glmocr package is not available. Install dependencies with `uv sync`.") from exc

        if config.vllm_api_key:
            os.environ["GLMOCR_OCR_API_KEY"] = config.vllm_api_key
        try:
            kwargs: dict[str, Any] = {
                "mode": "selfhosted",
                "timeout": config.request_timeout_seconds,
                "enable_layout": config.enable_layout,
                "log_level": config.log_level,
            }
            if config.config_path:
                kwargs["config_path"] = config.config_path
            return GlmOcr(**kwargs)
        finally:
            os.environ.pop("GLMOCR_OCR_API_KEY", None)

    async def process_full(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Run the full GLM-OCR pipeline on the image in the request.

        Args:
            request: Chat completion request containing an image as base64 data URI.

        Returns:
            An OpenAI-compatible response with the OCR markdown as content.
        """
        image_uri = extract_image_data_uri(request.messages)
        text_prompt = extract_text_prompt(request.messages)

        def _run_parse() -> Any:
            return self._parser.parse(image_uri, prompt=text_prompt or None, save_layout_visualization=False)

        start = time.perf_counter()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self._executor, _run_parse)
        parse_ms = (time.perf_counter() - start) * 1000

        markdown = extract_markdown_from_glmocr_result(result)
        if not markdown:
            logger.warning("GLM-OCR produced empty result", model=request.model)
            markdown = "No OCR content produced by GLM-OCR."

        logger.info(
            "GLM-OCR parse completed",
            model=request.model,
            parse_duration_ms=round(parse_ms, 2),
            result_length=len(markdown),
            text_prompt=text_prompt or None,
        )
        return build_openai_response(markdown, FULL_MODEL_NAME)

    async def process_raw(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Forward the request directly to vLLM and return the raw JSON response.

        Args:
            request: Chat completion request to pass through to vLLM.

        Returns:
            The raw JSON response from the vLLM server.

        Raises:
            HTTPException: On timeout, HTTP errors, or non-2xx vLLM responses.
        """
        payload = request.model_dump(exclude_none=True)
        if not payload.get("model"):
            payload["model"] = self._config.vllm_model_name
        try:
            response = await self._send_raw_request(payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            logger.error("vLLM request timed out", url=self._config.vllm_api_url)
            raise HTTPException(status_code=504, detail="Timed out calling external vLLM.") from exc
        except httpx.HTTPStatusError as exc:
            logger.error(
                "vLLM returned error status",
                url=self._config.vllm_api_url,
                status_code=exc.response.status_code,
                response_body=exc.response.text,
            )
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Upstream service returned {exc.response.status_code}.",
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("vLLM request failed", url=self._config.vllm_api_url, error=str(exc))
            raise HTTPException(status_code=502, detail="Upstream service request failed.") from exc
        return cast(dict[str, Any], response.json())

    async def _send_raw_request(self, payload: dict[str, Any]) -> httpx.Response:
        """Send a single HTTP request to vLLM with retry logic for transient failures."""

        @retry(
            stop=stop_after_attempt(self._config.retry_max_attempts),
            wait=wait_exponential(
                multiplier=self._config.retry_backoff_base_seconds,
                max=self._config.retry_backoff_max_seconds,
            ),
            retry=retry_if_exception(_is_retryable_httpx_error),
            reraise=True,
        )
        async def _do_request() -> httpx.Response:
            return await self._raw_http.post(self._config.vllm_api_url, json=payload)

        return await _do_request()

    async def close(self) -> None:
        """Shut down the HTTP client, thread pool, and GLM-OCR parser."""
        await self._raw_http.aclose()
        self._executor.shutdown(wait=False)
        parser_close = getattr(self._parser, "close", None)
        if callable(parser_close):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, parser_close)
        logger.info("OCR backend closed")
