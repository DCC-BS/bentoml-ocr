"""Tests for backend helper functions and DefaultOCRBackend internals."""

from __future__ import annotations

import base64
import sys
import types
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import respx
from pytest import MonkeyPatch

from bentoml_ocr.ocr_proxy.backend import (
    DefaultOCRBackend,
    extract_image_data_uri,
    extract_markdown_from_glmocr_result,
    extract_text_prompt,
    looks_like_base64,
)
from bentoml_ocr.ocr_proxy.config import AppConfig
from bentoml_ocr.ocr_proxy.models import (
    ChatCompletionRequest,
    ChatMessage,
    ContentImagePart,
    ContentTextPart,
    ImageUrl,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _test_config(**overrides: Any) -> AppConfig:
    defaults: dict[str, Any] = {
        "vllm_api_url": "http://vllm.local/v1/chat/completions",
        "vllm_model_name": "glm-ocr",
        "request_timeout_seconds": 10,
        "enable_layout": True,
        "max_workers": 16,
        "log_level": "INFO",
        "config_path": None,
        "vllm_api_key": None,
    }
    defaults.update(overrides)
    return AppConfig(**defaults)


@dataclass
class _FakeParseItem:
    markdown_result: str


class _FakeGlmOcrParser:
    """Lightweight stand-in for glmocr.GlmOcr."""

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs

    def parse(self, _image_uri: str, **kwargs: Any) -> _FakeParseItem:
        self.last_parse_kwargs = kwargs
        return _FakeParseItem(markdown_result="")

    def close(self) -> None:
        self.closed = True


def _make_backend(monkeypatch: MonkeyPatch, **config_overrides: Any) -> DefaultOCRBackend:
    parser = _FakeGlmOcrParser()
    monkeypatch.setattr(DefaultOCRBackend, "_init_glmocr_parser", staticmethod(lambda cfg: parser))
    return DefaultOCRBackend(_test_config(**config_overrides))


def _image_message(url: str) -> ChatMessage:
    return ChatMessage(
        role="user",
        content=[ContentImagePart(type="image_url", image_url=ImageUrl(url=url))],
    )


def _sample_request(image_b64: str, model: str = "glm-ocr") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[
            ChatMessage(
                role="user",
                content=[
                    ContentTextPart(type="text", text="OCR this"),
                    ContentImagePart(
                        type="image_url",
                        image_url=ImageUrl(url=f"data:image/png;base64,{image_b64}"),
                    ),
                ],
            )
        ],
    )


# ---------------------------------------------------------------------------
# looks_like_base64
# ---------------------------------------------------------------------------


class TestLooksLikeBase64:
    def test_valid_base64(self) -> None:
        assert looks_like_base64(base64.b64encode(b"hello").decode()) is True

    def test_invalid_base64(self) -> None:
        assert looks_like_base64("definitely not base64!!!") is False

    def test_empty_string_is_valid(self) -> None:
        assert looks_like_base64("") is True


# ---------------------------------------------------------------------------
# extract_image_data_uri
# ---------------------------------------------------------------------------


class TestExtractImageDataUri:
    def test_raw_base64_is_wrapped_in_data_uri(self) -> None:
        raw_b64 = base64.b64encode(b"fake image bytes").decode()
        messages = [_image_message(raw_b64)]
        result = extract_image_data_uri(messages)
        assert result == f"data:image/png;base64,{raw_b64}"

    def test_proper_data_uri_returned_unchanged(self) -> None:
        uri = "data:image/jpeg;base64,AAAA"
        messages = [_image_message(uri)]
        assert extract_image_data_uri(messages) == uri


# ---------------------------------------------------------------------------
# extract_text_prompt
# ---------------------------------------------------------------------------


class TestExtractTextPrompt:
    def test_string_content_message(self) -> None:
        messages = [ChatMessage(role="user", content="Describe the image")]
        assert extract_text_prompt(messages) == "Describe the image"

    def test_mixed_string_and_list_content(self) -> None:
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(
                role="user",
                content=[ContentTextPart(type="text", text="OCR this")],
            ),
        ]
        result = extract_text_prompt(messages)
        assert "You are helpful." in result
        assert "OCR this" in result


# ---------------------------------------------------------------------------
# extract_markdown_from_glmocr_result
# ---------------------------------------------------------------------------


class TestExtractMarkdownFromGlmOcrResult:
    def test_list_of_items(self) -> None:
        items = [_FakeParseItem("## Page 1"), _FakeParseItem("## Page 2")]
        result = extract_markdown_from_glmocr_result(items)
        assert "Page 1" in result
        assert "Page 2" in result

    def test_single_item(self) -> None:
        result = extract_markdown_from_glmocr_result(_FakeParseItem("hello"))
        assert result == "hello"

    def test_empty_list_returns_empty(self) -> None:
        assert extract_markdown_from_glmocr_result([]) == ""


# ---------------------------------------------------------------------------
# _init_glmocr_parser
# ---------------------------------------------------------------------------


class TestInitGlmOcrParser:
    def test_creates_parser_with_config_path(self, monkeypatch: MonkeyPatch) -> None:
        class SpyGlmOcr:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        fake_module = types.ModuleType("glmocr")
        fake_module.GlmOcr = SpyGlmOcr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "glmocr", fake_module)

        config = _test_config(config_path="/etc/glmocr.yaml")
        parser = DefaultOCRBackend._init_glmocr_parser(config)
        assert parser.kwargs["config_path"] == "/etc/glmocr.yaml"
        assert parser.kwargs["mode"] == "selfhosted"

    def test_creates_parser_without_config_path(self, monkeypatch: MonkeyPatch) -> None:
        class SpyGlmOcr:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        fake_module = types.ModuleType("glmocr")
        fake_module.GlmOcr = SpyGlmOcr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "glmocr", fake_module)

        config = _test_config(config_path=None)
        parser = DefaultOCRBackend._init_glmocr_parser(config)
        assert "config_path" not in parser.kwargs


# ---------------------------------------------------------------------------
# process_full
# ---------------------------------------------------------------------------

_VALID_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfakedata").decode()


class TestProcessFull:
    @pytest.mark.asyncio
    async def test_api_key_forwarded_to_parser(self, monkeypatch: MonkeyPatch) -> None:
        backend = _make_backend(monkeypatch, vllm_api_key="secret-key")
        parser: _FakeGlmOcrParser = backend._parser  # type: ignore[assignment]
        parser.parse = lambda _uri, **kw: (_FakeParseItem("ok"), kw)  # type: ignore[assignment]

        # Override parse to capture kwargs
        captured: dict[str, Any] = {}

        def spy_parse(_uri: str, **kwargs: Any) -> _FakeParseItem:
            captured.update(kwargs)
            return _FakeParseItem(markdown_result="ok")

        parser.parse = spy_parse  # type: ignore[assignment]

        request = _sample_request(_VALID_B64)
        await backend.process_full(request)
        assert captured["api_key"] == "secret-key"
        await backend.close()

    @pytest.mark.asyncio
    async def test_empty_markdown_produces_fallback_message(self, monkeypatch: MonkeyPatch) -> None:
        backend = _make_backend(monkeypatch)
        parser: _FakeGlmOcrParser = backend._parser  # type: ignore[assignment]
        parser.parse = lambda _uri, **kw: _FakeParseItem(markdown_result="")  # type: ignore[assignment]

        request = _sample_request(_VALID_B64)
        response = await backend.process_full(request)
        assert response.choices[0].message.content == "No OCR content produced by GLM-OCR."
        await backend.close()


# ---------------------------------------------------------------------------
# process_raw
# ---------------------------------------------------------------------------


class TestProcessRaw:
    @pytest.mark.asyncio
    async def test_missing_model_uses_config_default(self, monkeypatch: MonkeyPatch) -> None:
        backend = _make_backend(monkeypatch)

        with respx.mock(assert_all_called=True) as mocked:
            route = mocked.post("http://vllm.local/v1/chat/completions").respond(200, json={"choices": []})
            request = ChatCompletionRequest(
                model="",
                messages=[ChatMessage(role="user", content="test")],
            )
            result = await backend.process_raw(request)
            assert route.called

        assert isinstance(result, dict)
        await backend.close()

    @pytest.mark.asyncio
    async def test_generic_http_error_returns_502(self, monkeypatch: MonkeyPatch) -> None:
        backend = _make_backend(monkeypatch)

        with respx.mock() as mocked:
            mocked.post("http://vllm.local/v1/chat/completions").mock(
                side_effect=httpx.ConnectError("connection refused")
            )
            request = _sample_request(_VALID_B64, model="glm-ocr-raw")
            with pytest.raises(httpx.HTTPStatusError if False else Exception) as exc_info:
                await backend.process_raw(request)

            from fastapi import HTTPException

            assert isinstance(exc_info.value, HTTPException)
            assert exc_info.value.status_code == 502

        await backend.close()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_parser_close_is_invoked_in_thread(self, monkeypatch: MonkeyPatch) -> None:
        backend = _make_backend(monkeypatch)
        parser: _FakeGlmOcrParser = backend._parser  # type: ignore[assignment]
        assert not hasattr(parser, "closed")

        await backend.close()
        assert parser.closed is True
