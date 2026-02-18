"""Tests for backend helper functions and DefaultOCRBackend internals."""

from __future__ import annotations

import base64
import io
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import HTTPException
from PIL import Image
from pytest import MonkeyPatch

from bentoml_ocr.ocr_proxy.backend import (
    DefaultOCRBackend,
    build_openai_response,
    extract_image_data_uri,
    extract_markdown_from_glmocr_result,
    extract_text_prompt,
    looks_like_base64,
    validate_image_data_uri,
)
from bentoml_ocr.ocr_proxy.config import AppConfig
from bentoml_ocr.ocr_proxy.models import (
    MODEL_NAME,
    ChatCompletionRequest,
    ChatMessage,
    ContentImagePart,
    ContentTextPart,
    ImageUrl,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_image_b64() -> str:
    """Create a minimal valid PNG image as a base64 string."""
    img = Image.new("RGB", (4, 4), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


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
        import threading

        self.closed = True
        self.close_thread_id = threading.get_ident()


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
        raw_b64 = _valid_image_b64()
        messages = [_image_message(raw_b64)]
        result = extract_image_data_uri(messages)
        assert result == f"data:image/png;base64,{raw_b64}"

    def test_proper_data_uri_returned_unchanged(self) -> None:
        b64 = _valid_image_b64()
        uri = f"data:image/png;base64,{b64}"
        messages = [_image_message(uri)]
        assert extract_image_data_uri(messages) == uri

    def test_raises_400_when_no_image_parts(self) -> None:
        messages = [ChatMessage(role="user", content=[ContentTextPart(type="text", text="no image")])]
        with pytest.raises(HTTPException) as exc_info:
            extract_image_data_uri(messages)
        assert exc_info.value.status_code == 400

    def test_raises_400_for_http_url_not_base64(self) -> None:
        messages = [_image_message("https://example.com/image.png")]
        with pytest.raises(HTTPException) as exc_info:
            extract_image_data_uri(messages)
        assert exc_info.value.status_code == 400

    def test_returns_first_image_when_multiple_present(self) -> None:
        b64a, b64b = _valid_image_b64(), _valid_image_b64()
        msgs = [
            ChatMessage(
                role="user",
                content=[
                    ContentImagePart(type="image_url", image_url=ImageUrl(url=f"data:image/png;base64,{b64a}")),
                    ContentImagePart(type="image_url", image_url=ImageUrl(url=f"data:image/png;base64,{b64b}")),
                ],
            )
        ]
        result = extract_image_data_uri(msgs)
        assert b64a in result

    def test_valid_base64_non_image_raises_422(self) -> None:
        b64_text = base64.b64encode(b"this is plain text, not an image").decode()
        with pytest.raises(HTTPException) as exc_info:
            extract_image_data_uri([_image_message(b64_text)])
        assert exc_info.value.status_code == 422

    def test_raises_400_for_string_content_message(self) -> None:
        messages = [ChatMessage(role="user", content="just text")]
        with pytest.raises(HTTPException) as exc_info:
            extract_image_data_uri(messages)
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# validate_image_data_uri
# ---------------------------------------------------------------------------


class TestValidateImageDataUri:
    def test_valid_image_passes(self) -> None:
        b64 = _valid_image_b64()
        validate_image_data_uri(f"data:image/png;base64,{b64}")

    def test_corrupted_data_raises_422(self) -> None:
        garbage_b64 = base64.b64encode(b"\x00\xff\xfe corrupted").decode()
        with pytest.raises(HTTPException) as exc_info:
            validate_image_data_uri(f"data:image/png;base64,{garbage_b64}")
        assert exc_info.value.status_code == 422
        assert "corrupted" in exc_info.value.detail.lower() or "not a valid image" in exc_info.value.detail.lower()

    def test_data_uri_without_base64_marker_raises_422(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            validate_image_data_uri("data:image/png,notbase64")
        assert exc_info.value.status_code == 422

    def test_jpeg_image_passes(self) -> None:
        buf = io.BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        validate_image_data_uri(f"data:image/jpeg;base64,{b64}")


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

    def test_empty_messages_returns_empty_string(self) -> None:
        assert extract_text_prompt([]) == ""

    def test_all_image_parts_returns_empty_string(self) -> None:
        b64 = _valid_image_b64()
        messages = [
            ChatMessage(
                role="user",
                content=[ContentImagePart(type="image_url", image_url=ImageUrl(url=f"data:image/png;base64,{b64}"))],
            )
        ]
        assert extract_text_prompt(messages) == ""

    def test_multiple_messages_joined_with_newline(self) -> None:
        messages = [
            ChatMessage(role="user", content="line one"),
            ChatMessage(role="user", content="line two"),
        ]
        result = extract_text_prompt(messages)
        assert result == "line one\nline two"


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
# build_openai_response
# ---------------------------------------------------------------------------


class TestBuildOpenaiResponse:
    def test_response_has_all_required_fields(self) -> None:
        resp = build_openai_response("hello", "glm-ocr")
        assert resp.object == "chat.completion"
        assert resp.model == "glm-ocr"
        assert resp.id.startswith("chatcmpl-")
        assert resp.created > 0
        assert len(resp.choices) == 1
        assert resp.choices[0].finish_reason == "stop"
        assert resp.choices[0].index == 0

    def test_completion_tokens_estimated_from_content_length(self) -> None:
        content = "a" * 400
        resp = build_openai_response(content, "glm-ocr")
        assert resp.usage.completion_tokens == 100

    def test_model_name_echoed_in_response(self) -> None:
        resp = build_openai_response("text", "my-custom-model")
        assert resp.model == "my-custom-model"

    def test_empty_content_does_not_raise(self) -> None:
        resp = build_openai_response("", "glm-ocr")
        assert resp.choices[0].message.content == ""
        assert resp.usage.completion_tokens >= 1


# ---------------------------------------------------------------------------
# _init_glmocr_parser
# ---------------------------------------------------------------------------


class TestInitGlmOcrParser:
    def test_constructor_kwargs_with_config_path(self, monkeypatch: MonkeyPatch) -> None:
        class SpyGlmOcr:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        fake_module = types.ModuleType("glmocr")
        fake_module.GlmOcr = SpyGlmOcr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "glmocr", fake_module)

        config = _test_config(config_path="/etc/glmocr.yaml", vllm_api_key="secret")
        parser = DefaultOCRBackend._init_glmocr_parser(config)

        assert parser.kwargs["config_path"] == "/etc/glmocr.yaml"
        assert parser.kwargs["mode"] == "selfhosted"
        assert "api_url" not in parser.kwargs
        assert "model" not in parser.kwargs

    def test_omits_config_path_when_none(self, monkeypatch: MonkeyPatch) -> None:
        class SpyGlmOcr:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        fake_module = types.ModuleType("glmocr")
        fake_module.GlmOcr = SpyGlmOcr  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "glmocr", fake_module)

        config = _test_config(config_path=None, vllm_api_key=None)
        parser = DefaultOCRBackend._init_glmocr_parser(config)

        assert "config_path" not in parser.kwargs


# ---------------------------------------------------------------------------
# process_full
# ---------------------------------------------------------------------------

_VALID_B64 = _valid_image_b64()


class TestProcessFull:
    @pytest.mark.asyncio
    async def test_parse_called_with_image_uri(self, monkeypatch: MonkeyPatch) -> None:
        backend = _make_backend(monkeypatch)
        parser: _FakeGlmOcrParser = backend._parser  # type: ignore[assignment]

        captured_uri: list[str] = []

        def spy_parse(uri: str, **kwargs: Any) -> _FakeParseItem:
            captured_uri.append(uri)
            return _FakeParseItem(markdown_result="ok")

        parser.parse = spy_parse  # type: ignore[assignment]

        request = _sample_request(_VALID_B64)
        response = await backend.process_full(request)
        assert len(captured_uri) == 1
        assert _VALID_B64 in captured_uri[0]
        assert response.choices[0].message.content == "ok"
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

    @pytest.mark.asyncio
    async def test_parser_receives_save_layout_false_kwarg(self, monkeypatch: MonkeyPatch) -> None:
        backend = _make_backend(monkeypatch)
        parser: _FakeGlmOcrParser = backend._parser  # type: ignore[assignment]
        captured_kwargs: list[dict[str, Any]] = []
        original_parse = parser.parse

        def spy(uri: str, **kwargs: Any) -> _FakeParseItem:
            captured_kwargs.append(kwargs)
            return original_parse(uri, **kwargs)

        parser.parse = spy  # type: ignore[assignment]
        await backend.process_full(_sample_request(_VALID_B64))
        assert captured_kwargs[0].get("save_layout_visualization") is False
        await backend.close()

    @pytest.mark.asyncio
    async def test_response_model_is_always_full_model_name(self, monkeypatch: MonkeyPatch) -> None:
        backend = _make_backend(monkeypatch)
        parser: _FakeGlmOcrParser = backend._parser  # type: ignore[assignment]
        parser.parse = lambda _uri, **kw: _FakeParseItem(markdown_result="result")  # type: ignore[assignment]

        request = _sample_request(_VALID_B64)
        response = await backend.process_full(request)
        assert response.model == MODEL_NAME
        await backend.close()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_parser_close_is_invoked_in_thread(self, monkeypatch: MonkeyPatch) -> None:
        import threading

        backend = _make_backend(monkeypatch)
        parser: _FakeGlmOcrParser = backend._parser  # type: ignore[assignment]
        assert not hasattr(parser, "closed")

        main_thread_id = threading.get_ident()
        await backend.close()
        assert parser.closed is True
        assert parser.close_thread_id != main_thread_id

    @pytest.mark.asyncio
    async def test_close_without_parser_close_method_does_not_raise(self, monkeypatch: MonkeyPatch) -> None:
        class _ParserWithoutClose:
            def parse(self, _uri: str, **kwargs: Any) -> _FakeParseItem:
                return _FakeParseItem(markdown_result="")

        parser = _ParserWithoutClose()
        monkeypatch.setattr(DefaultOCRBackend, "_init_glmocr_parser", staticmethod(lambda cfg: parser))
        backend = DefaultOCRBackend(_test_config())
        await backend.close()
