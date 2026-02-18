"""Tests for the glmocr import failure path in DefaultOCRBackend._init_glmocr_parser."""

from __future__ import annotations

import sys
from typing import Any

import pytest
from pytest import MonkeyPatch

from bentoml_ocr.ocr_proxy.backend import DefaultOCRBackend
from bentoml_ocr.ocr_proxy.config import AppConfig


def _test_config() -> AppConfig:
    return AppConfig(
        vllm_api_url="http://vllm.local/v1/chat/completions",
        vllm_model_name="glm-ocr",
        request_timeout_seconds=10,
        enable_layout=True,
        max_workers=4,
        log_level="INFO",
        config_path=None,
    )


class TestGlmOcrImportFailure:
    def test_raises_runtime_error_when_glmocr_missing(self, monkeypatch: MonkeyPatch) -> None:
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "glmocr":
                raise ModuleNotFoundError("No module named 'glmocr'")
            return original_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "glmocr", raising=False)
        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(RuntimeError, match="glmocr package is not available"):
            DefaultOCRBackend._init_glmocr_parser(_test_config())
