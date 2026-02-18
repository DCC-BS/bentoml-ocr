"""Tests for AppConfig, including GLMOCR_OCR_* env var application."""

from __future__ import annotations

import os

from pytest import MonkeyPatch

from bentoml_ocr.ocr_proxy.config import AppConfig


def _test_config(**overrides: object) -> AppConfig:
    defaults: dict[str, object] = {
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
    return AppConfig(**defaults)  # type: ignore[arg-type]


class TestAppConfigApplyEnv:
    def test_sets_api_url_and_model(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.delenv("GLMOCR_OCR_API_URL", raising=False)
        monkeypatch.delenv("GLMOCR_OCR_MODEL", raising=False)

        cfg = _test_config(vllm_api_url="http://vllm/v1", vllm_model_name="my-model")
        cfg.apply_env()

        assert os.environ["GLMOCR_OCR_API_URL"] == "http://vllm/v1"
        assert os.environ["GLMOCR_OCR_MODEL"] == "my-model"

    def test_sets_api_key_when_provided(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.delenv("GLMOCR_OCR_API_KEY", raising=False)

        cfg = _test_config(vllm_api_key="secret")
        cfg.apply_env()

        assert os.environ["GLMOCR_OCR_API_KEY"] == "secret"

    def test_does_not_set_api_key_when_none(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.delenv("GLMOCR_OCR_API_KEY", raising=False)

        cfg = _test_config(vllm_api_key=None)
        cfg.apply_env()

        assert "GLMOCR_OCR_API_KEY" not in os.environ

    def test_from_env_calls_apply_env(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_API_URL", "http://x/v1")
        monkeypatch.setenv("VLLM_MODEL_NAME", "test-model")
        monkeypatch.delenv("GLMOCR_OCR_API_URL", raising=False)
        monkeypatch.delenv("GLMOCR_OCR_MODEL", raising=False)

        AppConfig.from_env()

        assert os.environ["GLMOCR_OCR_API_URL"] == "http://x/v1"
        assert os.environ["GLMOCR_OCR_MODEL"] == "test-model"
