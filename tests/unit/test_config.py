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

        assert "GLMOCR_OCR_API_KEY" not in os.environ

    def test_from_env_calls_apply_env(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_API_URL", "http://x/v1")
        monkeypatch.setenv("VLLM_MODEL_NAME", "test-model")
        monkeypatch.delenv("GLMOCR_OCR_API_URL", raising=False)
        monkeypatch.delenv("GLMOCR_OCR_MODEL", raising=False)

        AppConfig.from_env()

        assert os.environ["GLMOCR_OCR_API_URL"] == "http://x/v1"
        assert os.environ["GLMOCR_OCR_MODEL"] == "test-model"


class TestAppConfigStr:
    def test_masks_vllm_api_key(self) -> None:
        cfg = _test_config(vllm_api_key="super-secret-token-12345")
        text = str(cfg)
        assert "super-secret-token-12345" not in text
        assert "vllm_api_key=" in text
        assert "None" not in text.split("vllm_api_key=")[1].split(",")[0]

    def test_shows_none_when_keys_absent(self) -> None:
        cfg = _test_config(vllm_api_key=None)
        text = str(cfg)
        assert "vllm_api_key=None" in text

    def test_masks_config_path(self) -> None:
        cfg = _test_config(config_path="/secret/path/to/config.yaml")
        text = str(cfg)
        assert "/secret/path/to/config.yaml" not in text

    def test_includes_new_fields(self) -> None:
        cfg = _test_config()
        text = str(cfg)
        assert "max_body_size_bytes=" in text
        assert "max_http_connections=" in text
        assert "retry_max_attempts=" in text

    def test_from_env_loads_new_fields(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_API_URL", "http://x/v1")
        monkeypatch.setenv("VLLM_MODEL_NAME", "m")
        monkeypatch.setenv("MAX_BODY_SIZE_BYTES", "1024")
        monkeypatch.setenv("MAX_HTTP_CONNECTIONS", "50")
        monkeypatch.setenv("RETRY_MAX_ATTEMPTS", "5")

        cfg = AppConfig.from_env()

        assert cfg.max_body_size_bytes == 1024
        assert cfg.max_http_connections == 50
        assert cfg.retry_max_attempts == 5
