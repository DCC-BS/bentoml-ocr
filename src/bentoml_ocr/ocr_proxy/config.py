from __future__ import annotations

import os
from typing import override

from dcc_backend_common.config import AbstractAppConfig, get_env_or_throw, log_secret
from pydantic import Field


class AppConfig(AbstractAppConfig):
    """Application configuration loaded from environment variables."""

    vllm_api_url: str = Field(description="The URL for the vLLM chat completions endpoint")
    vllm_api_key: str | None = Field(default=None, description="API token for authenticating with the vLLM server")
    vllm_model_name: str = Field(description="The model name to use for vLLM requests")
    request_timeout_seconds: int = Field(default=300, description="Timeout in seconds for OCR requests")
    enable_layout: bool = Field(default=True, description="Enable layout analysis in GLM-OCR")
    max_workers: int = Field(default=16, description="Maximum number of concurrent workers")
    log_level: str = Field(default="INFO", description="Logging level")
    config_path: str | None = Field(default=None, description="Optional path to GLM-OCR config file")

    @classmethod
    @override
    def from_env(cls) -> AppConfig:
        """Load configuration from environment variables."""
        enable_layout_raw = os.getenv("GLMOCR_ENABLE_LAYOUT", "true").lower()

        return cls(
            vllm_api_url=get_env_or_throw("VLLM_API_URL"),
            vllm_api_key=os.getenv("VLLM_API_KEY"),
            vllm_model_name=get_env_or_throw("VLLM_MODEL_NAME"),
            request_timeout_seconds=int(os.getenv("GLMOCR_REQUEST_TIMEOUT_SECONDS", "300")),
            enable_layout=enable_layout_raw in {"1", "true", "yes", "on"},
            max_workers=int(os.getenv("GLMOCR_MAX_WORKERS", "16")),
            log_level=os.getenv("GLMOCR_LOG_LEVEL", "INFO"),
            config_path=os.getenv("GLMOCR_CONFIG_PATH"),
        )

    @override
    def __str__(self) -> str:
        return (
            f"AppConfig(\n"
            f"  vllm_api_url={self.vllm_api_url},\n"
            f"  vllm_api_key={log_secret(self.vllm_api_key) if self.vllm_api_key else None},\n"
            f"  vllm_model_name={self.vllm_model_name},\n"
            f"  request_timeout_seconds={self.request_timeout_seconds},\n"
            f"  enable_layout={self.enable_layout},\n"
            f"  max_workers={self.max_workers},\n"
            f"  log_level={self.log_level},\n"
            f"  config_path={log_secret(self.config_path) if self.config_path else None},\n"
            f")"
        )
