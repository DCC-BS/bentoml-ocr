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
    max_workers: int = Field(default=16, description="Maximum number of concurrent workers for OCR parsing thread pool")
    log_level: str = Field(default="INFO", description="Logging level")
    config_path: str | None = Field(default=None, description="Optional path to GLM-OCR config file")
    proxy_api_key: str | None = Field(
        default=None, description="API key for authenticating proxy clients (disabled if unset)"
    )
    max_body_size_bytes: int = Field(default=50 * 1024 * 1024, description="Maximum request body size in bytes")
    max_http_connections: int = Field(default=100, description="Maximum concurrent HTTP connections to vLLM")
    max_keepalive_connections: int = Field(default=20, description="Maximum keep-alive HTTP connections to vLLM")
    retry_max_attempts: int = Field(default=3, description="Maximum retry attempts for transient vLLM errors")
    retry_backoff_base_seconds: float = Field(default=0.5, description="Base delay in seconds for exponential backoff")
    retry_backoff_max_seconds: float = Field(
        default=8.0, description="Maximum delay in seconds for exponential backoff"
    )

    def apply_env(self) -> None:
        """Write required ``GLMOCR_OCR_*`` env vars derived from this config.

        The GlmOcr SDK in *selfhosted* mode reads connection settings from
        ``GLMOCR_OCR_API_URL`` and ``GLMOCR_OCR_MODEL``.  These must be
        present **before** the parser is instantiated.

        Note: ``GLMOCR_OCR_API_KEY`` is intentionally **not** persisted here.
        It is set temporarily during parser initialisation and cleared
        immediately afterwards (see ``DefaultOCRBackend._init_glmocr_parser``).
        """
        os.environ["GLMOCR_OCR_API_URL"] = self.vllm_api_url
        os.environ["GLMOCR_OCR_MODEL"] = self.vllm_model_name

    @classmethod
    @override
    def from_env(cls) -> AppConfig:
        """Load configuration from environment variables."""
        enable_layout_raw = os.getenv("GLMOCR_ENABLE_LAYOUT", "true").lower()

        config = cls(
            vllm_api_url=get_env_or_throw("VLLM_API_URL"),
            vllm_api_key=os.getenv("VLLM_API_KEY"),
            vllm_model_name=get_env_or_throw("VLLM_MODEL_NAME"),
            request_timeout_seconds=int(os.getenv("GLMOCR_REQUEST_TIMEOUT_SECONDS", "300")),
            enable_layout=enable_layout_raw in {"1", "true", "yes", "on"},
            max_workers=int(os.getenv("GLMOCR_MAX_WORKERS", "16")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            config_path=os.getenv("GLMOCR_CONFIG_PATH"),
            proxy_api_key=os.getenv("PROXY_API_KEY"),
            max_body_size_bytes=int(os.getenv("MAX_BODY_SIZE_BYTES", str(50 * 1024 * 1024))),
            max_http_connections=int(os.getenv("MAX_HTTP_CONNECTIONS", "100")),
            max_keepalive_connections=int(os.getenv("MAX_KEEPALIVE_CONNECTIONS", "20")),
            retry_max_attempts=int(os.getenv("RETRY_MAX_ATTEMPTS", "3")),
            retry_backoff_base_seconds=float(os.getenv("RETRY_BACKOFF_BASE_SECONDS", "0.5")),
            retry_backoff_max_seconds=float(os.getenv("RETRY_BACKOFF_MAX_SECONDS", "8.0")),
        )
        config.apply_env()
        return config

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
            f"  proxy_api_key={log_secret(self.proxy_api_key) if self.proxy_api_key else None},\n"
            f"  max_body_size_bytes={self.max_body_size_bytes},\n"
            f"  max_http_connections={self.max_http_connections},\n"
            f"  max_keepalive_connections={self.max_keepalive_connections},\n"
            f"  retry_max_attempts={self.retry_max_attempts},\n"
            f"  retry_backoff_base_seconds={self.retry_backoff_base_seconds},\n"
            f"  retry_backoff_max_seconds={self.retry_backoff_max_seconds},\n"
            f")"
        )
