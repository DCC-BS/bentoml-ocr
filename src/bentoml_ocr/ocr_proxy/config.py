from __future__ import annotations

import os
from dataclasses import dataclass

from bentoml_ocr.ocr_proxy.constants import FULL_MODEL_NAME


@dataclass
class RuntimeConfig:
    vllm_api_url: str
    vllm_model_name: str
    request_timeout_seconds: int
    enable_layout: bool
    max_workers: int
    log_level: str
    config_path: str | None


def load_runtime_config() -> RuntimeConfig:
    timeout = int(os.getenv("GLMOCR_REQUEST_TIMEOUT_SECONDS", "300"))
    max_workers = int(os.getenv("GLMOCR_MAX_WORKERS", "16"))
    enable_layout = os.getenv("GLMOCR_ENABLE_LAYOUT", "true").lower() == "true"
    config_path = os.getenv("GLMOCR_CONFIG_PATH")
    return RuntimeConfig(
        vllm_api_url=os.getenv("VLLM_API_URL", "http://localhost:8080/v1/chat/completions"),
        vllm_model_name=os.getenv("VLLM_MODEL_NAME", FULL_MODEL_NAME),
        request_timeout_seconds=timeout,
        enable_layout=enable_layout,
        max_workers=max_workers,
        log_level=os.getenv("GLMOCR_LOG_LEVEL", "INFO"),
        config_path=config_path,
    )
