"""BentoML service definition for the GLM-OCR Docling-compatible proxy."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import bentoml
from dcc_backend_common.logger import get_logger, init_logger
from fastapi import FastAPI

from bentoml_ocr.ocr_proxy.api import app, configure_app_state
from bentoml_ocr.ocr_proxy.container import Container

logger = get_logger(__name__)

image = (
    bentoml.images.Image(python_version="3.13")
    .python_packages(
        "bentoml>=1.4",
        "dependency-injector>=4.48",
        "fastapi>=0.115",
        "httpx>=0.28",
        "uvicorn>=0.34",
        "pydantic>=2.10",
        "pyyaml>=6.0.2",
        "tenacity>=9.0",
        "glmocr @ git+https://github.com/zai-org/GLM-OCR.git",
    )
    .system_packages("libgl1", "libglib2.0-0")
)

_container: Container | None = None


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    if _container is not None:
        backend = _container.backend()
        await backend.close()
        logger.info("Service shutdown complete")


app.router.lifespan_context = _lifespan


@bentoml.service(
    image=image,
    traffic={"timeout": 300},
)
@bentoml.asgi_app(app, path="/")
class GLMOCRProxy:
    """BentoML service wrapping the GLM-OCR proxy behind an ASGI FastAPI app."""

    layout_model = bentoml.models.HuggingFaceModel("PaddlePaddle/PP-DocLayoutV3_safetensors")

    def __init__(self) -> None:
        global _container  # noqa: PLW0603
        init_logger()
        _container = Container()
        _container.wire(modules=["bentoml_ocr.ocr_proxy.api"])
        config = _container.config()
        configure_app_state(
            max_body_size=config.max_body_size_bytes,
        )
        logger.info("Service initialized", config=str(config))
