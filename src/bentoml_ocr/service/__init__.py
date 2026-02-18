"""BentoML service definition for the GLM-OCR Docling-compatible proxy."""

import bentoml
from dcc_backend_common.logger import get_logger, init_logger

from bentoml_ocr.ocr_proxy.api import app
from bentoml_ocr.ocr_proxy.container import Container

logger = get_logger(__name__)

image = (
    bentoml.images.Image(python_version="3.12")
    .python_packages(
        "bentoml>=1.4",
        "dependency-injector>=4.48",
        "fastapi>=0.115",
        "httpx>=0.28",
        "uvicorn>=0.34",
        "pydantic>=2.10",
        "pyyaml>=6.0.2",
        "glmocr @ git+https://github.com/zai-org/GLM-OCR.git",
    )
    .system_packages("libgl1", "libglib2.0-0")
)


@bentoml.service(
    image=image,
    traffic={"timeout": 300},
)
@bentoml.asgi_app(app, path="/")
class GLMOCRProxy:
    """BentoML service wrapping the GLM-OCR proxy behind an ASGI FastAPI app."""

    layout_model = bentoml.models.HuggingFaceModel("PaddlePaddle/PP-DocLayoutV3_safetensors")

    def __init__(self) -> None:
        init_logger()
        container = Container()
        container.wire(modules=["bentoml_ocr.ocr_proxy.api"])
        config = container.config()
        logger.info("Service initialized", config=str(config))
