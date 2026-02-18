import bentoml
from dcc_backend_common.logger import get_logger, init_logger

from bentoml_ocr.ocr_proxy.api import app, set_backend_for_tests
from bentoml_ocr.ocr_proxy.backend import DefaultOCRBackend, build_openai_response
from bentoml_ocr.ocr_proxy.config import AppConfig

_build_openai_response = build_openai_response

logger = get_logger(__name__)

image = (
    bentoml.images.Image(python_version="3.12")
    .python_packages(
        "bentoml>=1.4",
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
    layout_model = bentoml.models.HuggingFaceModel("PaddlePaddle/PP-DocLayoutV3_safetensors")

    def __init__(self) -> None:
        init_logger()
        config = AppConfig.from_env()
        logger.info("Service initialized", config=str(config))
        set_backend_for_tests(DefaultOCRBackend(config))
