"""Constants for supported OCR model identifiers."""

from enum import StrEnum


class ModelName(StrEnum):
    """Supported model names exposed by this proxy."""

    FULL = "glm-ocr"
    RAW = "glm-ocr-raw"


FULL_MODEL_NAME = ModelName.FULL
RAW_MODEL_NAME = ModelName.RAW
