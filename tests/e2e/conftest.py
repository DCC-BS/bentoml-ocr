"""Shared fixtures for end-to-end tests."""

import os
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def pytest_collection_modifyitems(config, items):
    """Skip e2e tests when DOCLING_SERVE_URL is not set."""
    if os.environ.get("DOCLING_SERVE_URL"):
        return
    skip = pytest.mark.skip(reason="DOCLING_SERVE_URL not set")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)


@pytest.fixture()
def docling_serve_url():
    return os.environ["DOCLING_SERVE_URL"].rstrip("/")


@pytest.fixture()
def ocr_image_path():
    return DATA_DIR / "ocr.png"


@pytest.fixture()
def ocr_no_text_image_path():
    return DATA_DIR / "ocr_no_text.jpg"
