from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _image_to_b64_uri(path: Path, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _make_request(image_uri: str, model: str = "glm-ocr") -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please OCR this page."},
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ],
            }
        ],
        "temperature": 0.0,
    }


@pytest.fixture
def e2e_base_url() -> str:
    url = os.getenv("E2E_PROXY_URL")
    if not url:
        pytest.skip("Set E2E_PROXY_URL to run end-to-end tests.")
    return url


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_ocr_produces_text(e2e_base_url: str) -> None:
    payload = _make_request(_image_to_b64_uri(DATA_DIR / "ocr.png"))

    async with httpx.AsyncClient(base_url=e2e_base_url, timeout=300) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    assert len(content.strip()) > 0
    assert content.strip().lower().startswith("today is thursday")


# ---------------------------------------------------------------------------
# Image with no text
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_no_text_image_returns_200(e2e_base_url: str) -> None:
    """An image without readable text should still return 200 (possibly empty content)."""
    payload = _make_request(_image_to_b64_uri(DATA_DIR / "ocr_no_text.jpg", mime="image/jpeg"))

    async with httpx.AsyncClient(base_url=e2e_base_url, timeout=300) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert isinstance(content, str)


# ---------------------------------------------------------------------------
# Corrupted image data
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_corrupted_image_data(e2e_base_url: str) -> None:
    """Sending garbage bytes as a base64 image should not crash the service."""
    garbage_b64 = base64.b64encode(b"\x00\xff\xfe corrupted garbage").decode()
    payload = _make_request(f"data:image/png;base64,{garbage_b64}")

    async with httpx.AsyncClient(base_url=e2e_base_url, timeout=300) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 422
    assert "corrupted" in response.text.lower() or "not a valid image" in response.text.lower()


# ---------------------------------------------------------------------------
# Missing image (text-only request)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_missing_image_returns_400(e2e_base_url: str) -> None:
    payload = {
        "model": "glm-ocr",
        "messages": [{"role": "user", "content": "no image here"}],
    }

    async with httpx.AsyncClient(base_url=e2e_base_url, timeout=300) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert "image_url" in response.text.lower()


# ---------------------------------------------------------------------------
# Unsupported model
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_unsupported_model_returns_400(e2e_base_url: str) -> None:
    payload = _make_request(_image_to_b64_uri(DATA_DIR / "ocr.png"), model="does-not-exist")

    async with httpx.AsyncClient(base_url=e2e_base_url, timeout=300) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert "unsupported model" in response.text.lower()


# ---------------------------------------------------------------------------
# Malformed JSON body
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_malformed_json_returns_422(e2e_base_url: str) -> None:
    """A request missing required fields should be rejected by Pydantic validation."""
    payload: dict[str, Any] = {"model": "glm-ocr"}

    async with httpx.AsyncClient(base_url=e2e_base_url, timeout=300) as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 422
