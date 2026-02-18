from __future__ import annotations

import os
from typing import Any

import httpx
import pytest


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_with_real_vllm(sample_openai_request: dict[str, Any]) -> None:
    base_url = os.getenv("E2E_PROXY_URL")
    if not base_url:
        pytest.skip("Set E2E_PROXY_URL to run end-to-end tests.")

    async with httpx.AsyncClient(base_url=base_url, timeout=300) as client:
        response = await client.post("/v1/chat/completions", json=sample_openai_request)

    assert response.status_code == 200
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    assert len(content.strip()) > 0
