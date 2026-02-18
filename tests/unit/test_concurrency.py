"""Tests verifying correct behaviour under concurrent requests."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from bentoml_ocr.ocr_proxy.container import Container
from tests.conftest import FakeBackend


@pytest.mark.asyncio
async def test_concurrent_full_requests(
    wire_container: Container,
    app_client: httpx.AsyncClient,
    sample_openai_request: dict[str, Any],
) -> None:
    """Multiple concurrent full-pipeline requests should all succeed independently."""
    backend = FakeBackend()
    n = 10

    with wire_container.backend.override(backend):
        tasks = [app_client.post("/v1/chat/completions", json=sample_openai_request) for _ in range(n)]
        responses = await asyncio.gather(*tasks)

    assert all(r.status_code == 200 for r in responses)
    assert backend.full_calls == n
    models = {r.json()["model"] for r in responses}
    assert models == {"glm-ocr"}
