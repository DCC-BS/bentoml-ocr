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


@pytest.mark.asyncio
async def test_concurrent_raw_requests(
    wire_container: Container,
    app_client: httpx.AsyncClient,
    sample_openai_request: dict[str, Any],
) -> None:
    """Multiple concurrent raw-passthrough requests should all succeed independently."""
    backend = FakeBackend()
    sample_openai_request["model"] = "glm-ocr-raw"
    n = 10

    with wire_container.backend.override(backend):
        tasks = [app_client.post("/v1/chat/completions", json=sample_openai_request) for _ in range(n)]
        responses = await asyncio.gather(*tasks)

    assert all(r.status_code == 200 for r in responses)
    assert backend.raw_calls == n


@pytest.mark.asyncio
async def test_mixed_concurrent_requests(
    wire_container: Container,
    app_client: httpx.AsyncClient,
    sample_openai_request: dict[str, Any],
) -> None:
    """Full and raw requests running concurrently should not interfere."""
    backend = FakeBackend()

    full_req = dict(sample_openai_request)
    full_req["model"] = "glm-ocr"

    raw_req = dict(sample_openai_request)
    raw_req["model"] = "glm-ocr-raw"

    with wire_container.backend.override(backend):
        tasks = [
            *[app_client.post("/v1/chat/completions", json=full_req) for _ in range(5)],
            *[app_client.post("/v1/chat/completions", json=raw_req) for _ in range(5)],
        ]
        responses = await asyncio.gather(*tasks)

    assert all(r.status_code == 200 for r in responses)
    assert backend.full_calls == 5
    assert backend.raw_calls == 5
