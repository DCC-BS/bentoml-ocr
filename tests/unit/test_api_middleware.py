"""Tests for API middleware: authentication, body size limits, request correlation, and health endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator

import httpx
import pytest
import pytest_asyncio

from bentoml_ocr.ocr_proxy import api


@pytest.fixture
def _enable_body_size_limit() -> Generator[None, None, None]:
    """Temporarily enable body size limit on the shared app."""
    api.app.state.max_body_size_bytes = 1024
    yield
    api.app.state.max_body_size_bytes = 50 * 1024 * 1024


@pytest_asyncio.fixture
async def client(_enable_body_size_limit: None) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Client against an app with body size limit middleware enabled."""
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


# ---------------------------------------------------------------------------
# Body size limit middleware
# ---------------------------------------------------------------------------


class TestBodySizeLimitMiddleware:
    @pytest.mark.asyncio
    async def test_rejects_oversized_body(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/v1/chat/completions",
            content=b"x" * 2048,
            headers={"Content-Length": "2048", "Content-Type": "application/json"},
        )
        assert response.status_code == 413
        assert "too large" in response.text.lower()


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------


class TestRequestIDMiddleware:
    @pytest.mark.asyncio
    async def test_generates_request_id_if_missing(self, app_client: httpx.AsyncClient) -> None:
        response = await app_client.get("/healthz")
        assert response.headers.get("x-request-id")

    @pytest.mark.asyncio
    async def test_echoes_provided_request_id(self, app_client: httpx.AsyncClient) -> None:
        response = await app_client.get("/healthz", headers={"X-Request-ID": "my-trace-123"})
        assert response.headers["x-request-id"] == "my-trace-123"


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_healthz_returns_ok(self, app_client: httpx.AsyncClient) -> None:
        response = await app_client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_readyz_returns_ok(self, app_client: httpx.AsyncClient) -> None:
        response = await app_client.get("/readyz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
