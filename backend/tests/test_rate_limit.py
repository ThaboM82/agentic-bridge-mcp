"""
Tests for the application-level in-memory rate limiter and FastAPI dependency.

These tests cover sliding-window behavior, key isolation, reset behavior,
response metadata, Retry-After handling, and FastAPI integration.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.core.exceptions import RateLimitError
from app.core.rate_limit import (
    InMemoryRateLimiter,
    RateLimitDecision,
    rate_limit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def limiter() -> InMemoryRateLimiter:
    """Return a fresh in-memory rate limiter for each test."""
    return InMemoryRateLimiter()


@pytest.fixture
def app() -> FastAPI:
    """Return a fresh FastAPI application for each test."""
    return FastAPI()


# ---------------------------------------------------------------------------
# RateLimitDecision
# ---------------------------------------------------------------------------


def test_rate_limit_decision_defaults():
    decision = RateLimitDecision(
        allowed=True,
        limit=10,
        remaining=9,
        window_seconds=60,
    )

    assert decision.allowed is True
    assert decision.limit == 10
    assert decision.remaining == 9
    assert decision.window_seconds == 60
    assert decision.retry_after_seconds is None
    assert decision.reset_at is None


# ---------------------------------------------------------------------------
# InMemoryRateLimiter
# ---------------------------------------------------------------------------


def test_allows_requests_below_limit(
    limiter: InMemoryRateLimiter,
):
    first = limiter.check(
        "test-key",
        limit=2,
        window_seconds=60,
    )

    second = limiter.check(
        "test-key",
        limit=2,
        window_seconds=60,
    )

    assert first.allowed is True
    assert first.remaining == 1
    assert first.limit == 2
    assert first.window_seconds == 60

    assert second.allowed is True
    assert second.remaining == 0
    assert second.limit == 2
    assert second.window_seconds == 60


def test_blocks_requests_above_limit(
    limiter: InMemoryRateLimiter,
):
    limiter.check(
        "test-key",
        limit=1,
        window_seconds=60,
    )

    blocked = limiter.check(
        "test-key",
        limit=1,
        window_seconds=60,
    )

    assert blocked.allowed is False
    assert blocked.remaining == 0
    assert blocked.retry_after_seconds >= 1


def test_different_keys_are_independent(
    limiter: InMemoryRateLimiter,
):
    limiter.check(
        "key-a",
        limit=1,
        window_seconds=60,
    )

    result = limiter.check(
        "key-b",
        limit=1,
        window_seconds=60,
    )

    assert result.allowed is True
    assert result.remaining == 0


def test_different_windows_are_independent(
    limiter: InMemoryRateLimiter,
):
    limiter.check(
        "test-key",
        limit=1,
        window_seconds=60,
    )

    result = limiter.check(
        "test-key",
        limit=1,
        window_seconds=30,
    )

    assert result.allowed is True
    assert result.remaining == 0


def test_reset_clears_key(
    limiter: InMemoryRateLimiter,
):
    limiter.check(
        "test-key",
        limit=1,
        window_seconds=60,
    )

    limiter.reset("test-key")

    result = limiter.check(
        "test-key",
        limit=1,
        window_seconds=60,
    )

    assert result.allowed is True
    assert result.remaining == 0


def test_reset_unknown_key_is_safe(
    limiter: InMemoryRateLimiter,
):
    limiter.reset("unknown-key")

    result = limiter.check(
        "unknown-key",
        limit=1,
        window_seconds=60,
    )

    assert result.allowed is True


def test_reset_all_clears_every_key(
    limiter: InMemoryRateLimiter,
):
    limiter.check(
        "key-a",
        limit=1,
        window_seconds=60,
    )

    limiter.check(
        "key-b",
        limit=1,
        window_seconds=60,
    )

    limiter.reset_all()

    first = limiter.check(
        "key-a",
        limit=1,
        window_seconds=60,
    )

    second = limiter.check(
        "key-b",
        limit=1,
        window_seconds=60,
    )

    assert first.allowed is True
    assert second.allowed is True


def test_invalid_limit_rejected(
    limiter: InMemoryRateLimiter,
):
    with pytest.raises(ValueError):
        limiter.check(
            "test-key",
            limit=0,
            window_seconds=60,
        )


def test_invalid_window_rejected(
    limiter: InMemoryRateLimiter,
):
    with pytest.raises(ValueError):
        limiter.check(
            "test-key",
            limit=1,
            window_seconds=0,
        )


def test_empty_key_rejected(
    limiter: InMemoryRateLimiter,
):
    with pytest.raises(ValueError):
        limiter.check(
            "",
            limit=1,
            window_seconds=60,
        )


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def test_rate_limit_dependency_returns_429(
    app: FastAPI,
):
    @app.get(
        "/limited",
        dependencies=[
            rate_limit(
                limit=1,
                window_seconds=60,
                scope="test:limited",
            )
        ],
    )
    async def limited() -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(app)

    first = client.get("/limited")
    second = client.get("/limited")

    assert first.status_code == 200
    assert first.json() == {"status": "ok"}

    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) >= 1
    assert second.headers["X-RateLimit-Remaining"] == "0"
    assert second.headers["X-RateLimit-Limit"] == "1"

    body = second.json()

    assert body["error"] == "rate_limit_exceeded"
    assert "rate limit exceeded" in body["detail"].lower()


def test_rate_limit_dependency_isolates_clients(
    app: FastAPI,
):
    @app.get(
        "/limited",
        dependencies=[
            rate_limit(
                limit=1,
                window_seconds=60,
                scope="test:isolated",
            )
        ],
    )
    async def limited() -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(app)

    first_response = client.get(
        "/limited",
        headers={"X-Forwarded-For": "203.0.113.10"},
    )

    second_response = client.get(
        "/limited",
        headers={"X-Forwarded-For": "203.0.113.10"},
    )

    third_response = client.get(
        "/limited",
        headers={"X-Forwarded-For": "203.0.113.11"},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 429
    assert third_response.status_code == 200


def test_rate_limit_dependency_uses_authenticated_user(
    app: FastAPI,
):
    async def fake_user_resolver(
        request: Request,
    ) -> str:
        return "user-123"

    @app.get(
        "/user-limited",
        dependencies=[
            rate_limit(
                limit=1,
                window_seconds=60,
                scope="test:user",
                key_resolver=fake_user_resolver,
            )
        ],
    )
    async def user_limited() -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(app)

    first = client.get("/user-limited")
    second = client.get("/user-limited")

    assert first.status_code == 200
    assert second.status_code == 429


def test_rate_limit_dependency_can_be_disabled(
    app: FastAPI,
):
    @app.get(
        "/unlimited",
        dependencies=[
            rate_limit(
                limit=1,
                window_seconds=60,
                scope="test:disabled",
                enabled=False,
            )
        ],
    )
    async def unlimited() -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(app)

    for _ in range(5):
        response = client.get("/unlimited")

        assert response.status_code == 200


def test_rate_limit_dependency_includes_rate_limit_headers(
    app: FastAPI,
):
    @app.get(
        "/headers",
        dependencies=[
            rate_limit(
                limit=3,
                window_seconds=60,
                scope="test:headers",
            )
        ],
    )
    async def headers_endpoint() -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(app)

    first = client.get("/headers")
    second = client.get("/headers")
    third = client.get("/headers")
    fourth = client.get("/headers")

    assert first.headers["X-RateLimit-Limit"] == "3"
    assert first.headers["X-RateLimit-Remaining"] == "2"

    assert second.headers["X-RateLimit-Remaining"] == "1"
    assert third.headers["X-RateLimit-Remaining"] == "0"

    assert fourth.status_code == 429
    assert fourth.headers["X-RateLimit-Remaining"] == "0"
    assert int(fourth.headers["Retry-After"]) >= 1
