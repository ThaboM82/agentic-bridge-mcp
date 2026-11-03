"""
Comprehensive tests for the FastAPI request-context and security-header
middleware.

These tests cover request correlation, safe upstream header handling,
processing-time telemetry, security headers, error responses, HTTP methods,
response integrity, and middleware behavior across multiple requests.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.core.middleware import RequestContextMiddleware


REQUEST_ID_PATTERN = re.compile(
    r"^req_[A-Za-z0-9_-]+$"
)

MAX_REQUEST_ID_LENGTH = 128


@pytest.fixture
def app() -> FastAPI:
    """Return a FastAPI application with request-context middleware."""
    application = FastAPI()
    application.add_middleware(
        RequestContextMiddleware
    )

    @application.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/echo-request-id")
    async def echo_request_id(
        request: Request,
    ) -> dict[str, str | None]:
        return {
            "request_id": request.headers.get(
                "X-Request-ID"
            )
        }

    @application.post("/echo-json")
    async def echo_json(
        request: Request,
    ) -> JSONResponse:
        payload: dict[str, Any] = (
            await request.json()
        )

        return JSONResponse(
            {
                "received": payload,
                "request_id": request.headers.get(
                    "X-Request-ID"
                ),
            }
        )

    @application.get("/error")
    async def error() -> None:
        raise RuntimeError(
            "Unexpected internal error"
        )

    @application.get("/validation-error")
    async def validation_error() -> None:
        raise ValueError("Invalid input")

    @application.get("/slow")
    async def slow() -> dict[str, str]:
        import asyncio

        await asyncio.sleep(0.01)

        return {"status": "slow"}

    return application


@pytest.fixture
def client(
    app: FastAPI,
) -> TestClient:
    """Return a test client for the configured FastAPI application."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Request ID generation
# ---------------------------------------------------------------------------


def test_request_id_is_generated(
    client: TestClient,
):
    response = client.get("/ping")

    assert response.status_code == 200
    assert response.headers[
        "X-Request-ID"
    ].startswith("req_")
    assert REQUEST_ID_PATTERN.match(
        response.headers["X-Request-ID"]
    )
    assert (
        "X-Process-Time"
        in response.headers
    )


def test_request_id_has_expected_prefix(
    client: TestClient,
):
    response = client.get("/ping")

    assert response.headers[
        "X-Request-ID"
    ].startswith("req_")


def test_request_id_matches_safe_pattern(
    client: TestClient,
):
    response = client.get("/ping")

    request_id = response.headers[
        "X-Request-ID"
    ]

    assert REQUEST_ID_PATTERN.match(request_id)


def test_request_id_is_within_length_limit(
    client: TestClient,
):
    response = client.get("/ping")

    request_id = response.headers[
        "X-Request-ID"
    ]

    assert len(request_id) <= MAX_REQUEST_ID_LENGTH


def test_request_ids_are_unique_across_requests(
    client: TestClient,
):
    first_response = client.get("/ping")
    second_response = client.get("/ping")

    first_request_id = first_response.headers[
        "X-Request-ID"
    ]

    second_request_id = second_response.headers[
        "X-Request-ID"
    ]

    assert first_request_id != second_request_id


def test_request_ids_are_unique_across_methods(
    client: TestClient,
):
    get_response = client.get("/ping")
    post_response = client.post(
        "/echo-json",
        json={"value": 1},
    )

    assert (
        get_response.headers["X-Request-ID"]
        != post_response.headers["X-Request-ID"]
    )


# ---------------------------------------------------------------------------
# Upstream request ID handling
# ---------------------------------------------------------------------------


def test_valid_upstream_request_id_is_preserved(
    client: TestClient,
):
    response = client.get(
        "/ping",
        headers={
            "X-Request-ID": "external-request-123"
        },
    )

    assert response.status_code == 200
    assert (
        response.headers["X-Request-ID"]
        == "external-request-123"
    )


def test_valid_uuid_request_id_is_preserved(
    client: TestClient,
):
    upstream_request_id = (
        "123e4567-e89b-12d3-a456-426614174000"
    )

    response = client.get(
        "/ping",
        headers={
            "X-Request-ID": upstream_request_id
        },
    )

    assert response.status_code == 200
    assert (
        response.headers["X-Request-ID"]
        == upstream_request_id
    )


def test_whitespace_request_id_is_trimmed(
    client: TestClient,
):
    response = client.get(
        "/ping",
        headers={
            "X-Request-ID": "  external-request-456  "
        },
    )

    assert response.status_code == 200
    assert (
        response.headers["X-Request-ID"]
        == "external-request-456"
    )


def test_invalid_upstream_request_id_is_replaced(
    client: TestClient,
):
    response = client.get(
        "/ping",
        headers={
            "X-Request-ID": (
                "invalid request id with spaces"
            )
        },
    )

    assert response.status_code == 200
    assert response.headers[
        "X-Request-ID"
    ].startswith("req_")


def test_empty_upstream_request_id_is_replaced(
    client: TestClient,
):
    response = client.get(
        "/ping",
        headers={"X-Request-ID": ""},
    )

    assert response.status_code == 200
    assert response.headers[
        "X-Request-ID"
    ].startswith("req_")


def test_none_upstream_request_id_is_replaced(
    client: TestClient,
):
    response = client.get(
        "/ping",
        headers={"X-Request-ID": None},
    )

    assert response.status_code == 200
    assert response.headers[
        "X-Request-ID"
    ].startswith("req_")


def test_oversized_request_id_is_replaced(
    client: TestClient,
):
    oversized_request_id = "x" * 500

    response = client.get(
        "/ping",
        headers={
            "X-Request-ID": oversized_request_id
        },
    )

    assert response.status_code == 200
    assert (
        response.headers["X-Request-ID"]
        != oversized_request_id
    )
    assert response.headers[
        "X-Request-ID"
    ].startswith("req_")


def test_request_id_with_control_characters_is_replaced(
    client: TestClient,
):
    unsafe_request_id = (
        "request\x00id\x1fwith\x7fcontrols"
    )

    response = client.get(
        "/ping",
        headers={
            "X-Request-ID": unsafe_request_id
        },
    )

    assert response.status_code == 200
    assert (
        response.headers["X-Request-ID"]
        != unsafe_request_id
    )
    assert response.headers[
        "X-Request-ID"
    ].startswith("req_")


def test_request_id_with_newlines_is_replaced(
    client: TestClient,
):
    unsafe_request_id = (
        "request\nid\r\nwith\rnewlines"
    )

    response = client.get(
        "/ping",
        headers={
            "X-Request-ID": unsafe_request_id
        },
    )

    assert response.status_code == 200
    assert response.headers[
        "X-Request-ID"
    ].startswith("req_")


# ---------------------------------------------------------------------------
# Processing-time telemetry
# ---------------------------------------------------------------------------


def test_process_time_header_is_present(
    client: TestClient,
):
    response = client.get("/ping")

    assert response.status_code == 200
    assert (
        "X-Process-Time"
        in response.headers
    )

    process_time = float(
        response.headers["X-Process-Time"]
    )

    assert process_time >= 0


def test_process_time_is_finite(
    client: TestClient,
):
    response = client.get("/ping")

    process_time = float(
        response.headers["X-Process-Time"]
    )

    assert process_time == process_time
    assert process_time != float("inf")


def test_process_time_is_numeric(
    client: TestClient,
):
    response = client.get("/ping")

    process_time = response.headers[
        "X-Process-Time"
    ]

    assert float(process_time) >= 0


def test_process_time_is_present_on_slow_response(
    client: TestClient,
):
    response = client.get("/slow")

    assert response.status_code == 200
    assert (
        "X-Process-Time"
        in response.headers
    )

    process_time = float(
        response.headers["X-Process-Time"]
    )

    assert process_time > 0


def test_process_time_is_present_on_error_response(
    client: TestClient,
):
    response = client.get("/error")

    assert response.status_code >= 400
    assert (
        "X-Process-Time"
        in response.headers
    )


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


def test_security_headers_are_added(
    client: TestClient,
):
    response = client.get("/ping")

    assert response.status_code == 200
    assert (
        response.headers[
            "X-Content-Type-Options"
        ]
        == "nosniff"
    )
    assert (
        response.headers["X-Frame-Options"]
        == "DENY"
    )
    assert (
        response.headers["Referrer-Policy"]
        == "strict-origin-when-cross-origin"
    )


def test_security_headers_are_added_to_errors(
    client: TestClient,
):
    response = client.get("/error")

    assert response.status_code >= 400
    assert (
        response.headers[
            "X-Content-Type-Options"
        ]
        == "nosniff"
    )
    assert (
        response.headers["X-Frame-Options"]
        == "DENY"
    )
    assert (
        response.headers["Referrer-Policy"]
        == "strict-origin-when-cross-origin"
    )
    assert (
        "X-Request-ID"
        in response.headers
    )
    assert (
        "X-Process-Time"
        in response.headers
    )


def test_security_headers_are_added_to_post_requests(
    client: TestClient,
):
    response = client.post(
        "/echo-json",
        json={"value": 1},
    )

    assert response.status_code == 200
    assert (
        response.headers[
            "X-Content-Type-Options"
        ]
        == "nosniff"
    )
    assert (
        response.headers["X-Frame-Options"]
        == "DENY"
    )
    assert (
        response.headers["Referrer-Policy"]
        == "strict-origin-when-cross-origin"
    )


def test_security_headers_are_added_to_validation_errors(
    client: TestClient,
):
    response = client.get(
        "/validation-error"
    )

    assert response.status_code >= 400
    assert (
        response.headers[
            "X-Content-Type-Options"
        ]
        == "nosniff"
    )
    assert (
        response.headers["X-Frame-Options"]
        == "DENY"
    )
    assert (
        response.headers["Referrer-Policy"]
        == "strict-origin-when-cross-origin"
    )


# ---------------------------------------------------------------------------
# Middleware integration
# ---------------------------------------------------------------------------


def test_middleware_does_not_break_normal_responses(
    client: TestClient,
):
    response = client.get("/ping")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_middleware_adds_headers_to_all_routes(
    client: TestClient,
):
    ping_response = client.get("/ping")
    echo_response = client.get(
        "/echo-request-id"
    )

    assert (
        "X-Request-ID"
        in ping_response.headers
    )
    assert (
        "X-Request-ID"
        in echo_response.headers
    )

    assert (
        ping_response.headers[
            "X-Request-ID"
        ]
        != echo_response.headers[
            "X-Request-ID"
        ]
    )


def test_middleware_preserves_request_body(
    client: TestClient,
):
    payload = {
        "name": "dashboard",
        "features": [
            "authentication",
            "audit_logs",
        ],
    }

    response = client.post(
        "/echo-json",
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["received"] == payload


def test_middleware_adds_request_id_to_post_response(
    client: TestClient,
):
    response = client.post(
        "/echo-json",
        json={"value": 1},
        headers={
            "X-Request-ID": "external-request-789"
        },
    )

    assert response.status_code == 200
    assert (
        response.json()["request_id"]
        == "external-request-789"
    )
    assert (
        response.headers["X-Request-ID"]
        == "external-request-789"
    )


def test_error_response_does_not_leak_internal_details(
    client: TestClient,
):
    response = client.get("/error")

    assert response.status_code >= 400

    response_text = response.text.lower()

    assert (
        "unexpected internal error"
        not in response_text
    )
    assert "traceback" not in response_text
    assert "runtimeerror" not in response_text


def test_validation_error_response_does_not_leak_internal_details(
    client: TestClient,
):
    response = client.get(
        "/validation-error"
    )

    assert response.status_code >= 400

    response_text = response.text.lower()

    assert "invalid input" not in response_text
    assert "traceback" not in response_text
    assert "valueerror" not in response_text


def test_middleware_handles_multiple_requests(
    client: TestClient,
):
    for _ in range(5):
        response = client.get("/ping")

        assert response.status_code == 200
        assert (
            "X-Request-ID"
            in response.headers
        )
        assert (
            "X-Process-Time"
            in response.headers
        )
        assert (
            response.headers[
                "X-Content-Type-Options"
            ]
            == "nosniff"
        )


def test_middleware_preserves_response_json_on_error(
    client: TestClient,
):
    response = client.get("/error")

    assert response.status_code >= 400

    response_json = response.json()

    assert isinstance(response_json, dict)
    assert "detail" in response_json
