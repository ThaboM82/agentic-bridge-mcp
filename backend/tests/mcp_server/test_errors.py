"""
Comprehensive tests for MCP server error handling and sanitization.

These tests cover error status mapping, stable error codes, safe user-facing
messages, retryability, retry-after metadata, message and metadata
sanitization, tool-result redaction, structured error responses, and the MCP
tool error boundary.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.core.exceptions import (
    AppError,
    AuthenticationError,
    AuthorizationError,
    ExternalProviderError,
    NotFoundError,
    ProviderTimeoutError,
    RateLimitError,
    ValidationError,
    WorkflowStateError,
)
from app.mcp_server.errors import (
    build_error_response,
    get_error_code,
    get_error_status_code,
    get_retry_after_seconds,
    get_safe_error_message,
    is_retryable_error,
    mcp_tool_error_boundary,
    sanitize_tool_arguments,
    sanitize_tool_metadata,
    sanitize_tool_result,
    sanitize_error_message,
)


# ---------------------------------------------------------------------------
# Error status mapping
# ---------------------------------------------------------------------------


def test_error_status_codes():
    assert (
        get_error_status_code(
            ValidationError("bad")
        )
        == 422
    )
    assert (
        get_error_status_code(
            AuthenticationError("no")
        )
        == 401
    )
    assert (
        get_error_status_code(
            AuthorizationError("no")
        )
        == 403
    )
    assert (
        get_error_status_code(
            NotFoundError("missing")
        )
        == 404
    )
    assert (
        get_error_status_code(
            RateLimitError("slow")
        )
        == 429
    )
    assert (
        get_error_status_code(
            ProviderTimeoutError("timeout")
        )
        == 504
    )
    assert (
        get_error_status_code(
            ExternalProviderError("down")
        )
        == 502
    )
    assert (
        get_error_status_code(
            WorkflowStateError("bad state")
        )
        == 400
    )
    assert (
        get_error_status_code(
            RuntimeError("boom")
        )
        == 500
    )


@pytest.mark.parametrize(
    "error,expected_status",
    [
        (ValidationError("bad"), 422),
        (AuthenticationError("no"), 401),
        (AuthorizationError("no"), 403),
        (NotFoundError("missing"), 404),
        (RateLimitError("slow"), 429),
        (ProviderTimeoutError("timeout"), 504),
        (ExternalProviderError("down"), 502),
        (WorkflowStateError("bad state"), 400),
        (RuntimeError("boom"), 500),
        (ValueError("boom"), 500),
        (KeyError("missing"), 500),
    ],
)
def test_error_status_mapping(
    error: Exception,
    expected_status: int,
):
    assert (
        get_error_status_code(error)
        == expected_status
    )


def test_app_error_defaults_to_server_error():
    assert (
        get_error_status_code(
            AppError("failure")
        )
        == 500
    )


# ---------------------------------------------------------------------------
# Error-code mapping
# ---------------------------------------------------------------------------


def test_error_codes():
    assert (
        get_error_code(
            ValidationError("bad")
        )
        == "validation_error"
    )
    assert (
        get_error_code(
            AuthenticationError("no")
        )
        == "authentication_error"
    )
    assert (
        get_error_code(
            AuthorizationError("no")
        )
        == "authorization_error"
    )
    assert (
        get_error_code(
            NotFoundError("missing")
        )
        == "not_found"
    )
    assert (
        get_error_code(
            RateLimitError("slow")
        )
        == "rate_limit_exceeded"
    )
    assert (
        get_error_code(
            ProviderTimeoutError("timeout")
        )
        == "provider_timeout"
    )
    assert (
        get_error_code(
            ExternalProviderError("down")
        )
        == "external_provider_error"
    )
    assert (
        get_error_code(
            WorkflowStateError("bad state")
        )
        == "workflow_error"
    )
    assert (
        get_error_code(
            RuntimeError("boom")
        )
        == "internal_error"
    )


@pytest.mark.parametrize(
    "error,expected_code",
    [
        (ValidationError("bad"), "validation_error"),
        (
            AuthenticationError("no"),
            "authentication_error",
        ),
        (
            AuthorizationError("no"),
            "authorization_error",
        ),
        (NotFoundError("missing"), "not_found"),
        (
            RateLimitError("slow"),
            "rate_limit_exceeded",
        ),
        (
            ProviderTimeoutError("timeout"),
            "provider_timeout",
        ),
        (
            ExternalProviderError("down"),
            "external_provider_error",
        ),
        (
            WorkflowStateError("bad state"),
            "workflow_error",
        ),
        (RuntimeError("boom"), "internal_error"),
        (ValueError("boom"), "internal_error"),
        (KeyError("missing"), "internal_error"),
    ],
)
def test_error_code_mapping(
    error: Exception,
    expected_code: str,
):
    assert (
        get_error_code(error)
        == expected_code
    )


def test_app_error_uses_custom_code():
    exc = AppError(
        "Custom failure",
        code="custom_failure",
    )

    assert (
        get_error_code(exc)
        == "custom_failure"
    )


def test_error_codes_are_stable():
    assert (
        get_error_code(
            ValidationError("one")
        )
        == get_error_code(
            ValidationError("two")
        )
    )


# ---------------------------------------------------------------------------
# Safe error messages
# ---------------------------------------------------------------------------


def test_safe_error_messages():
    assert (
        get_safe_error_message(
            ProviderTimeoutError("secret")
        )
        == (
            "The requested operation timed out "
            "while communicating with an "
            "external provider."
        )
    )

    assert (
        get_safe_error_message(
            ExternalProviderError("secret")
        )
        == (
            "An external provider is "
            "temporarily unavailable."
        )
    )

    assert (
        get_safe_error_message(
            ValidationError("Invalid brief")
        )
        == "Invalid brief"
    )

    assert (
        get_safe_error_message(
            RuntimeError("internal secret")
        )
        == (
            "The requested MCP operation "
            "failed due to an unexpected "
            "server error."
        )
    )


def test_safe_error_message_hides_provider_error_details():
    message = get_safe_error_message(
        ProviderTimeoutError(
            "secret-provider-internal-detail"
        )
    )

    assert (
        "secret-provider-internal-detail"
        not in message
    )


def test_safe_error_message_hides_internal_error_details():
    message = get_safe_error_message(
        RuntimeError(
            "database password is hunter2"
        )
    )

    assert "hunter2" not in message
    assert "database" not in message


def test_safe_error_message_preserves_validation_details():
    message = get_safe_error_message(
        ValidationError(
            "Brief text is required."
        )
    )

    assert (
        message
        == "Brief text is required."
    )


@pytest.mark.parametrize(
    "error",
    [
        ValidationError("Invalid input."),
        AuthenticationError("Authentication required."),
        AuthorizationError("Access denied."),
        NotFoundError("Resource not found."),
        RateLimitError("Too many requests."),
        WorkflowStateError("Invalid workflow state."),
    ],
)
def test_safe_error_message_preserves_expected_domain_messages(
    error: Exception,
):
    assert (
        get_safe_error_message(error)
        == str(error)
    )


# ---------------------------------------------------------------------------
# Retryability
# ---------------------------------------------------------------------------


def test_retryable_errors():
    assert is_retryable_error(
        ProviderTimeoutError("timeout")
    )
    assert is_retryable_error(
        ExternalProviderError("down")
    )
    assert is_retryable_error(
        RateLimitError("slow")
    )
    assert not is_retryable_error(
        ValidationError("bad")
    )


@pytest.mark.parametrize(
    "error,expected_retryable",
    [
        (ProviderTimeoutError("timeout"), True),
        (ExternalProviderError("down"), True),
        (RateLimitError("slow"), True),
        (ValidationError("bad"), False),
        (AuthenticationError("no"), False),
        (AuthorizationError("no"), False),
        (NotFoundError("missing"), False),
        (WorkflowStateError("bad state"), False),
        (RuntimeError("boom"), False),
    ],
)
def test_retryable_error_classification(
    error: Exception,
    expected_retryable: bool,
):
    assert (
        is_retryable_error(error)
        is expected_retryable
    )


# ---------------------------------------------------------------------------
# Retry-after metadata
# ---------------------------------------------------------------------------


def test_retry_after():
    assert (
        get_retry_after_seconds(
            RateLimitError("slow")
        )
        == 60
    )
    assert (
        get_retry_after_seconds(
            ProviderTimeoutError("timeout")
        )
        == 30
    )
    assert (
        get_retry_after_seconds(
            ValidationError("bad")
        )
        is None
    )


@pytest.mark.parametrize(
    "error,expected_retry_after",
    [
        (RateLimitError("slow"), 60),
        (ProviderTimeoutError("timeout"), 30),
        (ExternalProviderError("down"), 30),
        (ValidationError("bad"), None),
        (NotFoundError("missing"), None),
        (RuntimeError("boom"), None),
    ],
)
def test_retry_after_mapping(
    error: Exception,
    expected_retry_after,
):
    assert (
        get_retry_after_seconds(error)
        == expected_retry_after
    )


# ---------------------------------------------------------------------------
# Error-message sanitization
# ---------------------------------------------------------------------------


def test_sanitize_error_message():
    sanitized = sanitize_error_message(
        "postgresql://user:password@localhost/db"
    )

    assert (
        "password" not in sanitized
    )
    assert "[REDACTED]" in sanitized


@pytest.mark.parametrize(
    "unsafe_message",
    [
        "postgresql://user:password@localhost/db",
        "postgres://user:password@localhost/db",
        "redis://user:password@localhost:6379/0",
        "api_key=sk-should-not-leak",
        "apikey=sk-should-not-leak",
        "password=hunter2",
        "token=abc123",
        "Bearer abcdef123456",
        "https://storage.example.com/signed?signature=secret",
    ],
)
def test_sanitize_error_message_redacts_sensitive_patterns(
    unsafe_message: str,
):
    sanitized = sanitize_error_message(
        unsafe_message
    )

    assert "should-not-leak" not in sanitized
    assert "hunter2" not in sanitized
    assert "abc123" not in sanitized
    assert "abcdef123456" not in sanitized
    assert "password@localhost" not in sanitized
    assert "signature=secret" not in sanitized


def test_sanitize_error_message_preserves_safe_text():
    assert (
        sanitize_error_message(
            "Workflow validation failed."
        )
        == "Workflow validation failed."
    )


def test_sanitize_error_message_handles_empty_string():
    assert (
        sanitize_error_message("")
        == ""
    )


def test_sanitize_error_message_handles_none():
    assert (
        sanitize_error_message(None)
        == ""
    )


# ---------------------------------------------------------------------------
# Tool-metadata sanitization
# ---------------------------------------------------------------------------


def test_sanitize_tool_metadata():
    metadata = {
        "api_key": "should-not-leak",
        "prompt": "should-not-leak",
        "model_name": "gpt-4.1-mini",
        "tenant_id": "tenant_123",
    }

    safe_metadata = sanitize_tool_metadata(
        metadata
    )

    assert (
        safe_metadata["api_key"]
        == "[REDACTED]"
    )
    assert (
        safe_metadata["prompt"]
        == "[REDACTED]"
    )
    assert (
        safe_metadata["model_name"]
        == "gpt-4.1-mini"
    )
    assert (
        safe_metadata["tenant_id"]
        == "tenant_123"
    )


@pytest.mark.parametrize(
    "sensitive_key",
    [
        "api_key",
        "apikey",
        "API_KEY",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "credential",
        "cookie",
        "private_key",
        "database_url",
        "connection_string",
        "prompt",
        "completion",
        "messages",
        "input",
        "output",
        "embedding",
        "vector",
        "generated_code",
        "signed_url",
        "storage_url",
        "file_url",
        "download_url",
    ],
)
def test_sanitize_tool_metadata_redacts_sensitive_keys(
    sensitive_key: str,
):
    metadata = {
        "safe": "value",
        sensitive_key: "should-not-leak",
    }

    safe_metadata = sanitize_tool_metadata(
        metadata
    )

    assert (
        safe_metadata["safe"] == "value"
    )
    assert (
        safe_metadata[sensitive_key]
        == "[REDACTED]"
    )


def test_sanitize_tool_metadata_handles_nested_values():
    metadata = {
        "request": {
            "api_key": "should-not-leak",
            "model": "gpt-4.1-mini",
        },
        "tags": ["workflow", "security"],
    }

    safe_metadata = sanitize_tool_metadata(
        metadata
    )

    assert (
        safe_metadata["request"]["api_key"]
        == "[REDACTED]"
    )
    assert (
        safe_metadata["request"]["model"]
        == "gpt-4.1-mini"
    )
    assert safe_metadata["tags"] == [
        "workflow",
        "security",
    ]


def test_sanitize_tool_metadata_handles_none():
    assert (
        sanitize_tool_metadata(None)
        == {}
    )


def test_sanitize_tool_metadata_handles_empty_dict():
    assert (
        sanitize_tool_metadata({})
        == {}
    )


def test_sanitize_tool_metadata_handles_non_dict():
    assert (
        sanitize_tool_metadata(
            "not-a-dictionary"
        )
        == {}
    )


# ---------------------------------------------------------------------------
# Tool-result sanitization
# ---------------------------------------------------------------------------


def test_sanitize_tool_result():
    result = {
        "execution_id": "wf_123",
        "generated_code_files": {
            "app/main.py": (
                "print('secret')"
            ),
        },
        "requirements": [
            "Build dashboard",
            "Show metrics",
        ],
    }

    safe_result = sanitize_tool_result(
        result
    )

    assert (
        safe_result["execution_id"]
        == "wf_123"
    )
    assert safe_result[
        "requirements"
    ] == [
        "Build dashboard",
        "Show metrics",
    ]
    assert safe_result[
        "generated_code_files"
    ] == "[UNSERIALIZABLE]"


@pytest.mark.parametrize(
    "sensitive_key",
    [
        "api_key",
        "apikey",
        "password",
        "secret",
        "token",
        "authorization",
        "credential",
        "prompt",
        "completion",
        "messages",
        "embedding",
        "vector",
        "generated_code",
        "storage_url",
        "signed_url",
        "database_url",
        "connection_string",
    ],
)
def test_sanitize_tool_result_redacts_sensitive_keys(
    sensitive_key: str,
):
    result = {
        "safe": "value",
        sensitive_key: "should-not-leak",
    }

    safe_result = sanitize_tool_result(
        result
    )

    assert (
        safe_result["safe"] == "value"
    )
    assert (
        safe_result[sensitive_key]
        == "[REDACTED]"
    )


def test_sanitize_tool_result_does_not_leak_generated_code():
    result = {
        "generated_code_files": {
            "app/main.py": (
                "API_KEY = 'sk-should-not-leak'"
            )
        }
    }

    safe_result = sanitize_tool_result(
        result
    )

    serialized_result = str(safe_result)

    assert (
        "sk-should-not-leak"
        not in serialized_result
    )
    assert (
        "API_KEY"
        not in serialized_result
    )


def test_sanitize_tool_result_handles_none():
    assert (
        sanitize_tool_result(None)
        == {}
    )


def test_sanitize_tool_result_handles_empty_dict():
    assert (
        sanitize_tool_result({})
        == {}
    )


def test_sanitize_tool_result_handles_non_dict():
    assert (
        sanitize_tool_result(
            "not-a-dictionary"
        )
        == {}
    )


# ---------------------------------------------------------------------------
# Tool-argument sanitization
# ---------------------------------------------------------------------------


def test_sanitize_tool_arguments():
    arguments = {
        "brief_text": "Build a dashboard",
        "api_key": "should-not-leak",
        "document_ids": [str(uuid4())],
    }

    safe_arguments = (
        sanitize_tool_arguments(
            arguments
        )
    )

    assert (
        safe_arguments["brief_text"]
        == "Build a dashboard"
    )
    assert (
        safe_arguments["api_key"]
        == "[REDACTED]"
    )
    assert safe_arguments[
        "document_ids"
    ] == "[UNSERIALIZABLE]"


@pytest.mark.parametrize(
    "sensitive_key",
    [
        "api_key",
        "apikey",
        "password",
        "secret",
        "token",
        "authorization",
        "credential",
        "prompt",
        "completion",
        "messages",
        "embedding",
        "vector",
        "generated_code",
        "storage_url",
        "signed_url",
        "database_url",
        "connection_string",
    ],
)
def test_sanitize_tool_arguments_redacts_sensitive_keys(
    sensitive_key: str,
):
    arguments = {
        "safe": "value",
        sensitive_key: "should-not-leak",
    }

    safe_arguments = (
        sanitize_tool_arguments(
            arguments
        )
    )

    assert (
        safe_arguments["safe"]
        == "value"
    )
    assert (
        safe_arguments[sensitive_key]
        == "[REDACTED]"
    )


def test_sanitize_tool_arguments_handles_none():
    assert (
        sanitize_tool_arguments(None)
        == {}
    )


def test_sanitize_tool_arguments_handles_empty_dict():
    assert (
        sanitize_tool_arguments({})
        == {}
    )


def test_sanitize_tool_arguments_handles_non_dict():
    assert (
        sanitize_tool_arguments(
            "not-a-dictionary"
        )
        == {}
    )


# ---------------------------------------------------------------------------
# Error-response construction
# ---------------------------------------------------------------------------


def test_build_error_response():
    response = build_error_response(
        ValidationError("Invalid brief"),
        request_id="req_1",
        tool_name=(
            "start_generation_workflow"
        ),
    )

    assert (
        response["error"]["code"]
        == "validation_error"
    )
    assert (
        response["error"]["message"]
        == "Invalid brief"
    )
    assert (
        response["error"]["status"]
        == 422
    )
    assert (
        response["error"]["request_id"]
        == "req_1"
    )
    assert (
        response["error"]["tool_name"]
        == "start_generation_workflow"
    )
    assert (
        response["error"]["retryable"]
        is False
    )


def test_build_error_response_includes_retry_after():
    response = build_error_response(
        RateLimitError("Too many requests"),
        request_id="req_1",
    )

    assert (
        response["error"][
            "retry_after_seconds"
        ]
        == 60
    )
    assert (
        response["error"]["retryable"]
        is True
    )


@pytest.mark.parametrize(
    "error,expected_code,expected_status",
    [
        (ValidationError("bad"), "validation_error", 422),
        (
            AuthenticationError("no"),
            "authentication_error",
            401,
        ),
        (
            AuthorizationError("no"),
            "authorization_error",
            403,
        ),
        (NotFoundError("missing"), "not_found", 404),
        (
            RateLimitError("slow"),
            "rate_limit_exceeded",
            429,
        ),
        (
            ProviderTimeoutError("timeout"),
            "provider_timeout",
            504,
        ),
        (
            ExternalProviderError("down"),
            "external_provider_error",
            502,
        ),
        (
            WorkflowStateError("bad state"),
            "workflow_error",
            400,
        ),
        (RuntimeError("boom"), "internal_error", 500),
    ],
)
def test_build_error_response_maps_error_contract(
    error: Exception,
    expected_code: str,
    expected_status: int,
):
    response = build_error_response(
        error,
        request_id="req_1",
    )

    assert (
        response["error"]["code"]
        == expected_code
    )
    assert (
        response["error"]["status"]
        == expected_status
    )
    assert (
        response["error"]["request_id"]
        == "req_1"
    )


def test_build_error_response_hides_internal_error_message():
    response = build_error_response(
        RuntimeError(
            "database password is hunter2"
        ),
        request_id="req_1",
    )

    serialized_response = str(
        response
    ).lower()

    assert "hunter2" not in serialized_response
    assert "database" not in serialized_response
    assert (
        response["error"]["message"]
        == (
            "The requested MCP operation "
            "failed due to an unexpected "
            "server error."
        )
    )


def test_build_error_response_hides_provider_error_message():
    response = build_error_response(
        ProviderTimeoutError(
            "secret-provider-internal-detail"
        ),
        request_id="req_1",
    )

    serialized_response = str(
        response
    )

    assert (
        "secret-provider-internal-detail"
        not in serialized_response
    )


def test_build_error_response_omits_retry_after_for_non_retryable_errors():
    response = build_error_response(
        ValidationError("Invalid brief"),
        request_id="req_1",
    )

    assert (
        "retry_after_seconds"
        not in response["error"]
    )


def test_build_error_response_handles_missing_request_metadata():
    response = build_error_response(
        ValidationError("Invalid brief")
    )

    assert (
        response["error"]["request_id"]
        is None
    )
    assert (
        response["error"]["tool_name"]
        is None
    )


# ---------------------------------------------------------------------------
# MCP tool error boundary
# ---------------------------------------------------------------------------


async def test_tool_boundary_sanitizes_result():
    @mcp_tool_error_boundary
    async def sample_tool():
        return {
            "api_key": "should-not-leak",
            "status": "ok",
        }

    result = await sample_tool()

    assert (
        result["api_key"]
        == "[REDACTED]"
    )
    assert result["status"] == "ok"


async def test_tool_boundary_wraps_unexpected_error():
    @mcp_tool_error_boundary
    async def failing_tool():
        raise RuntimeError(
            "internal secret"
        )

    with pytest.raises(AppError):
        await failing_tool()


async def test_tool_boundary_preserves_validation_error():
    @mcp_tool_error_boundary
    async def invalid_tool():
        raise ValidationError(
            "Invalid brief"
        )

    with pytest.raises(ValidationError):
        await invalid_tool()


@pytest.mark.parametrize(
    "error_type",
    [
        AuthenticationError,
        AuthorizationError,
        NotFoundError,
        RateLimitError,
        ProviderTimeoutError,
        ExternalProviderError,
        WorkflowStateError,
    ],
)
async def test_tool_boundary_preserves_domain_errors(
    error_type,
):
    @mcp_tool_error_boundary
    async def failing_tool():
        raise error_type("domain failure")

    with pytest.raises(error_type):
        await failing_tool()


async def test_tool_boundary_wraps_value_error():
    @mcp_tool_error_boundary
    async def failing_tool():
        raise ValueError("unexpected failure")

    with pytest.raises(AppError):
        await failing_tool()


async def test_tool_boundary_wraps_key_error():
    @mcp_tool_error_boundary
    async def failing_tool():
        raise KeyError("missing_key")

    with pytest.raises(AppError):
        await failing_tool()


async def test_tool_boundary_does_not_leak_internal_error_details():
    @mcp_tool_error_boundary
    async def failing_tool():
        raise RuntimeError(
            "database password is hunter2"
        )

    with pytest.raises(AppError) as exc_info:
        await failing_tool()

    serialized_error = str(
        exc_info.value
    ).lower()

    assert "hunter2" not in serialized_error
    assert "database" not in serialized_error


async def test_tool_boundary_sanitizes_nested_result():
    @mcp_tool_error_boundary
    async def sample_tool():
        return {
            "metadata": {
                "api_key": "should-not-leak",
                "model": "gpt-4.1-mini",
            },
            "status": "ok",
        }

    result = await sample_tool()

    assert (
        result["metadata"]["api_key"]
        == "[REDACTED]"
    )
    assert (
        result["metadata"]["model"]
        == "gpt-4.1-mini"
    )
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    asyncio.run(
        pytest.main(
            [
                __file__,
                "-v",
            ]
        )
    )
