"""
Comprehensive tests for MCP request-context handling.

These tests cover request and correlation ID generation, principal context
creation and validation, context attribute handling, metadata and value
sanitization, principal resolution, structured logging, and secret-leak
prevention.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID, uuid4

import pytest

from app.core.exceptions import (
    AuthenticationError,
    ValidationError,
)
from app.mcp_server.auth import MCPPrincipal
from app.mcp_server.context import (
    create_request_context,
    generate_mcp_correlation_id,
    generate_mcp_request_id,
    get_context_principal,
    log_context_event,
    sanitize_context_metadata,
    sanitize_context_value,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def principal():
    return MCPPrincipal(
        user_id=str(uuid4()),
        tenant_id=str(uuid4()),
        role="developer",
        scopes=frozenset(
            {"workflows:read"}
        ),
    )


@pytest.fixture
def context(principal):
    return create_request_context(
        principal=principal,
    )


# ---------------------------------------------------------------------------
# Request and correlation IDs
# ---------------------------------------------------------------------------


def test_generate_ids():
    request_id = (
        generate_mcp_request_id()
    )
    correlation_id = (
        generate_mcp_correlation_id()
    )

    assert (
        request_id
        != generate_mcp_request_id()
    )
    assert (
        correlation_id
        != generate_mcp_correlation_id()
    )


def test_generate_request_id_returns_uuid_string():
    request_id = (
        generate_mcp_request_id()
    )

    assert isinstance(
        request_id,
        str,
    )
    assert UUID(request_id)


def test_generate_correlation_id_returns_uuid_string():
    correlation_id = (
        generate_mcp_correlation_id()
    )

    assert isinstance(
        correlation_id,
        str,
    )
    assert UUID(correlation_id)


def test_generated_ids_are_url_safe():
    request_id = (
        generate_mcp_request_id()
    )
    correlation_id = (
        generate_mcp_correlation_id()
    )

    for value in (
        request_id,
        correlation_id,
    ):
        assert " " not in value
        assert "/" not in value
        assert "+" not in value


# ---------------------------------------------------------------------------
# Request-context creation
# ---------------------------------------------------------------------------


def test_create_request_context(
    principal,
):
    context = create_request_context(
        principal=principal,
    )

    assert (
        context.tenant_id
        == principal.tenant_id
    )
    assert (
        context.user_id
        == principal.user_id
    )
    assert context.role == principal.role
    assert (
        "workflows:read"
        in context.scopes
    )
    assert (
        context.elapsed_ms() >= 0
    )


def test_create_request_context_with_ids(
    principal,
):
    request_id = uuid4()
    correlation_id = uuid4()

    context = create_request_context(
        principal=principal,
        request_id=request_id,
        correlation_id=correlation_id,
    )

    assert (
        context.request_id
        == str(request_id)
    )
    assert (
        context.correlation_id
        == str(correlation_id)
    )


def test_create_request_context_generates_ids_when_missing(
    principal,
):
    context = create_request_context(
        principal=principal,
    )

    assert context.request_id
    assert context.correlation_id
    assert UUID(context.request_id)
    assert UUID(
        context.correlation_id
    )


def test_create_request_context_generates_unique_ids(
    principal,
):
    first_context = (
        create_request_context(
            principal=principal,
        )
    )
    second_context = (
        create_request_context(
            principal=principal,
        )
    )

    assert (
        first_context.request_id
        != second_context.request_id
    )
    assert (
        first_context.correlation_id
        != second_context.correlation_id
    )


def test_create_request_context_preserves_scopes(
    principal,
):
    context = create_request_context(
        principal=principal,
    )

    assert context.scopes == (
        principal.scopes
    )


def test_create_request_context_elapsed_time_increases(
    principal,
):
    import asyncio

    context = create_request_context(
        principal=principal,
    )

    initial_elapsed = (
        context.elapsed_ms()
    )

    asyncio.run(asyncio.sleep(0.01))

    assert (
        context.elapsed_ms()
        >= initial_elapsed
    )


def test_create_request_context_rejects_missing_principal():
    with pytest.raises(
        AuthenticationError
    ):
        create_request_context(
            principal=None,
        )


def test_create_request_context_rejects_incomplete_principal():
    incomplete_principal = MCPPrincipal(
        user_id="",
        tenant_id="",
        role="developer",
        scopes=frozenset(),
    )

    with pytest.raises(
        AuthenticationError
    ):
        create_request_context(
            principal=incomplete_principal,
        )


@pytest.mark.parametrize(
    "invalid_principal",
    [
        None,
        "not-a-principal",
        123,
        {},
        [],
    ],
)
def test_create_request_context_rejects_invalid_principal_types(
    invalid_principal,
):
    with pytest.raises(
        AuthenticationError
    ):
        create_request_context(
            principal=invalid_principal,
        )


# ---------------------------------------------------------------------------
# Principal resolution
# ---------------------------------------------------------------------------


def test_get_context_principal(
    principal,
):
    context = create_request_context(
        principal=principal,
    )

    resolved_principal = (
        get_context_principal(
            context
        )
    )

    assert (
        resolved_principal is principal
    )


def test_get_context_principal_rejects_missing_context():
    with pytest.raises(
        AuthenticationError
    ):
        get_context_principal(
            None
        )


@pytest.mark.parametrize(
    "invalid_context",
    [
        "not-a-context",
        123,
        {},
        [],
    ],
)
def test_get_context_principal_rejects_invalid_context(
    invalid_context,
):
    with pytest.raises(
        AuthenticationError
    ):
        get_context_principal(
            invalid_context
        )


# ---------------------------------------------------------------------------
# Context-metadata sanitization
# ---------------------------------------------------------------------------


def test_sanitize_context_metadata():
    metadata = {
        "api_key": "should-not-leak",
        "prompt": "should-not-leak",
        "model_name": "gpt-4.1-mini",
        "tenant_id": "tenant_123",
    }

    safe_metadata = (
        sanitize_context_metadata(
            metadata
        )
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
def test_sanitize_context_metadata_redacts_sensitive_keys(
    sensitive_key: str,
):
    metadata = {
        "safe": "value",
        sensitive_key: "should-not-leak",
    }

    safe_metadata = (
        sanitize_context_metadata(
            metadata
        )
    )

    assert (
        safe_metadata["safe"]
        == "value"
    )
    assert (
        safe_metadata[sensitive_key]
        == "[REDACTED]"
    )


def test_sanitize_context_metadata_handles_nested_values():
    metadata = {
        "request": {
            "api_key": "should-not-leak",
            "model": "gpt-4.1-mini",
        },
        "tags": ["workflow", "security"],
    }

    safe_metadata = (
        sanitize_context_metadata(
            metadata
        )
    )

    assert safe_metadata[
        "request"
    ]["api_key"] == "[REDACTED]"
    assert safe_metadata[
        "request"
    ]["model"] == "gpt-4.1-mini"
    assert safe_metadata["tags"] == [
        "workflow",
        "security",
    ]


def test_sanitize_context_metadata_handles_none():
    assert (
        sanitize_context_metadata(
            None
        )
        == {}
    )


def test_sanitize_context_metadata_handles_empty_dict():
    assert (
        sanitize_context_metadata({})
        == {}
    )


def test_sanitize_context_metadata_handles_non_dict():
    assert (
        sanitize_context_metadata(
            "not-a-dictionary"
        )
        == {}
    )


# ---------------------------------------------------------------------------
# Context-value sanitization
# ---------------------------------------------------------------------------


def test_sanitize_context_value():
    assert (
        sanitize_context_value("hello")
        == "hello"
    )
    assert (
        sanitize_context_value(42)
        == 42
    )
    assert (
        sanitize_context_value(None)
        is None
    )
    assert (
        sanitize_context_value(
            uuid4()
        )
        is not None
    )
    assert (
        sanitize_context_value(
            {"api_key": "secret"}
        )
        == {
            "api_key": "[REDACTED]"
        }
    )
    assert (
        sanitize_context_value(
            object()
        )
        == "[UNSERIALIZABLE]"
    )


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "api_key=sk-should-not-leak",
        "apikey=sk-should-not-leak",
        "password=hunter2",
        "token=abc123",
        "Bearer abcdef123456",
        "postgresql://user:password@localhost/db",
        "postgres://user:password@localhost/db",
        "redis://user:password@localhost:6379/0",
        "https://storage.example.com/signed?signature=secret",
    ],
)
def test_sanitize_context_value_redacts_sensitive_patterns(
    unsafe_value: str,
):
    sanitized = sanitize_context_value(
        unsafe_value
    )

    assert (
        "should-not-leak"
        not in sanitized
    )
    assert "hunter2" not in sanitized
    assert "abc123" not in sanitized
    assert (
        "abcdef123456" not in sanitized
    )
    assert (
        "password@localhost"
        not in sanitized
    )
    assert (
        "signature=secret"
        not in sanitized
    )


def test_sanitize_context_value_handles_primitives():
    assert (
        sanitize_context_value(True)
        is True
    )
    assert (
        sanitize_context_value(False)
        is False
    )
    assert (
        sanitize_context_value(3.14)
        == 3.14
    )


def test_sanitize_context_value_handles_uuid():
    value = uuid4()

    assert (
        sanitize_context_value(value)
        == str(value)
    )


def test_sanitize_context_value_handles_lists():
    assert (
        sanitize_context_value(
            ["one", "two"]
        )
        == ["one", "two"]
    )


def test_sanitize_context_value_handles_nested_sensitive_values():
    sanitized = sanitize_context_value(
        {
            "metadata": {
                "api_key": "should-not-leak",
                "model": "gpt-4.1-mini",
            }
        }
    )

    assert sanitized["metadata"][
        "api_key"
    ] == "[REDACTED]"
    assert sanitized["metadata"][
        "model"
    ] == "gpt-4.1-mini"


# ---------------------------------------------------------------------------
# Context attributes
# ---------------------------------------------------------------------------


def test_set_attribute_redacts_sensitive_values(
    principal,
):
    context = create_request_context(
        principal=principal,
    )

    context.set_attribute(
        "api_key",
        "should-not-leak",
    )

    assert (
        context.get_attribute("api_key")
        == "[REDACTED]"
    )


def test_set_attribute_preserves_safe_values(
    principal,
):
    context = create_request_context(
        principal=principal,
    )

    context.set_attribute(
        "workflow_step",
        "requirements",
    )

    assert (
        context.get_attribute(
            "workflow_step"
        )
        == "requirements"
    )


def test_set_attribute_rejects_invalid_key(
    principal,
):
    context = create_request_context(
        principal=principal,
    )

    with pytest.raises(ValidationError):
        context.set_attribute(
            "",
            "value",
        )


@pytest.mark.parametrize(
    "invalid_key",
    [
        "",
        "   ",
        None,
        123,
        [],
        {},
    ],
)
def test_set_attribute_rejects_invalid_keys(
    principal,
    invalid_key,
):
    context = create_request_context(
        principal=principal,
    )

    with pytest.raises(ValidationError):
        context.set_attribute(
            invalid_key,
            "value",
        )


def test_get_attribute_returns_none_for_missing_key(
    principal,
):
    context = create_request_context(
        principal=principal,
    )

    assert (
        context.get_attribute(
            "missing_attribute"
        )
        is None
    )


def test_set_attribute_does_not_leak_sensitive_value(
    principal,
):
    context = create_request_context(
        principal=principal,
    )

    context.set_attribute(
        "password",
        "hunter2",
    )

    serialized_context = str(context)

    assert (
        "hunter2"
        not in serialized_context
    )


# ---------------------------------------------------------------------------
# Context logging
# ---------------------------------------------------------------------------


def test_log_context_event(
    principal,
    caplog,
):
    context = create_request_context(
        principal=principal,
    )

    with caplog.at_level("INFO"):
        log_context_event(
            context,
            "mcp_tool_invocation_started",
            tool_name="get_workflow_status",
            api_key="should-not-leak",
        )

    assert (
        "mcp_tool_invocation_started"
        in caplog.text
    )
    assert (
        "should-not-leak"
        not in caplog.text
    )


def test_log_context_event_includes_tool_name(
    principal,
    caplog,
):
    context = create_request_context(
        principal=principal,
    )

    with caplog.at_level("INFO"):
        log_context_event(
            context,
            "mcp_tool_invocation_started",
            tool_name="start_generation_workflow",
        )

    assert (
        "start_generation_workflow"
        in caplog.text
    )


def test_log_context_event_includes_request_context(
    principal,
    caplog,
):
    context = create_request_context(
        principal=principal,
    )

    with caplog.at_level("INFO"):
        log_context_event(
            context,
            "mcp_tool_invocation_started",
        )

    assert (
        context.request_id
        in caplog.text
    )
    assert (
        context.correlation_id
        in caplog.text
    )
    assert (
        context.tenant_id
        in caplog.text
    )


def test_log_context_event_redacts_nested_sensitive_values(
    principal,
    caplog,
):
    context = create_request_context(
        principal=principal,
    )

    with caplog.at_level("INFO"):
        log_context_event(
            context,
            "mcp_tool_invocation_started",
            metadata={
                "api_key": "should-not-leak",
                "model": "gpt-4.1-mini",
            },
        )

    assert (
        "should-not-leak"
        not in caplog.text
    )
    assert (
        "gpt-4.1-mini" in caplog.text
    )


def test_log_context_event_handles_unserializable_values(
    principal,
    caplog,
):
    context = create_request_context(
        principal=principal,
    )

    with caplog.at_level("INFO"):
        log_context_event(
            context,
            "mcp_tool_invocation_started",
            unsupported=object(),
        )

    assert (
        "mcp_tool_invocation_started"
        in caplog.text
    )


def test_log_context_event_does_not_leak_prompt(
    principal,
    caplog,
):
    context = create_request_context(
        principal=principal,
    )

    with caplog.at_level("INFO"):
        log_context_event(
            context,
            "mcp_tool_invocation_started",
            prompt="secret user prompt",
        )

    assert (
        "secret user prompt"
        not in caplog.text
    )


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
