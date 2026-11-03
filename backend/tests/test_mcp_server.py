"""
Comprehensive tests for MCP server helpers, authentication, validation,
sanitization, and tool error boundaries.

These tests cover string and UUID validation, UUID-list normalization,
metadata and result sanitization, authenticated principal resolution,
authorization scopes, tenant isolation, error boundaries, and MCP server
composition.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

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
from app.mcp_server.auth import MCPPrincipal
from app.mcp_server.server import (
    MCP_SERVER_NAME,
    MCP_SERVER_VERSION,
    _require_string,
    _require_uuid,
    _sanitize_metadata,
    _sanitize_tool_result,
    _validate_uuid_list,
    get_authenticated_principal,
    get_mcp_server,
    get_server_health,
    get_server_info,
    list_server_capabilities,
    mcp_tool_error_boundary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def principal() -> MCPPrincipal:
    """Return an authenticated MCP principal with workflow scopes."""
    return MCPPrincipal(
        user_id=str(uuid4()),
        tenant_id=str(uuid4()),
        role="developer",
        scopes=frozenset(
            {
                "workflows:write",
                "workflows:read",
                "workflows:cancel",
                "workflows:review",
                "documents:read",
            }
        ),
    )


@pytest.fixture
def readonly_principal() -> MCPPrincipal:
    """Return an authenticated MCP principal with read-only scopes."""
    return MCPPrincipal(
        user_id=str(uuid4()),
        tenant_id=str(uuid4()),
        role="viewer",
        scopes=frozenset(
            {
                "workflows:read",
                "documents:read",
            }
        ),
    )


@pytest.fixture
def authenticated_context(
    principal: MCPPrincipal,
) -> SimpleNamespace:
    """Return an MCP request context containing an authenticated principal."""
    return SimpleNamespace(
        request_context=SimpleNamespace(
            principal=principal,
        )
    )


# ---------------------------------------------------------------------------
# String validation
# ---------------------------------------------------------------------------


def test_require_string():
    assert (
        _require_string(
            "  Build a dashboard  ",
            field_name="brief_text",
            max_length=100,
        )
        == "Build a dashboard"
    )


def test_require_string_preserves_internal_spaces():
    assert (
        _require_string(
            "Build a secure dashboard",
            field_name="brief_text",
            max_length=100,
        )
        == "Build a secure dashboard"
    )


def test_require_string_accepts_maximum_length():
    value = "x" * 100

    assert (
        _require_string(
            value,
            field_name="brief_text",
            max_length=100,
        )
        == value
    )


def test_require_string_rejects_invalid_values():
    with pytest.raises(ValidationError):
        _require_string(
            "",
            field_name="brief_text",
            max_length=100,
        )

    with pytest.raises(ValidationError):
        _require_string(
            None,
            field_name="brief_text",
            max_length=100,
        )

    with pytest.raises(ValidationError):
        _require_string(
            "x" * 101,
            field_name="brief_text",
            max_length=100,
        )


def test_require_string_rejects_whitespace_only_value():
    with pytest.raises(ValidationError):
        _require_string(
            "   ",
            field_name="brief_text",
            max_length=100,
        )


def test_require_string_rejects_non_string_value():
    with pytest.raises(ValidationError):
        _require_string(
            123,
            field_name="brief_text",
            max_length=100,
        )

    with pytest.raises(ValidationError):
        _require_string(
            ["Build a dashboard"],
            field_name="brief_text",
            max_length=100,
        )

    with pytest.raises(ValidationError):
        _require_string(
            {"text": "Build a dashboard"},
            field_name="brief_text",
            max_length=100,
        )


def test_require_string_rejects_invalid_max_length():
    with pytest.raises(ValidationError):
        _require_string(
            "Build a dashboard",
            field_name="brief_text",
            max_length=0,
        )


def test_require_string_rejects_negative_max_length():
    with pytest.raises(ValidationError):
        _require_string(
            "Build a dashboard",
            field_name="brief_text",
            max_length=-1,
        )


# ---------------------------------------------------------------------------
# UUID validation
# ---------------------------------------------------------------------------


def test_require_uuid():
    value = uuid4()

    assert (
        _require_uuid(
            str(value),
            field_name="execution_id",
        )
        == value
    )


def test_require_uuid_accepts_uuid_object():
    value = uuid4()

    assert (
        _require_uuid(
            value,
            field_name="execution_id",
        )
        == value
    )


def test_require_uuid_trims_whitespace():
    value = uuid4()

    assert (
        _require_uuid(
            f"  {value}  ",
            field_name="execution_id",
        )
        == value
    )


def test_require_uuid_rejects_invalid_value():
    with pytest.raises(ValidationError):
        _require_uuid(
            "not-a-uuid",
            field_name="execution_id",
        )


def test_require_uuid_rejects_empty_value():
    with pytest.raises(ValidationError):
        _require_uuid(
            "",
            field_name="execution_id",
        )


def test_require_uuid_rejects_whitespace_only_value():
    with pytest.raises(ValidationError):
        _require_uuid(
            "   ",
            field_name="execution_id",
        )


def test_require_uuid_rejects_none_value():
    with pytest.raises(ValidationError):
        _require_uuid(
            None,
            field_name="execution_id",
        )


def test_require_uuid_rejects_integer_value():
    with pytest.raises(ValidationError):
        _require_uuid(
            12345,
            field_name="execution_id",
        )


# ---------------------------------------------------------------------------
# UUID list validation
# ---------------------------------------------------------------------------


def test_validate_uuid_list_deduplicates():
    first_id = uuid4()
    second_id = uuid4()

    ids = _validate_uuid_list(
        [
            str(first_id),
            first_id,
            str(second_id),
        ],
        field_name="document_ids",
        max_items=10,
    )

    assert ids == [first_id, second_id]


def test_validate_uuid_list_preserves_order():
    first_id = uuid4()
    second_id = uuid4()
    third_id = uuid4()

    ids = _validate_uuid_list(
        [
            third_id,
            first_id,
            second_id,
        ],
        field_name="document_ids",
        max_items=10,
    )

    assert ids == [
        third_id,
        first_id,
        second_id,
    ]


def test_validate_uuid_list_accepts_empty_list():
    ids = _validate_uuid_list(
        [],
        field_name="document_ids",
        max_items=10,
    )

    assert ids == []


def test_validate_uuid_list_accepts_none():
    ids = _validate_uuid_list(
        None,
        field_name="document_ids",
        max_items=10,
    )

    assert ids == []


def test_validate_uuid_list_rejects_too_many_items():
    with pytest.raises(ValidationError):
        _validate_uuid_list(
            [
                str(uuid4())
                for _ in range(26)
            ],
            field_name="document_ids",
            max_items=25,
        )


def test_validate_uuid_list_accepts_maximum_items():
    document_ids = [
        uuid4() for _ in range(25)
    ]

    ids = _validate_uuid_list(
        document_ids,
        field_name="document_ids",
        max_items=25,
    )

    assert ids == document_ids


def test_validate_uuid_list_rejects_invalid_item():
    with pytest.raises(ValidationError):
        _validate_uuid_list(
            [
                str(uuid4()),
                "not-a-uuid",
            ],
            field_name="document_ids",
            max_items=10,
        )


def test_validate_uuid_list_rejects_empty_item():
    with pytest.raises(ValidationError):
        _validate_uuid_list(
            [
                str(uuid4()),
                "",
            ],
            field_name="document_ids",
            max_items=10,
        )


def test_validate_uuid_list_rejects_non_list_value():
    with pytest.raises(ValidationError):
        _validate_uuid_list(
            "not-a-list",
            field_name="document_ids",
            max_items=10,
        )


def test_validate_uuid_list_rejects_invalid_max_items():
    with pytest.raises(ValidationError):
        _validate_uuid_list(
            [str(uuid4())],
            field_name="document_ids",
            max_items=0,
        )


# ---------------------------------------------------------------------------
# Metadata sanitization
# ---------------------------------------------------------------------------


def test_sanitize_metadata_redacts_sensitive_keys():
    metadata = {
        "source": "sharepoint",
        "api_key": "should-not-leak",
        "prompt": "should-not-leak",
        "model": "gpt-4.1-mini",
    }

    safe_metadata = _sanitize_metadata(
        metadata
    )

    assert safe_metadata["source"] == "sharepoint"
    assert (
        safe_metadata["api_key"]
        == "[REDACTED]"
    )
    assert (
        safe_metadata["prompt"]
        == "[REDACTED]"
    )
    assert (
        safe_metadata["model"]
        == "gpt-4.1-mini"
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
        "authorization",
        "credential",
        "private_key",
        "database_url",
        "connection_string",
        "embedding",
        "vector",
        "completion",
        "messages",
    ],
)
def test_sanitize_metadata_redacts_common_sensitive_keys(
    sensitive_key: str,
):
    metadata = {
        "safe": "value",
        sensitive_key: "should-not-leak",
    }

    safe_metadata = _sanitize_metadata(
        metadata
    )

    assert safe_metadata["safe"] == "value"
    assert (
        safe_metadata[sensitive_key]
        == "[REDACTED]"
    )


def test_sanitize_metadata_handles_nested_values():
    metadata = {
        "request": {
            "api_key": "should-not-leak",
            "model": "gpt-4.1-mini",
        },
        "tags": ["requirements", "security"],
    }

    safe_metadata = _sanitize_metadata(
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
        "requirements",
        "security",
    ]


def test_sanitize_metadata_bounds_string_values():
    metadata = {
        "safe": "x" * 5_000,
    }

    safe_metadata = _sanitize_metadata(
        metadata
    )

    assert len(safe_metadata["safe"]) <= 1_000


def test_sanitize_metadata_handles_none():
    assert _sanitize_metadata(None) == {}


def test_sanitize_metadata_handles_empty_dict():
    assert _sanitize_metadata({}) == {}


def test_sanitize_metadata_handles_non_dict_value():
    assert (
        _sanitize_metadata("not-a-dict")
        == {}
    )


# ---------------------------------------------------------------------------
# Tool-result sanitization
# ---------------------------------------------------------------------------


def test_sanitize_tool_result():
    result = {
        "execution_id": uuid4(),
        "generated_code_files": {
            "app/main.py": "print('secret')",
        },
        "requirements": [
            "Build dashboard",
            "Show metrics",
        ],
    }

    safe_result = _sanitize_tool_result(
        result
    )

    assert safe_result["requirements"] == [
        "Build dashboard",
        "Show metrics",
    ]
    assert safe_result[
        "generated_code_files"
    ] == "[UNSERIALIZABLE]"


def test_sanitize_tool_result_redacts_sensitive_keys():
    result = {
        "execution_id": str(uuid4()),
        "api_key": "should-not-leak",
        "prompt": "should-not-leak",
        "status": "queued",
    }

    safe_result = _sanitize_tool_result(
        result
    )

    assert (
        safe_result["api_key"]
        == "[REDACTED]"
    )
    assert (
        safe_result["prompt"]
        == "[REDACTED]"
    )
    assert safe_result["status"] == "queued"


def test_sanitize_tool_result_bounds_strings():
    result = {
        "summary": "x" * 50_000,
    }

    safe_result = _sanitize_tool_result(
        result
    )

    assert len(safe_result["summary"]) <= 20_000


def test_sanitize_tool_result_handles_lists():
    result = {
        "items": [
            "one",
            "two",
            "three",
        ],
    }

    safe_result = _sanitize_tool_result(
        result
    )

    assert safe_result["items"] == [
        "one",
        "two",
        "three",
    ]


def test_sanitize_tool_result_handles_none():
    assert _sanitize_tool_result(None) is None


def test_sanitize_tool_result_handles_primitives():
    assert _sanitize_tool_result(True) is True
    assert _sanitize_tool_result(42) == 42
    assert _sanitize_tool_result(3.14) == 3.14
    assert (
        _sanitize_tool_result("safe")
        == "safe"
    )


def test_sanitize_tool_result_handles_unserializable_objects():
    class Unsupported:
        pass

    result = {
        "payload": Unsupported(),
    }

    safe_result = _sanitize_tool_result(
        result
    )

    assert (
        safe_result["payload"]
        == "[UNSERIALIZABLE]"
    )


# ---------------------------------------------------------------------------
# Authenticated principal resolution
# ---------------------------------------------------------------------------


async def test_get_authenticated_principal(
    principal: MCPPrincipal,
    authenticated_context: SimpleNamespace,
):
    resolved_principal = (
        await get_authenticated_principal(
            authenticated_context
        )
    )

    assert resolved_principal == principal


async def test_get_authenticated_principal_requires_auth():
    context = SimpleNamespace(
        request_context=SimpleNamespace(
            principal=None,
        )
    )

    with pytest.raises(AuthenticationError):
        await get_authenticated_principal(
            context
        )


async def test_get_authenticated_principal_requires_request_context():
    context = SimpleNamespace()

    with pytest.raises(AuthenticationError):
        await get_authenticated_principal(
            context
        )


async def test_get_authenticated_principal_requires_tenant(
    authenticated_context: SimpleNamespace,
):
    authenticated_context.request_context.principal = MCPPrincipal(
        user_id=str(uuid4()),
        tenant_id="",
        role="developer",
        scopes=frozenset(
            {"workflows:read"}
        ),
    )

    with pytest.raises(AuthorizationError):
        await get_authenticated_principal(
            authenticated_context
        )


async def test_get_authenticated_principal_requires_user(
    authenticated_context: SimpleNamespace,
):
    authenticated_context.request_context.principal = MCPPrincipal(
        user_id="",
        tenant_id=str(uuid4()),
        role="developer",
        scopes=frozenset(
            {"workflows:read"}
        ),
    )

    with pytest.raises(AuthorizationError):
        await get_authenticated_principal(
            authenticated_context
        )


async def test_get_authenticated_principal_rejects_invalid_principal_type():
    context = SimpleNamespace(
        request_context=SimpleNamespace(
            principal="not-a-principal",
        )
    )

    with pytest.raises(AuthenticationError):
        await get_authenticated_principal(
            context
        )


# ---------------------------------------------------------------------------
# Principal properties
# ---------------------------------------------------------------------------


def test_principal_scope_properties(principal):
    assert principal.can_start_workflows is True
    assert principal.can_read_workflows is True
    assert principal.can_cancel_workflows is True
    assert principal.can_review_workflows is True
    assert principal.can_read_documents is True


def test_readonly_principal_scope_properties(
    readonly_principal,
):
    assert (
        readonly_principal.can_start_workflows
        is False
    )
    assert (
        readonly_principal.can_read_workflows
        is True
    )
    assert (
        readonly_principal.can_cancel_workflows
        is False
    )
    assert (
        readonly_principal.can_review_workflows
        is False
    )
    assert (
        readonly_principal.can_read_documents
        is True
    )


# ---------------------------------------------------------------------------
# Tool error boundary
# ---------------------------------------------------------------------------


async def test_tool_error_boundary_sanitizes_result():
    @mcp_tool_error_boundary
    async def sample_tool():
        return {
            "api_key": "should-not-leak",
            "status": "ok",
        }

    result = await sample_tool()

    assert (
        result["api_key"] == "[REDACTED]"
    )
    assert result["status"] == "ok"


async def test_tool_error_boundary_wraps_unexpected_error():
    @mcp_tool_error_boundary
    async def failing_tool():
        raise RuntimeError("internal secret")

    with pytest.raises(AppError):
        await failing_tool()


async def test_tool_error_boundary_preserves_validation_error():
    @mcp_tool_error_boundary
    async def invalid_tool():
        raise ValidationError("Invalid brief")

    with pytest.raises(ValidationError):
        await invalid_tool()


async def test_tool_error_boundary_preserves_authentication_error():
    @mcp_tool_error_boundary
    async def unauthenticated_tool():
        raise AuthenticationError(
            "Authentication required"
        )

    with pytest.raises(AuthenticationError):
        await unauthenticated_tool()


async def test_tool_error_boundary_preserves_authorization_error():
    @mcp_tool_error_boundary
    async def unauthorized_tool():
        raise AuthorizationError(
            "Access denied"
        )

    with pytest.raises(AuthorizationError):
        await unauthorized_tool()


async def test_tool_error_boundary_preserves_not_found_error():
    @mcp_tool_error_boundary
    async def missing_tool():
        raise NotFoundError("Not found")

    with pytest.raises(NotFoundError):
        await missing_tool()


async def test_tool_error_boundary_preserves_rate_limit_error():
    @mcp_tool_error_boundary
    async def limited_tool():
        raise RateLimitError("Too many requests")

    with pytest.raises(RateLimitError):
        await limited_tool()


async def test_tool_error_boundary_preserves_provider_timeout():
    @mcp_tool_error_boundary
    async def timeout_tool():
        raise ProviderTimeoutError("Timed out")

    with pytest.raises(ProviderTimeoutError):
        await timeout_tool()


async def test_tool_error_boundary_preserves_external_provider_error():
    @mcp_tool_error_boundary
    async def provider_tool():
        raise ExternalProviderError(
            "Provider unavailable"
        )

    with pytest.raises(ExternalProviderError):
        await provider_tool()


async def test_tool_error_boundary_preserves_workflow_state_error():
    @mcp_tool_error_boundary
    async def invalid_state_tool():
        raise WorkflowStateError(
            "Invalid workflow state"
        )

    with pytest.raises(WorkflowStateError):
        await invalid_state_tool()


async def test_tool_error_boundary_does_not_leak_internal_error():
    @mcp_tool_error_boundary
    async def failing_tool():
        raise RuntimeError(
            "internal secret database password"
        )

    with pytest.raises(AppError) as exc_info:
        await failing_tool()

    serialized_error = str(exc_info.value).lower()

    assert "internal secret" not in serialized_error
    assert "database password" not in serialized_error


# ---------------------------------------------------------------------------
# MCP server metadata
# ---------------------------------------------------------------------------


def test_get_mcp_server():
    server = get_mcp_server()

    assert server is not None
    assert server.name == MCP_SERVER_NAME


def test_mcp_server_version():
    assert MCP_SERVER_VERSION == "1.2.0"


async def test_get_server_info():
    server_info = await get_server_info()

    assert (
        server_info["server_name"]
        == MCP_SERVER_NAME
    )
    assert (
        server_info["server_version"]
        == MCP_SERVER_VERSION
    )
    assert "workflow" in server_info[
        "available_tool_families"
    ]
    assert "wordpress" in server_info[
        "available_tool_families"
    ]
    assert "figma" in server_info[
        "available_tool_families"
    ]


async def test_get_server_health():
    health = await get_server_health()

    assert health["status"] == "healthy"
    assert health["checks"]["server"] == "healthy"
    assert (
        health["checks"]["tool_registry"]
        == "healthy"
    )
    assert (
        health["checks"]["security_policy"]
        == "healthy"
    )


async def test_list_server_capabilities():
    capabilities = (
        await list_server_capabilities()
    )

    capability_families = [
        capability["family"]
        for capability in capabilities[
            "capabilities"
        ]
    ]

    assert "workflow" in capability_families
    assert "documents" in capability_families
    assert "retrieval" in capability_families
    assert "review" in capability_families
    assert "cms" in capability_families
    assert "figma" in capability_families
    assert "wordpress" in capability_families


async def test_server_metadata_does_not_leak_secrets():
    server_info = await get_server_info()
    health = await get_server_health()
    capabilities = (
        await list_server_capabilities()
    )

    serialized_metadata = str(
        [
            server_info,
            health,
            capabilities,
        ]
    ).lower()

    forbidden_fragments = (
        "api_key",
        "apikey",
        "secret",
        "password",
        "token",
        "authorization",
        "postgresql://",
        "redis://",
    )

    for fragment in forbidden_fragments:
        assert (
            fragment not in serialized_metadata
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
