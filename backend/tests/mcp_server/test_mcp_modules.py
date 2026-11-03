"""
Comprehensive integration-style tests for MCP authentication, authorization,
request context, error responses, rate limiting, and workflow schemas.

These tests cover principal construction, scope and tenant authorization,
authenticated-context resolution, structured error responses, per-principal
rate limits, workflow start/review validation, and security boundaries.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    ValidationError,
)
from app.mcp_server.auth import (
    MCPPrincipal,
    build_principal_from_claims,
    ensure_tenant_access,
    get_authenticated_principal,
    require_scope,
)
from app.mcp_server.context import create_request_context
from app.mcp_server.errors import build_error_response
from app.mcp_server.rate_limit import MCPRateLimiter
from app.mcp_server.schemas import (
    WorkflowReviewRequest,
    WorkflowStartRequest,
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
def viewer_principal():
    return MCPPrincipal(
        user_id=str(uuid4()),
        tenant_id=str(uuid4()),
        role="viewer",
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
# Principal construction
# ---------------------------------------------------------------------------


def test_build_principal_from_claims():
    principal = build_principal_from_claims(
        {
            "sub": "user_1",
            "tenant_id": "tenant_1",
            "role": "reviewer",
            "scopes": [
                "workflows:read",
                "workflows:review",
            ],
        }
    )

    assert (
        principal.user_id == "user_1"
    )
    assert (
        principal.tenant_id
        == "tenant_1"
    )
    assert (
        principal.can_read_workflows
    )
    assert (
        principal.can_review_workflows
    )


def test_build_principal_requires_authentication():
    with pytest.raises(
        AuthenticationError
    ):
        build_principal_from_claims(
            {}
        )


@pytest.mark.parametrize(
    "claims",
    [
        {},
        {"sub": ""},
        {"sub": "   "},
        {"tenant_id": "tenant_1"},
        {"sub": "user_1"},
        {"sub": "", "tenant_id": ""},
        {"sub": None, "tenant_id": None},
    ],
)
def test_build_principal_rejects_incomplete_claims(
    claims,
):
    with pytest.raises(
        AuthenticationError
    ):
        build_principal_from_claims(
            claims
        )


def test_build_principal_defaults_role_to_viewer():
    principal = build_principal_from_claims(
        {
            "sub": "user_1",
            "tenant_id": "tenant_1",
        }
    )

    assert principal.role == "viewer"


def test_build_principal_defaults_to_empty_scopes():
    principal = build_principal_from_claims(
        {
            "sub": "user_1",
            "tenant_id": "tenant_1",
        }
    )

    assert principal.scopes == frozenset()


def test_build_principal_normalizes_scopes():
    principal = build_principal_from_claims(
        {
            "sub": "user_1",
            "tenant_id": "tenant_1",
            "scopes": [
                "workflows:read",
                "WORKFLOWS:READ",
                "  workflows:review  ",
            ],
        }
    )

    assert principal.scopes == frozenset(
        {
            "workflows:read",
            "workflows:review",
        }
    )


# ---------------------------------------------------------------------------
# Scope authorization
# ---------------------------------------------------------------------------


def test_require_scope(
    viewer_principal,
):
    require_scope(
        viewer_principal,
        scope="workflows:read",
    )

    with pytest.raises(
        AuthorizationError
    ):
        require_scope(
            viewer_principal,
            scope="workflows:write",
        )


@pytest.mark.parametrize(
    "scope",
    [
        "workflows:read",
        "workflows:write",
        "workflows:cancel",
        "workflows:review",
        "documents:read",
    ],
)
def test_require_scope_allows_authorized_scopes(
    principal,
    scope: str,
):
    require_scope(
        principal,
        scope=scope,
    )


@pytest.mark.parametrize(
    "scope",
    [
        "workflows:delete",
        "documents:write",
        "admin:all",
        "cms:publish",
    ],
)
def test_require_scope_rejects_unauthorized_scopes(
    viewer_principal,
    scope: str,
):
    with pytest.raises(
        AuthorizationError
    ):
        require_scope(
            viewer_principal,
            scope=scope,
        )


@pytest.mark.parametrize(
    "invalid_scope",
    [
        "",
        "   ",
        None,
        123,
        [],
        {},
    ],
)
def test_require_scope_rejects_invalid_scope(
    principal,
    invalid_scope,
):
    with pytest.raises(ValidationError):
        require_scope(
            principal,
            scope=invalid_scope,
        )


# ---------------------------------------------------------------------------
# Tenant authorization
# ---------------------------------------------------------------------------


def test_ensure_tenant_access(
    viewer_principal,
):
    ensure_tenant_access(
        viewer_principal,
        tenant_id=viewer_principal.tenant_id,
    )

    with pytest.raises(
        AuthorizationError
    ):
        ensure_tenant_access(
            viewer_principal,
            tenant_id="tenant_2",
        )


def test_ensure_tenant_access_allows_matching_tenant(
    principal,
):
    ensure_tenant_access(
        principal,
        tenant_id=principal.tenant_id,
    )


def test_ensure_tenant_access_rejects_cross_tenant_access(
    principal,
):
    with pytest.raises(
        AuthorizationError
    ):
        ensure_tenant_access(
            principal,
            tenant_id=str(uuid4()),
        )


@pytest.mark.parametrize(
    "invalid_tenant_id",
    [
        "",
        "   ",
        None,
        123,
        [],
        {},
    ],
)
def test_ensure_tenant_access_rejects_invalid_tenant_id(
    principal,
    invalid_tenant_id,
):
    with pytest.raises(ValidationError):
        ensure_tenant_access(
            principal,
            tenant_id=invalid_tenant_id,
        )


# ---------------------------------------------------------------------------
# Authenticated principal resolution
# ---------------------------------------------------------------------------


async def test_get_authenticated_principal(
    principal,
):
    context = SimpleNamespace(
        request_context=SimpleNamespace(
            principal=principal,
        )
    )

    assert (
        await get_authenticated_principal(
            context
        )
        == principal
    )


async def test_get_authenticated_principal_rejects_missing_context():
    with pytest.raises(
        AuthenticationError
    ):
        await get_authenticated_principal(
            None
        )


async def test_get_authenticated_principal_rejects_missing_request_context():
    context = SimpleNamespace()

    with pytest.raises(
        AuthenticationError
    ):
        await get_authenticated_principal(
            context
        )


async def test_get_authenticated_principal_rejects_missing_principal():
    context = SimpleNamespace(
        request_context=SimpleNamespace(
            principal=None,
        )
    )

    with pytest.raises(
        AuthenticationError
    ):
        await get_authenticated_principal(
            context
        )


# ---------------------------------------------------------------------------
# Request context
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
    assert context.request_id
    assert context.correlation_id


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


def test_create_request_context_rejects_missing_principal():
    with pytest.raises(
        AuthenticationError
    ):
        create_request_context(
            principal=None,
        )


# ---------------------------------------------------------------------------
# Error responses
# ---------------------------------------------------------------------------


def test_build_error_response():
    response = build_error_response(
        ValidationError("Invalid brief"),
        request_id="req_1",
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
        response["error"]["request_id"]
        == "req_1"
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
        (RateLimitError("slow"), "rate_limit_exceeded", 429),
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


def test_build_error_response_hides_internal_error_details():
    response = build_error_response(
        RuntimeError(
            "database password is hunter2"
        ),
        request_id="req_1",
    )

    serialized_response = str(
        response
    ).lower()

    assert (
        "hunter2"
        not in serialized_response
    )
    assert (
        "database"
        not in serialized_response
    )


def test_build_error_response_includes_retry_after_for_rate_limit():
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


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limiter():
    limiter = MCPRateLimiter(
        default_requests_per_minute=60,
        default_burst=2,
    )

    principal = MCPPrincipal(
        user_id="user_1",
        tenant_id="tenant_1",
        role="developer",
        scopes=frozenset(
            {"workflows:write"}
        ),
    )

    limiter.check_rate_limit(
        principal=principal,
        tool_name="start_generation_workflow",
    )
    limiter.check_rate_limit(
        principal=principal,
        tool_name="start_generation_workflow",
    )

    with pytest.raises(RateLimitError):
        limiter.check_rate_limit(
            principal=principal,
            tool_name="start_generation_workflow",
        )


def test_rate_limiter_is_per_principal_and_tool():
    limiter = MCPRateLimiter(
        default_requests_per_minute=60,
        default_burst=1,
    )

    first_principal = MCPPrincipal(
        user_id="user_1",
        tenant_id="tenant_1",
        role="developer",
        scopes=frozenset(
            {"workflows:write"}
        ),
    )

    second_principal = MCPPrincipal(
        user_id="user_2",
        tenant_id="tenant_1",
        role="developer",
        scopes=frozenset(
            {"workflows:write"}
        ),
    )

    limiter.check_rate_limit(
        principal=first_principal,
        tool_name="start_generation_workflow",
    )

    with pytest.raises(RateLimitError):
        limiter.check_rate_limit(
            principal=first_principal,
            tool_name="start_generation_workflow",
        )

    limiter.check_rate_limit(
        principal=second_principal,
        tool_name="start_generation_workflow",
    )


def test_rate_limiter_is_per_tool():
    limiter = MCPRateLimiter(
        default_requests_per_minute=60,
        default_burst=1,
    )

    principal = MCPPrincipal(
        user_id="user_1",
        tenant_id="tenant_1",
        role="developer",
        scopes=frozenset(
            {
                "workflows:write",
                "workflows:read",
            }
        ),
    )

    limiter.check_rate_limit(
        principal=principal,
        tool_name="start_generation_workflow",
    )

    with pytest.raises(RateLimitError):
        limiter.check_rate_limit(
            principal=principal,
            tool_name="start_generation_workflow",
        )

    limiter.check_rate_limit(
        principal=principal,
        tool_name="get_workflow_status",
    )


def test_rate_limiter_rejects_missing_principal():
    limiter = MCPRateLimiter()

    with pytest.raises(
        AuthenticationError
    ):
        limiter.check_rate_limit(
            principal=None,
            tool_name="start_generation_workflow",
        )


@pytest.mark.parametrize(
    "invalid_tool_name",
    [
        "",
        "   ",
        None,
        123,
        [],
        {},
    ],
)
def test_rate_limiter_rejects_invalid_tool_name(
    principal,
    invalid_tool_name,
):
    limiter = MCPRateLimiter()

    with pytest.raises(ValidationError):
        limiter.check_rate_limit(
            principal=principal,
            tool_name=invalid_tool_name,
        )


# ---------------------------------------------------------------------------
# Workflow start schema
# ---------------------------------------------------------------------------


def test_workflow_start_request():
    request = WorkflowStartRequest(
        brief_text=(
            "  Build a dashboard  "
        ),
    )

    assert request.brief_text == (
        "Build a dashboard"
    )
    assert (
        request.workflow_name
        == "design_to_code"
    )
    assert (
        request.approval_required
        is True
    )


@pytest.mark.parametrize(
    "invalid_brief_text",
    [
        "",
        "   ",
        None,
    ],
)
def test_workflow_start_request_rejects_invalid_brief_text(
    invalid_brief_text,
):
    with pytest.raises(ValidationError):
        WorkflowStartRequest(
            brief_text=invalid_brief_text
        )


def test_workflow_start_request_preserves_optional_fields():
    request = WorkflowStartRequest(
        brief_text="Build a dashboard.",
        workflow_name="custom_workflow",
        approval_required=False,
    )

    assert (
        request.workflow_name
        == "custom_workflow"
    )
    assert (
        request.approval_required
        is False
    )


# ---------------------------------------------------------------------------
# Workflow review schema
# ---------------------------------------------------------------------------


def test_workflow_review_request():
    request = WorkflowReviewRequest(
        execution_id=uuid4(),
        decision="request_changes",
        comment=(
            "  Clarify metrics.  "
        ),
    )

    assert request.comment == (
        "Clarify metrics."
    )


@pytest.mark.parametrize(
    "decision",
    [
        "approved",
        "rejected",
        "request_changes",
    ],
)
def test_workflow_review_request_accepts_supported_decisions(
    decision: str,
):
    request = WorkflowReviewRequest(
        execution_id=uuid4(),
        decision=decision,
    )

    assert request.decision == decision


@pytest.mark.parametrize(
    "invalid_decision",
    [
        "",
        "   ",
        "maybe",
        "unknown",
        "approve",
        "reject",
        "changes",
        None,
    ],
)
def test_workflow_review_request_rejects_invalid_decisions(
    invalid_decision,
):
    with pytest.raises(ValidationError):
        WorkflowReviewRequest(
            execution_id=uuid4(),
            decision=invalid_decision,
        )


@pytest.mark.parametrize(
    "invalid_execution_id",
    [
        "",
        "not-a-uuid",
        "123",
        None,
    ],
)
def test_workflow_review_request_rejects_invalid_execution_id(
    invalid_execution_id,
):
    with pytest.raises(ValidationError):
        WorkflowReviewRequest(
            execution_id=invalid_execution_id,
            decision="approved",
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
