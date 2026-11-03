"""
MCP middleware helpers for request correlation, timing, audit logging,
authorization, and rate limiting.

These helpers add observability and policy enforcement at the MCP boundary
without exposing tool arguments, results, credentials, prompts, completions,
embeddings, generated code, or customer content.

Security and reliability rules:
- Authenticate and authorize every MCP tool invocation.
- Enforce tenant isolation through the authenticated principal.
- Treat tool arguments and results as untrusted input.
- Rate-limit by principal and tool name.
- Measure execution duration without recording arguments or results.
- Emit bounded, structured audit events.
- Middleware failures must never expose internal implementation details.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, ParamSpec, TypeVar
from uuid import UUID, uuid4

from app.core.exceptions import (
    AppError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
)
from app.mcp_server.auth import (
    MCPPrincipal,
    ensure_tenant_access,
    require_scope,
)
from app.mcp_server.context import (
    MCPRequestContext,
    create_request_context,
    log_context_event,
)
from app.mcp_server.errors import (
    build_error_response,
    sanitize_tool_arguments,
)
from app.mcp_server.rate_limit import rate_limiter

logger = logging.getLogger("agentic_bridge.mcp_server.middleware")

P = ParamSpec("P")
R = TypeVar("R")

MAX_TOOL_NAME_CHARS = 100
MAX_AUDIT_METADATA_KEYS = 50
MAX_AUDIT_METADATA_VALUE_CHARS = 1_000


# ---------------------------------------------------------------------------
# Request identity
# ---------------------------------------------------------------------------


def generate_mcp_request_id() -> str:
    """Generate a non-sensitive MCP request correlation ID."""
    return str(uuid4())


def generate_mcp_correlation_id() -> str:
    """Generate a non-sensitive MCP business correlation ID."""
    return str(uuid4())


def normalize_tool_name(
    tool_name: str | None,
) -> str:
    """
    Validate and normalize an MCP tool name.

    Args:
        tool_name: MCP tool name.

    Returns:
        A bounded tool name.

    Raises:
        ValidationError: If the tool name is missing or malformed.
    """
    from app.core.exceptions import ValidationError

    if not tool_name or not isinstance(tool_name, str):
        raise ValidationError(
            "MCP tool name is required."
        )

    normalized_tool_name = tool_name.strip()

    if not normalized_tool_name:
        raise ValidationError(
            "MCP tool name is required."
        )

    if len(normalized_tool_name) > MAX_TOOL_NAME_CHARS:
        raise ValidationError(
            "MCP tool name exceeds the allowed length."
        )

    return normalized_tool_name


# ---------------------------------------------------------------------------
# Authorization helpers
# ---------------------------------------------------------------------------


def authorize_tool_invocation(
    principal: MCPPrincipal,
    *,
    tool_name: str,
    required_scope: str,
    tenant_id: str | UUID | None = None,
) -> None:
    """
    Authorize an MCP tool invocation.

    Args:
        principal: Authenticated MCP principal.
        tool_name: MCP tool name.
        required_scope: Scope required for the tool.
        tenant_id: Optional tenant identifier for tenant-scoped resources.

    Raises:
        AuthorizationError: If the caller lacks the required scope or attempts
            to access another tenant's resources.
    """
    require_scope(
        principal,
        scope=required_scope,
    )

    if tenant_id is not None:
        ensure_tenant_access(
            principal,
            tenant_id=tenant_id,
        )

    logger.debug(
        "mcp_tool_authorized",
        tool_name=tool_name,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        required_scope=required_scope,
    )


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------


def sanitize_audit_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Sanitize and bound audit metadata.

    Args:
        metadata: Metadata supplied by an MCP tool.

    Returns:
        A bounded metadata dictionary safe for audit logging.
    """
    if not metadata or not isinstance(metadata, dict):
        return {}

    safe_metadata: dict[str, Any] = {}

    for key, value in list(metadata.items())[
        :MAX_AUDIT_METADATA_KEYS
    ]:
        if not isinstance(key, str):
            continue

        normalized_key = key.strip()

        if not normalized_key:
            continue

        key_lower = normalized_key.lower()

        sensitive_fragments = (
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
            "authorization",
            "credential",
            "cookie",
            "signed_url",
            "storage_url",
            "file_url",
            "download_url",
            "prompt",
            "completion",
            "messages",
            "input",
            "output",
            "embedding",
            "vector",
            "generated_code",
            "database_url",
            "connection_string",
        )

        if any(
            fragment in key_lower
            for fragment in sensitive_fragments
        ):
            safe_metadata[normalized_key] = "[REDACTED]"
            continue

        if value is None or isinstance(
            value,
            (bool, int, float),
        ):
            safe_metadata[normalized_key] = value
        elif isinstance(value, str):
            safe_metadata[normalized_key] = value[
                :MAX_AUDIT_METADATA_VALUE_CHARS
            ]
        elif isinstance(value, UUID):
            safe_metadata[normalized_key] = str(value)
        else:
            safe_metadata[normalized_key] = "[UNSERIALIZABLE]"

    return safe_metadata


def record_audit_event(
    context: MCPRequestContext,
    *,
    event_name: str,
    tool_name: str,
    outcome: Literal["started", "completed", "failed", "rate_limited"],
    duration_ms: float | None = None,
    error_code: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Record a bounded MCP audit event.

    Args:
        context: Request-scoped MCP context.
        event_name: Stable audit event name.
        tool_name: MCP tool name.
        outcome: Invocation outcome.
        duration_ms: Optional execution duration.
        error_code: Optional stable error code.
        metadata: Optional non-sensitive audit metadata.
    """
    audit_metadata = sanitize_audit_metadata(
        {
            **(metadata or {}),
            "outcome": outcome,
            "duration_ms": duration_ms,
            "error_code": error_code,
        }
    )

    log_context_event(
        context,
        event_name,
        tool_name=tool_name,
        **audit_metadata,
    )


# ---------------------------------------------------------------------------
# Tool execution middleware
# ---------------------------------------------------------------------------


async def with_mcp_context(
    func: Callable[P, Awaitable[R]],
    *args: P.args,
    principal: MCPPrincipal,
    tool_name: str,
    required_scope: str | None = None,
    tenant_id: str | UUID | None = None,
    request_id: str | None = None,
    correlation_id: str | None = None,
    audit_metadata: dict[str, Any] | None = None,
    **kwargs: P.kwargs,
) -> R:
    """
    Execute an MCP tool with context, authorization, rate limiting, timing,
    and audit telemetry.

    Args:
        func: Async MCP tool function.
        principal: Authenticated MCP principal.
        tool_name: MCP tool name.
        required_scope: Optional scope checked before execution.
        tenant_id: Optional tenant identifier for tenant-scoped resources.
        request_id: Optional request correlation ID.
        correlation_id: Optional business correlation ID.
        audit_metadata: Optional non-sensitive audit metadata.
        args: Positional tool arguments.
        kwargs: Keyword tool arguments.

    Returns:
        The tool result.

    Raises:
        AuthenticationError: If the caller is not authenticated.
        AuthorizationError: If the caller lacks required access.
        RateLimitError: If the caller exceeds the configured limit.
        AppError: If the tool fails.
    """
    normalized_tool_name = normalize_tool_name(tool_name)

    context = create_request_context(
        principal=principal,
        request_id=request_id,
        correlation_id=correlation_id,
    )

    if required_scope:
        authorize_tool_invocation(
            principal,
            tool_name=normalized_tool_name,
            required_scope=required_scope,
            tenant_id=tenant_id,
        )

    rate_limit_metadata = rate_limiter.check_rate_limit(
        principal=principal,
        tool_name=normalized_tool_name,
    )

    safe_arguments = sanitize_tool_arguments(
        kwargs
    )

    record_audit_event(
        context,
        event_name="mcp_tool_invocation_started",
        tool_name=normalized_tool_name,
        outcome="started",
        metadata={
            **safe_arguments,
            **rate_limit_metadata,
            **(audit_metadata or {}),
        },
    )

    started_at = time.perf_counter()

    try:
        result = await func(*args, **kwargs)

        duration_ms = (
            time.perf_counter() - started_at
        ) * 1_000

        record_audit_event(
            context,
            event_name="mcp_tool_invocation_completed",
            tool_name=normalized_tool_name,
            outcome="completed",
            duration_ms=round(duration_ms, 3),
            metadata=audit_metadata,
        )

        return result

    except RateLimitError as exc:
        duration_ms = (
            time.perf_counter() - started_at
        ) * 1_000

        record_audit_event(
            context,
            event_name="mcp_tool_invocation_rate_limited",
            tool_name=normalized_tool_name,
            outcome="rate_limited",
            duration_ms=round(duration_ms, 3),
            error_code="rate_limit_exceeded",
            metadata=audit_metadata,
        )

        raise

    except (AuthenticationError, AuthorizationError) as exc:
        duration_ms = (
            time.perf_counter() - started_at
        ) * 1_000

        error_response = build_error_response(
            exc,
            request_id=context.request_id,
            tool_name=normalized_tool_name,
        )

        record_audit_event(
            context,
            event_name="mcp_tool_invocation_failed",
            tool_name=normalized_tool_name,
            outcome="failed",
            duration_ms=round(duration_ms, 3),
            error_code=error_response["error"]["code"],
            metadata=audit_metadata,
        )

        raise

    except Exception as exc:
        duration_ms = (
            time.perf_counter() - started_at
        ) * 1_000

        error_response = build_error_response(
            exc,
            request_id=context.request_id,
            tool_name=normalized_tool_name,
        )

        record_audit_event(
            context,
            event_name="mcp_tool_invocation_failed",
            tool_name=normalized_tool_name,
            outcome="failed",
            duration_ms=round(duration_ms, 3),
            error_code=error_response["error"]["code"],
            metadata=audit_metadata,
        )

        raise


# ---------------------------------------------------------------------------
# Decorator middleware
# ---------------------------------------------------------------------------


def mcp_middleware(
    *,
    tool_name: str,
    required_scope: str | None = None,
) -> Callable[
    [Callable[P, Awaitable[R]]],
    Callable[P, Awaitable[R]],
]:
    """
    Decorator that applies MCP context, authorization, rate limiting, and
    audit telemetry to an async tool function.

    The decorated function must accept `principal` as a keyword argument.

    Args:
        tool_name: MCP tool name.
        required_scope: Optional required authorization scope.

    Returns:
        A decorator for an async MCP tool function.
    """

    def decorator(
        func: Callable[P, Awaitable[R]],
    ) -> Callable[P, Awaitable[R]]:
        async def wrapper(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R:
            principal = kwargs.get("principal")

            if not isinstance(
                principal,
                MCPPrincipal,
            ):
                raise AuthenticationError(
                    "MCP caller authentication is required."
                )

            return await with_mcp_context(
                func,
                principal=principal,
                tool_name=tool_name,
                required_scope=required_scope,
                *args,
                **kwargs,
            )

        return wrapper

    return decorator