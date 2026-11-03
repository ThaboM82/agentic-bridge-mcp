"""
Request-scoped context helpers for the AgenticBridge MCP server.

This module creates correlation IDs, binds structured logging context, tracks
request timing, and provides safe access to authenticated caller information.

Security and observability rules:
- Never trust caller-supplied user or tenant identities.
- Bind only non-sensitive identity and correlation fields to logs.
- Treat event metadata as untrusted input.
- Bound and sanitize all logged metadata.
- Never log tool arguments, results, credentials, prompts, completions,
  embeddings, generated code, storage URLs, or customer PII.
- Keep context creation synchronous and cheap; authentication remains async.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from app.core.exceptions import AuthenticationError
from app.core.logging_context import bind_workflow_context
from app.core.utils import truncate_text
from app.mcp_server.auth import MCPPrincipal


logger = logging.getLogger("agentic_bridge.mcp_server.context")

MAX_REQUEST_ID_CHARS = 128
MAX_CORRELATION_ID_CHARS = 128
MAX_EVENT_NAME_CHARS = 200
MAX_METADATA_KEYS = 50
MAX_METADATA_VALUE_CHARS = 1_000
MAX_LIST_ITEMS = 50

REDACTED_VALUE = "[REDACTED]"
UNSERIALIZABLE_VALUE = "[UNSERIALIZABLE]"

SENSITIVE_KEY_FRAGMENTS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "cookie",
    "session",
    "private_key",
    "access_key",
    "refresh_token",
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
    "stack_trace",
    "traceback",
)


# ---------------------------------------------------------------------------
# Request context
# ---------------------------------------------------------------------------


@dataclass
class MCPRequestContext:
    """
    Request-scoped MCP execution context.

    The context carries only non-sensitive correlation and identity data.
    Tool arguments and results must never be attached to this object.
    """

    request_id: str
    correlation_id: str
    principal: MCPPrincipal
    started_at_monotonic: float
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def tenant_id(self) -> str:
        """Authenticated tenant identifier."""
        return self.principal.tenant_id

    @property
    def user_id(self) -> str:
        """Authenticated user identifier."""
        return self.principal.user_id

    @property
    def role(self) -> str:
        """Authenticated caller role."""
        return self.principal.role

    @property
    def scopes(self) -> frozenset[str]:
        """Authenticated caller scopes."""
        return self.principal.scopes

    def elapsed_ms(self) -> float:
        """
        Return elapsed request time in milliseconds.

        Returns:
            Elapsed duration since context creation.
        """
        return (
            time.perf_counter()
            - self.started_at_monotonic
        ) * 1_000

    def set_attribute(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Set a bounded, non-sensitive context attribute.

        Args:
            key: Attribute name.
            value: Attribute value.

        Raises:
            ValidationError: If the key is invalid.
        """
        from app.core.exceptions import ValidationError

        if not key or not isinstance(
            key,
            str,
        ):
            raise ValidationError(
                "Context attribute key must be a non-empty string."
            )

        normalized_key = key.strip()

        if not normalized_key:
            raise ValidationError(
                "Context attribute key must be a non-empty string."
            )

        if len(self.attributes) >= MAX_METADATA_KEYS:
            raise ValidationError(
                "Context attribute limit exceeded."
            )

        key_lower = normalized_key.lower()

        if any(
            fragment in key_lower
            for fragment in SENSITIVE_KEY_FRAGMENTS
        ):
            self.attributes[normalized_key] = REDACTED_VALUE
            return

        self.attributes[normalized_key] = (
            sanitize_context_value(value)
        )

    def get_attribute(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get a context attribute.

        Args:
            key: Attribute name.
            default: Value returned when the attribute is absent.

        Returns:
            The attribute value or default.
        """
        return self.attributes.get(
            key,
            default,
        )


# ---------------------------------------------------------------------------
# Identifier helpers
# ---------------------------------------------------------------------------


def generate_mcp_request_id() -> str:
    """Generate a non-sensitive MCP request correlation ID."""
    return str(uuid4())


def generate_mcp_correlation_id() -> str:
    """Generate a non-sensitive MCP business correlation ID."""
    return str(uuid4())


def normalize_context_id(
    value: str | UUID | None,
    *,
    field_name: str,
    max_length: int,
) -> str:
    """
    Normalize and bound a context identifier.

    Args:
        value: Caller-supplied or generated identifier.
        field_name: Identifier field name.
        max_length: Maximum accepted identifier length.

    Returns:
        A bounded identifier string.

    Raises:
        ValidationError: If the identifier is malformed.
    """
    from app.core.exceptions import ValidationError

    if value is None:
        return str(uuid4())

    if isinstance(value, UUID):
        return str(value)

    if not isinstance(value, str):
        raise ValidationError(
            f"{field_name} must be a string or UUID."
        )

    normalized_value = value.strip()

    if not normalized_value:
        return str(uuid4())

    if len(normalized_value) > max_length:
        raise ValidationError(
            f"{field_name} exceeds the allowed length."
        )

    return normalized_value


# ---------------------------------------------------------------------------
# Context creation
# ---------------------------------------------------------------------------


def create_request_context(
    *,
    principal: MCPPrincipal,
    request_id: str | UUID | None = None,
    correlation_id: str | UUID | None = None,
) -> MCPRequestContext:
    """
    Create a request-scoped MCP context.

    Args:
        principal: Authenticated MCP principal.
        request_id: Optional request correlation ID.
        correlation_id: Optional business correlation ID.

    Returns:
        A request-scoped MCP context.

    Raises:
        AuthenticationError: If the principal is missing or incomplete.
        ValidationError: If supplied identifiers are malformed.
    """
    if principal is None:
        raise AuthenticationError(
            "MCP caller authentication is required."
        )

    if not principal.user_id or not principal.tenant_id:
        raise AuthenticationError(
            "Authenticated MCP caller is missing tenant or user identity."
        )

    normalized_request_id = normalize_context_id(
        request_id,
        field_name="request_id",
        max_length=MAX_REQUEST_ID_CHARS,
    )

    normalized_correlation_id = normalize_context_id(
        correlation_id,
        field_name="correlation_id",
        max_length=MAX_CORRELATION_ID_CHARS,
    )

    bind_workflow_context(
        normalized_correlation_id,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
    )

    return MCPRequestContext(
        request_id=normalized_request_id,
        correlation_id=normalized_correlation_id,
        principal=principal,
        started_at_monotonic=time.perf_counter(),
    )


# ---------------------------------------------------------------------------
# Principal access
# ---------------------------------------------------------------------------


def get_context_principal(
    context: MCPRequestContext,
) -> MCPPrincipal:
    """
    Return the authenticated principal attached to an MCP request context.

    Args:
        context: Request-scoped MCP context.

    Returns:
        The authenticated principal.

    Raises:
        AuthenticationError: If no principal is attached.
    """
    if context is None:
        raise AuthenticationError(
            "MCP request context is unavailable."
        )

    principal = getattr(
        context,
        "principal",
        None,
    )

    if not isinstance(
        principal,
        MCPPrincipal,
    ):
        raise AuthenticationError(
            "MCP caller authentication is required."
        )

    if not principal.user_id or not principal.tenant_id:
        raise AuthenticationError(
            "Authenticated MCP caller is missing tenant or user identity."
        )

    return principal


# ---------------------------------------------------------------------------
# Metadata sanitization
# ---------------------------------------------------------------------------


def sanitize_context_value(
    value: Any,
    *,
    max_length: int = MAX_METADATA_VALUE_CHARS,
) -> Any:
    """
    Sanitize and bound a context metadata value.

    Args:
        value: Arbitrary metadata value.
        max_length: Maximum string length.

    Returns:
        A JSON-safe, bounded representation of the value.
    """
    if value is None or isinstance(
        value,
        (bool, int, float),
    ):
        return value

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, str):
        return truncate_text(
            value.strip(),
            max_length,
        )

    if isinstance(value, dict):
        return sanitize_context_metadata(value)

    if isinstance(value, (list, tuple, set)):
        return [
            sanitize_context_value(
                item,
                max_length=max_length,
            )
            for item in list(value)[:MAX_LIST_ITEMS]
        ]

    return UNSERIALIZABLE_VALUE


def sanitize_context_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Sanitize and bound context metadata for structured logging.

    Args:
        metadata: Arbitrary metadata supplied by a tool or middleware.

    Returns:
        A bounded metadata dictionary with sensitive values redacted.
    """
    if not metadata or not isinstance(metadata, dict):
        return {}

    safe_metadata: dict[str, Any] = {}

    for key, value in list(metadata.items())[:MAX_METADATA_KEYS]:
        if not isinstance(key, str):
            continue

        normalized_key = key.strip()

        if not normalized_key:
            continue

        key_lower = normalized_key.lower()

        if any(
            fragment in key_lower
            for fragment in SENSITIVE_KEY_FRAGMENTS
        ):
            safe_metadata[normalized_key] = REDACTED_VALUE
            continue

        safe_metadata[normalized_key] = sanitize_context_value(
            value
        )

    return safe_metadata


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_context_event(
    context: MCPRequestContext,
    event_name: str,
    *,
    level: int = logging.INFO,
    **metadata: Any,
) -> None:
    """
    Emit a bounded, context-aware MCP log event.

    Args:
        context: Request-scoped MCP context.
        event_name: Stable event name.
        level: Logging level.
        metadata: Non-sensitive event metadata.

    Raises:
        ValidationError: If the event name is invalid.
    """
    from app.core.exceptions import ValidationError

    if not event_name or not isinstance(
        event_name,
        str,
    ):
        raise ValidationError(
            "Context event name must be a non-empty string."
        )

    normalized_event_name = event_name.strip()

    if not normalized_event_name:
        raise ValidationError(
            "Context event name must be a non-empty string."
        )

    if len(normalized_event_name) > MAX_EVENT_NAME_CHARS:
        raise ValidationError(
            "Context event name exceeds the allowed length."
        )

    safe_metadata = sanitize_context_metadata(
        metadata
    )

    logger.log(
        level,
        normalized_event_name,
        request_id=context.request_id,
        correlation_id=context.correlation_id,
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        **safe_metadata,
    )


def log_context_error(
    context: MCPRequestContext,
    event_name: str,
    exc: BaseException,
    **metadata: Any,
) -> None:
    """
    Emit a bounded, context-aware MCP error event.

    Args:
        context: Request-scoped MCP context.
        event_name: Stable event name.
        exc: Exception associated with the event.
        metadata: Non-sensitive event metadata.
    """
    safe_metadata = sanitize_context_metadata(
        metadata
    )

    logger.error(
        event_name,
        request_id=context.request_id,
        correlation_id=context.correlation_id,
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        error_type=type(exc).__name__,
        **safe_metadata,
        exc_info=exc,
    )