"""
Request-context models and helpers.

This module defines client-safe request correlation metadata, internal request
context construction, header extraction, elapsed-time tracking, and conversion
of internal contexts into API responses. It is designed to support tracing,
audit events, structured logging, and consistent debugging information.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


# ---------------------------------------------------------------------------
# Client-safe response models
# ---------------------------------------------------------------------------


class RequestContextResponse(BaseModel):
    """Client-safe request correlation metadata."""

    request_id: uuid.UUID = Field(
        description="Unique identifier for the current HTTP request.",
    )
    correlation_id: uuid.UUID = Field(
        description=(
            "Identifier used to correlate a request across services."
        ),
    )
    tenant_id: uuid.UUID = Field(
        description="Tenant associated with the authenticated principal.",
    )
    user_id: uuid.UUID = Field(
        description="Authenticated user associated with the request.",
    )
    role: str = Field(
        description="Primary role assigned to the authenticated principal.",
    )
    scopes: list[str] = Field(
        description="Authorization scopes granted to the principal.",
    )
    elapsed_ms: int = Field(
        ge=0,
        description="Elapsed request time in milliseconds.",
    )
    path: str | None = Field(
        default=None,
        description="HTTP request path.",
    )
    method: str | None = Field(
        default=None,
        description="HTTP request method.",
    )
    client_ip: str | None = Field(
        default=None,
        description="Client IP address, when available.",
    )
    user_agent: str | None = Field(
        default=None,
        description="Client user-agent header, when available.",
    )
    started_at: datetime = Field(
        description="UTC timestamp when request processing started.",
    )

    @field_validator(
        "scopes"
    )
    @classmethod
    def sort_scopes(
        cls,
        value: list[str],
    ) -> list[str]:
        """Return scopes in a stable, deterministic order."""
        return sorted(
            set(value)
        )


# ---------------------------------------------------------------------------
# Internal request context
# ---------------------------------------------------------------------------


class RequestContext:
    """Internal request context used for logging, tracing, and auditing."""

    def __init__(
        self,
        *,
        request_id: str | uuid.UUID | None = None,
        correlation_id: str | uuid.UUID | None = None,
        tenant_id: str | uuid.UUID | None = None,
        user_id: str | uuid.UUID | None = None,
        role: str = "anonymous",
        scopes: list[str] | tuple[str, ...] | set[str] | None = None,
        path: str | None = None,
        method: str | None = None,
        client_ip: str | None = None,
        user_agent: str | None = None,
        started_at: float | None = None,
    ) -> None:
        """Create an internal request context."""
        self.request_id = (
            str(request_id)
            if request_id
            else str(uuid.uuid4())
        )

        self.correlation_id = (
            str(correlation_id)
            if correlation_id
            else self.request_id
        )

        self.tenant_id = (
            str(tenant_id)
            if tenant_id
            else ""
        )

        self.user_id = (
            str(user_id)
            if user_id
            else ""
        )

        self.role = role

        self.scopes = sorted(
            set(scopes or [])
        )

        self.path = path
        self.method = method
        self.client_ip = client_ip
        self.user_agent = user_agent

        self.started_at = (
            started_at
            if started_at is not None
            else time.perf_counter()
        )

        self.created_at = datetime.now(
            timezone.utc
        )

    def elapsed_ms(self) -> int:
        """Return elapsed processing time in whole milliseconds."""
        return max(
            int(
                (
                    time.perf_counter()
                    - self.started_at
                )
                * 1_000
            ),
            0,
        )

    def elapsed_seconds(self) -> float:
        """Return elapsed processing time in seconds."""
        return max(
            time.perf_counter()
            - self.started_at,
            0.0,
        )

    def has_scope(
        self,
        required_scope: str,
    ) -> bool:
        """Return whether the context includes a required scope."""
        return required_scope in self.scopes

    def has_any_scope(
        self,
        required_scopes: list[str] | tuple[str, ...],
    ) -> bool:
        """Return whether the context includes any required scope."""
        return any(
            required_scope in self.scopes
            for required_scope in required_scopes
        )

    def to_log_context(self) -> dict[str, Any]:
        """Return a structured dictionary suitable for logging."""
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "role": self.role,
            "scopes": self.scopes,
            "path": self.path,
            "method": self.method,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "elapsed_ms": self.elapsed_ms(),
            "created_at": self.created_at.isoformat(),
        }

    def to_audit_context(self) -> dict[str, Any]:
        """Return a compact context suitable for audit-event metadata."""
        return {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "role": self.role,
            "path": self.path,
            "method": self.method,
            "client_ip": self.client_ip,
        }

    def to_response(
        self,
    ) -> RequestContextResponse:
        """Convert the internal context into a client-safe response."""
        return build_request_context_response(
            self
        )


# ---------------------------------------------------------------------------
# Header extraction
# ---------------------------------------------------------------------------


def get_header_value(
    headers: Mapping[str, str],
    header_name: str,
    default: str | None = None,
) -> str | None:
    """Retrieve a header value case-insensitively."""
    normalized_name = (
        header_name.lower()
    )

    for key, value in headers.items():
        if key.lower() == normalized_name:
            return value

    return default


def extract_request_context_from_headers(
    headers: Mapping[str, str],
    *,
    tenant_id: str | uuid.UUID | None = None,
    user_id: str | uuid.UUID | None = None,
    role: str = "anonymous",
    scopes: list[str] | tuple[str, ...] | set[str] | None = None,
    path: str | None = None,
    method: str | None = None,
) -> RequestContext:
    """Build a request context from HTTP headers and principal data."""
    request_id = get_header_value(
        headers,
        "x-request-id",
    )

    correlation_id = get_header_value(
        headers,
        "x-correlation-id",
        default=request_id,
    )

    client_ip = get_header_value(
        headers,
        "x-forwarded-for",
    )

    if client_ip:
        client_ip = client_ip.split(
            ",",
            maxsplit=1,
        )[0].strip()

    user_agent = get_header_value(
        headers,
        "user-agent",
    )

    return RequestContext(
        request_id=request_id,
        correlation_id=correlation_id,
        tenant_id=tenant_id,
        user_id=user_id,
        role=role,
        scopes=scopes,
        path=path,
        method=method,
        client_ip=client_ip,
        user_agent=user_agent,
    )


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------


def build_request_context_response(
    context: Any,
) -> RequestContextResponse:
    """Convert an internal request context into a safe API response."""
    return RequestContextResponse(
        request_id=uuid.UUID(
            str(
                context.request_id
            )
        ),
        correlation_id=uuid.UUID(
            str(
                context.correlation_id
            )
        ),
        tenant_id=uuid.UUID(
            str(
                context.tenant_id
            )
        ),
        user_id=uuid.UUID(
            str(
                context.user_id
            )
        ),
        role=context.role,
        scopes=list(
            context.scopes
        ),
        elapsed_ms=context.elapsed_ms(),
        path=getattr(
            context,
            "path",
            None,
        ),
        method=getattr(
            context,
            "method",
            None,
        ),
        client_ip=getattr(
            context,
            "client_ip",
            None,
        ),
        user_agent=getattr(
            context,
            "user_agent",
            None,
        ),
        started_at=getattr(
            context,
            "created_at",
            datetime.now(
                timezone.utc
            ),
        ),
    )