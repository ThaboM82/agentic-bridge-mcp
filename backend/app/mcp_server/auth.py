"""
Authentication and authorization for the AgenticBridge MCP server.

The HTTP gateway or MCP transport adapter verifies caller identity and supplies
a trusted principal. This module validates that principal, checks required
scopes, and enforces tenant isolation.

Security rules:
- Never trust caller-supplied user or tenant IDs for authorization.
- Require an authenticated principal for every tool invocation.
- Use scope-based authorization for each tool category.
- Never log tokens, credentials, or raw authentication headers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
)


@dataclass(frozen=True)
class MCPPrincipal:
    """Authenticated MCP caller identity and authorization scopes."""

    user_id: str
    tenant_id: str
    role: str
    scopes: frozenset[str]

    @property
    def can_start_workflows(self) -> bool:
        """Whether the caller may start workflow executions."""
        return "workflows:write" in self.scopes

    @property
    def can_read_workflows(self) -> bool:
        """Whether the caller may read workflow execution status."""
        return "workflows:read" in self.scopes

    @property
    def can_cancel_workflows(self) -> bool:
        """Whether the caller may cancel workflow executions."""
        return "workflows:cancel" in self.scopes

    @property
    def can_review_workflows(self) -> bool:
        """Whether the caller may approve or reject workflows."""
        return "workflows:review" in self.scopes

    @property
    def can_read_documents(self) -> bool:
        """Whether the caller may read document metadata."""
        return "documents:read" in self.scopes

    @property
    def can_retrieve_context(self) -> bool:
        """Whether the caller may perform semantic retrieval."""
        return "documents:read" in self.scopes


def build_principal_from_claims(
    claims: dict[str, Any],
) -> MCPPrincipal:
    """
    Build an MCP principal from trusted authentication claims.

    Args:
        claims: Claims supplied by the verified authentication adapter.

    Returns:
        An authenticated MCP principal.

    Raises:
        AuthenticationError: If required claims are missing.
        AuthorizationError: If the principal has no scopes.
    """
    user_id = claims.get("sub") or claims.get("user_id")
    tenant_id = claims.get("tenant_id")
    role = claims.get("role") or "viewer"
    scopes = claims.get("scopes") or []

    if not user_id or not isinstance(user_id, str):
        raise AuthenticationError(
            "Authenticated caller is missing a user identity."
        )

    if not tenant_id or not isinstance(tenant_id, str):
        raise AuthenticationError(
            "Authenticated caller is missing a tenant identity."
        )

    if not isinstance(scopes, list):
        raise AuthorizationError(
            "Authenticated caller has invalid authorization scopes."
        )

    normalized_scopes = frozenset(
        scope.strip()
        for scope in scopes
        if isinstance(scope, str) and scope.strip()
    )

    if not normalized_scopes:
        raise AuthorizationError(
            "Authenticated caller has no authorization scopes."
        )

    return MCPPrincipal(
        user_id=user_id,
        tenant_id=tenant_id,
        role=str(role),
        scopes=normalized_scopes,
    )


async def get_authenticated_principal(
    context: Any,
) -> MCPPrincipal:
    """
    Resolve the authenticated MCP principal from a request context.

    Args:
        context: MCP request context.

    Returns:
        An authenticated MCP principal.

    Raises:
        AuthenticationError: If authentication context is unavailable.
    """
    request_context = getattr(
        context,
        "request_context",
        None,
    )

    if request_context is None:
        raise AuthenticationError(
            "MCP request context is unavailable."
        )

    principal = getattr(
        request_context,
        "principal",
        None,
    )

    if not isinstance(principal, MCPPrincipal):
        raise AuthenticationError(
            "MCP caller authentication is required."
        )

    if not principal.user_id or not principal.tenant_id:
        raise AuthorizationError(
            "Authenticated MCP caller is missing tenant or user identity."
        )

    return principal


def require_scope(
    principal: MCPPrincipal,
    *,
    scope: str,
) -> None:
    """
    Require an authorization scope.

    Args:
        principal: Authenticated MCP principal.
        scope: Required scope.

    Raises:
        AuthorizationError: If the scope is missing.
    """
    if scope not in principal.scopes:
        raise AuthorizationError(
            "Caller is not authorized to perform this operation."
        )


def ensure_tenant_access(
    principal: MCPPrincipal,
    *,
    tenant_id: str | UUID,
) -> None:
    """
    Ensure a caller may access a tenant-scoped resource.

    Args:
        principal: Authenticated MCP principal.
        tenant_id: Tenant identifier for the requested resource.

    Raises:
        AuthorizationError: If the tenant does not match the caller.
    """
    if principal.tenant_id != str(tenant_id):
        raise AuthorizationError(
            "Caller is not authorized to access this tenant resource."
        )