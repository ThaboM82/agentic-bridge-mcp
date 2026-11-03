"""
Figma MCP tools.

These tools expose a secure, tenant-scoped MCP interface for inspecting Figma
design context. They are read-only and never expose raw Figma API responses,
provider credentials, storage URLs, design source files, or customer PII.

Security rules:
- Authenticate and authorize every tool invocation.
- Enforce tenant isolation on every Figma operation.
- Treat all MCP tool arguments as untrusted input.
- Validate and bound every argument before it reaches the Figma service.
- Never return raw Figma API payloads, access tokens, file contents, image
  URLs, comments, or customer PII.
- Never log tool arguments or results that may contain sensitive material.
- Keep tools thin: Figma API access, normalization, caching, and business
  rules belong in the Figma application service.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from mcp.server.fastmcp import (
    Context,
    FastMCP,
)

from app.core.exceptions import (
    ValidationError,
)
from app.core.utils import truncate_text
from app.schemas.figma import (
    FigmaDesignContextRequest,
    FigmaDesignContextResponse,
    FigmaFrameRequest,
    FigmaFrameResponse,
)
from app.services.figma_service import (
    FigmaService,
)
from app.mcp_server.auth import (
    get_authenticated_principal,
)
from app.mcp_server.errors import (
    mcp_tool_error_boundary,
)
from app.mcp_server.middleware import (
    with_mcp_context,
)
from app.mcp_server.validation import (
    validate_optional_positive_int,
    validate_required_string,
    validate_uuid,
)


logger = logging.getLogger(
    "agentic_bridge.mcp_server.figma_tools",
)


MAX_FIGMA_FILE_KEY_CHARS = 100
MAX_FIGMA_NODE_ID_CHARS = 100
MAX_FIGMA_QUERY_CHARS = 500
MAX_FIGMA_LIMIT = 50
DEFAULT_FIGMA_LIMIT = 20


def _safe_file_key(
    value: str,
) -> str:
    """
    Validate and normalize a Figma file key.
    """
    normalized = validate_required_string(
        value,
        field_name="figma_file_key",
        max_length=MAX_FIGMA_FILE_KEY_CHARS,
    )

    return normalized.strip()


def _safe_node_id(
    value: str,
) -> str:
    """
    Validate and normalize a Figma node identifier.
    """
    normalized = validate_required_string(
        value,
        field_name="figma_node_id",
        max_length=MAX_FIGMA_NODE_ID_CHARS,
    )

    return normalized.strip()


def _safe_query(
    value: str | None,
) -> str | None:
    """
    Validate an optional Figma design-intent query.
    """
    if value is None:
        return None

    normalized = validate_required_string(
        value,
        field_name="query",
        max_length=MAX_FIGMA_QUERY_CHARS,
    )

    return normalized.strip() or None


def _safe_limit(
    value: int | None,
) -> int:
    """
    Validate an optional bounded result limit.
    """
    return validate_optional_positive_int(
        value,
        field_name="limit",
        default_value=DEFAULT_FIGMA_LIMIT,
        max_value=MAX_FIGMA_LIMIT,
    )


def register_figma_tools(
    mcp: FastMCP,
) -> None:
    """
    Register Figma-related MCP tools.

    Args:
        mcp: MCP server instance.
    """

    @mcp.tool()
    @mcp_tool_error_boundary
    async def get_figma_design_context(
        context: Context,
        *,
        figma_file_key: str,
        query: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve workflow-safe Figma design context.

        Args:
            context: MCP request context.
            figma_file_key: Figma file key.
            query: Optional design-intent query.
            limit: Optional maximum number of design context items.

        Returns:
            Tenant-safe Figma design context, including bounded design
            summaries and component metadata. Raw Figma API responses, image
            URLs, source file contents, and credentials are never returned.

        Raises:
            AuthenticationError: If the caller is not authenticated.
            AuthorizationError: If the caller lacks required access.
            ValidationError: If arguments are invalid.
            NotFoundError: If the Figma file cannot be found.
            RateLimitError: If the caller exceeds the configured limit.
        """
        principal = (
            await get_authenticated_principal(
                context,
            )
        )

        validated_file_key = _safe_file_key(
            figma_file_key,
        )
        validated_query = _safe_query(
            query,
        )
        validated_limit = _safe_limit(
            limit,
        )

        request = FigmaDesignContextRequest(
            figma_file_key=validated_file_key,
            query=validated_query,
            limit=validated_limit,
        )

        async def load_design_context(
            *,
            tenant_id: str,
            user_id: str,
            request: FigmaDesignContextRequest,
        ) -> FigmaDesignContextResponse:
            figma_service = FigmaService()

            return (
                await figma_service.get_design_context(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    request=request,
                )
            )

        response = await with_mcp_context(
            load_design_context,
            principal=principal,
            tool_name=(
                "get_figma_design_context"
            ),
            required_scope="figma:read",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            request=request,
        )

        logger.info(
            "figma_design_context_retrieved",
            extra={
                "tenant_id": principal.tenant_id,
                "user_id": principal.user_id,
                "limit": validated_limit,
                "has_query": bool(
                    validated_query,
                ),
            },
        )

        return response.model_dump(
            mode="json",
            exclude_none=True,
        )

    @mcp.tool()
    @mcp_tool_error_boundary
    async def get_figma_frame_summary(
        context: Context,
        *,
        figma_file_key: str,
        figma_node_id: str,
    ) -> dict[str, Any]:
        """
        Retrieve a workflow-safe summary for a Figma frame or component.

        Args:
            context: MCP request context.
            figma_file_key: Figma file key.
            figma_node_id: Figma node identifier.

        Returns:
            A bounded Figma frame summary containing safe metadata such as
            frame name, component type, dimensions, and design intent. Raw
            Figma payloads, image URLs, source file contents, and credentials
            are never returned.

        Raises:
            AuthenticationError: If the caller is not authenticated.
            AuthorizationError: If the caller lacks required access.
            ValidationError: If arguments are invalid.
            NotFoundError: If the Figma file or node cannot be found.
            RateLimitError: If the caller exceeds the configured limit.
        """
        principal = (
            await get_authenticated_principal(
                context,
            )
        )

        validated_file_key = _safe_file_key(
            figma_file_key,
        )
        validated_node_id = _safe_node_id(
            figma_node_id,
        )

        request = FigmaFrameRequest(
            figma_file_key=validated_file_key,
            figma_node_id=validated_node_id,
        )

        async def load_frame_summary(
            *,
            tenant_id: str,
            user_id: str,
            request: FigmaFrameRequest,
        ) -> FigmaFrameResponse:
            figma_service = FigmaService()

            return (
                await figma_service.get_frame_summary(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    request=request,
                )
            )

        response = await with_mcp_context(
            load_frame_summary,
            principal=principal,
            tool_name=(
                "get_figma_frame_summary"
            ),
            required_scope="figma:read",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            request=request,
        )

        logger.info(
            "figma_frame_summary_retrieved",
            extra={
                "tenant_id": principal.tenant_id,
                "user_id": principal.user_id,
            },
        )

        return response.model_dump(
            mode="json",
            exclude_none=True,
        )

    logger.info(
        "figma_tools_registered",
        extra={
            "tools": [
                "get_figma_design_context",
                "get_figma_frame_summary",
            ],
        },
    )


__all__ = [
    "register_figma_tools",
]