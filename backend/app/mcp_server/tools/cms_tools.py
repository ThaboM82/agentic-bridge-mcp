"""
MCP CMS tools.

These tools expose a narrow, authenticated interface to AgenticBridge CMS
content: listing, inspecting, searching, creating drafts, and requesting
publication of approved content.

Security and architecture rules:
- Authenticate and authorize every CMS operation.
- Enforce tenant isolation through the authenticated principal.
- Treat all CMS arguments and content as untrusted input.
- Validate and bound every argument before it reaches a CMS service.
- Never return or log raw storage URLs, signed URLs, credentials, internal
  identifiers, customer PII, or unpublished draft contents unless explicitly
  authorized.
- CMS publication remains governed by workflow approval and CMS policy.
- Keep this module thin: content business rules belong in the CMS service.
"""

from __future__ import annotations

import logging
from typing import Any, Literal
from uuid import UUID

from mcp.server.fastmcp import Context, FastMCP  # type: ignore[import-not-found]

from app.core.exceptions import AuthorizationError
from app.mcp_server.auth import (
    get_authenticated_principal,
    require_scope,
)
from app.mcp_server.dependencies import get_cms_service
from app.mcp_server.errors import mcp_tool_error_boundary
from app.mcp_server.middleware import with_mcp_context
from app.mcp_server.rate_limit import rate_limiter
from app.mcp_server.schemas import (
    CMSContentCreateRequest,
    CMSContentListRequest,
    CMSContentSearchRequest,
    CMSContentSummaryRequest,
    CMSContentUpdateRequest,
    CMSPublishRequest,
)


logger = logging.getLogger(
    "agentic_bridge.mcp_server.tools.cms"
)

REQUIRED_CMS_READ_SCOPE = "cms:read"
REQUIRED_CMS_WRITE_SCOPE = "cms:write"
REQUIRED_CMS_PUBLISH_SCOPE = "cms:publish"

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

DEFAULT_SEARCH_LIMIT = 20
MAX_SEARCH_LIMIT = 100


def register_cms_tools(
    mcp: FastMCP,
) -> None:
    """Register CMS MCP tools."""

    @mcp.tool()
    @mcp_tool_error_boundary
    async def list_cms_content(
        content_status: str | None = None,
        content_type: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """
        List CMS content visible to the authenticated tenant.

        Args:
            content_status: Optional content-status filter.
            content_type: Optional content-type filter.
            page: Optional one-based page number.
            page_size: Optional page size.
            context: MCP request context.

        Returns:
            A bounded, paginated list of CMS content summaries.
        """
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope=REQUIRED_CMS_READ_SCOPE,
        )

        request = CMSContentListRequest(
            content_status=content_status,
            content_type=content_type,
            page=page or 1,
            page_size=page_size or DEFAULT_PAGE_SIZE,
        )

        cms_service = get_cms_service()

        async def list_content() -> Any:
            return await cms_service.list_content(
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                content_status=request.content_status,
                content_type=request.content_type,
                page=request.page,
                page_size=request.page_size,
            )

        result = await with_mcp_context(
            list_content,
            principal=principal,
            tool_name="list_cms_content",
        )

        logger.info(
            "mcp_cms_content_listed",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            page=request.page,
            page_size=request.page_size,
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )

    @mcp.tool()
    @mcp_tool_error_boundary
    async def get_cms_content(
        content_id: str,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """
        Get workflow-safe metadata and content for a CMS item.

        Args:
            content_id: CMS content identifier.
            context: MCP request context.

        Returns:
            A bounded CMS content response.

        Raises:
            AuthorizationError: If the caller cannot access the content.
        """
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope=REQUIRED_CMS_READ_SCOPE,
        )

        request = CMSContentSummaryRequest(
            content_id=content_id,
        )

        cms_service = get_cms_service()

        async def load_content() -> Any:
            return await cms_service.get_content(
                content_id=request.content_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )

        result = await with_mcp_context(
            load_content,
            principal=principal,
            tool_name="get_cms_content",
        )

        logger.info(
            "mcp_cms_content_retrieved",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            content_id=str(request.content_id),
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )

    @mcp.tool()
    @mcp_tool_error_boundary
    async def search_cms_content(
        query: str,
        content_type: str | None = None,
        limit: int | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """
        Search published or authorized CMS content.

        Args:
            query: Natural-language search query.
            content_type: Optional content-type filter.
            limit: Optional maximum number of results.
            context: MCP request context.

        Returns:
            Bounded CMS search results and citations.
        """
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope=REQUIRED_CMS_READ_SCOPE,
        )

        request = CMSContentSearchRequest(
            query=query,
            content_type=content_type,
            limit=limit or DEFAULT_SEARCH_LIMIT,
        )

        cms_service = get_cms_service()

        async def search_content() -> Any:
            return await cms_service.search_content(
                query=request.query,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                content_type=request.content_type,
                limit=request.limit,
            )

        result = await with_mcp_context(
            search_content,
            principal=principal,
            tool_name="search_cms_content",
        )

        logger.info(
            "mcp_cms_content_searched",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            result_count=len(result.results),
            content_type=request.content_type,
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )

    @mcp.tool()
    @mcp_tool_error_boundary
    async def create_cms_draft(
        title: str,
        content: str,
        content_type: str = "article",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """
        Create a CMS draft.

        Args:
            title: Content title.
            content: Draft body content.
            content_type: Content type.
            tags: Optional content tags.
            metadata: Optional non-sensitive metadata.
            context: MCP request context.

        Returns:
            A bounded CMS draft acknowledgement.
        """
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope=REQUIRED_CMS_WRITE_SCOPE,
        )

        request = CMSContentCreateRequest(
            title=title,
            content=content,
            content_type=content_type,
            tags=tags or [],
            metadata=metadata or {},
        )

        cms_service = get_cms_service()

        async def create_draft() -> Any:
            return await cms_service.create_draft(
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                title=request.title,
                content=request.content,
                content_type=request.content_type,
                tags=request.tags,
                metadata=request.metadata,
            )

        result = await with_mcp_context(
            create_draft,
            principal=principal,
            tool_name="create_cms_draft",
        )

        logger.info(
            "mcp_cms_draft_created",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            content_id=str(result.content_id),
            content_type=request.content_type,
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )

    @mcp.tool()
    @mcp_tool_error_boundary
    async def update_cms_draft(
        content_id: str,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """
        Update an editable CMS draft.

        Args:
            content_id: CMS content identifier.
            title: Optional updated title.
            content: Optional updated body content.
            tags: Optional updated tags.
            metadata: Optional updated non-sensitive metadata.
            context: MCP request context.

        Returns:
            A bounded CMS update acknowledgement.
        """
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope=REQUIRED_CMS_WRITE_SCOPE,
        )

        request = CMSContentUpdateRequest(
            content_id=content_id,
            title=title,
            content=content,
            tags=tags,
            metadata=metadata,
        )

        cms_service = get_cms_service()

        async def update_draft() -> Any:
            return await cms_service.update_draft(
                content_id=request.content_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                title=request.title,
                content=request.content,
                tags=request.tags,
                metadata=request.metadata,
            )

        result = await with_mcp_context(
            update_draft,
            principal=principal,
            tool_name="update_cms_draft",
        )

        logger.info(
            "mcp_cms_draft_updated",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            content_id=str(request.content_id),
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )

    @mcp.tool()
    @mcp_tool_error_boundary
    async def request_cms_publication(
        content_id: str,
        comment: str | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """
        Request publication of CMS content.

        This tool requests publication; it does not publish content directly.
        Publication remains subject to CMS policy and workflow approval.

        Args:
            content_id: CMS content identifier.
            comment: Optional publication request comment.
            context: MCP request context.

        Returns:
            A publication-request acknowledgement.
        """
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope=REQUIRED_CMS_WRITE_SCOPE,
        )

        request = CMSPublishRequest(
            content_id=content_id,
            comment=comment,
        )

        cms_service = get_cms_service()

        async def request_publication() -> Any:
            return await cms_service.request_publication(
                content_id=request.content_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                comment=request.comment,
            )

        result = await with_mcp_context(
            request_publication,
            principal=principal,
            tool_name="request_cms_publication",
        )

        logger.info(
            "mcp_cms_publication_requested",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            content_id=str(request.content_id),
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )


def _validate_content_type(
    content_type: str | None,
) -> str:
    """
    Validate and normalize a CMS content type.

    Args:
        content_type: Caller-supplied content type.

    Returns:
        A bounded content-type identifier.

    Raises:
        AuthorizationError: If the content type is invalid.
    """
    if content_type is None:
        return "article"

    if not isinstance(content_type, str):
        raise AuthorizationError(
            "CMS content type must be a string."
        )

    normalized_content_type = content_type.strip().lower()

    if not normalized_content_type:
        return "article"

    if len(normalized_content_type) > 100:
        raise AuthorizationError(
            "CMS content type exceeds the allowed length."
        )

    return normalized_content_type


# Configure conservative CMS rate limits.
rate_limiter.configure_tool(
    tool_name="search_cms_content",
    requests_per_minute=30,
    burst=10,
)

rate_limiter.configure_tool(
    tool_name="create_cms_draft",
    requests_per_minute=10,
    burst=5,
)

rate_limiter.configure_tool(
    tool_name="update_cms_draft",
    requests_per_minute=30,
    burst=10,
)

rate_limiter.configure_tool(
    tool_name="request_cms_publication",
    requests_per_minute=10,
    burst=5,
)