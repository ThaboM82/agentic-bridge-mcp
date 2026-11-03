"""
WordPress MCP tools.

These tools expose a secure, tenant-scoped MCP interface for searching,
creating, updating, and requesting publication of WordPress content. They
never expose WordPress credentials, raw provider responses, database URLs,
storage URLs, customer PII, or unpublished content to unauthorized callers.

Security and architecture rules:
- Authenticate and authorize every tool invocation.
- Enforce tenant isolation on every WordPress operation.
- Treat all MCP tool arguments as untrusted input.
- Validate and bound every argument before it reaches the WordPress service.
- Never return raw WordPress API payloads, credentials, database URLs,
  storage URLs, customer PII, prompts, completions, embeddings, or generated
  file contents.
- Never log tool arguments or results that may contain sensitive material.
- Keep tools thin: WordPress API access, normalization, caching, publishing
  policy, and business rules belong in the WordPress application service.
- Human approval remains required before publication when configured.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from mcp.server.fastmcp import Context, FastMCP

from app.core.exceptions import ValidationError
from app.core.utils import truncate_text
from app.schemas.wordpress import (
    WordPressContentSearchRequest,
    WordPressContentSearchResponse,
    WordPressCreateDraftRequest,
    WordPressCreateDraftResponse,
    WordPressPostRequest,
    WordPressPostResponse,
    WordPressUpdateDraftRequest,
    WordPressUpdateDraftResponse,
)
from app.services.wordpress_service import WordPressService
from app.mcp_server.auth import (
    get_authenticated_principal,
)
from app.mcp_server.errors import mcp_tool_error_boundary
from app.mcp_server.middleware import with_mcp_context
from app.mcp_server.validation import (
    validate_optional_positive_int,
    validate_required_string,
)


logger = logging.getLogger("agentic_bridge.mcp_server.wordpress_tools")

MAX_WORDPRESS_QUERY_CHARS = 500
MAX_WORDPRESS_TITLE_CHARS = 300
MAX_WORDPRESS_SLUG_CHARS = 200
MAX_WORDPRESS_CONTENT_CHARS = 50_000
MAX_WORDPRESS_EXCERPT_CHARS = 1_000
MAX_WORDPRESS_TAG_CHARS = 100
MAX_WORDPRESS_TAGS = 25
MAX_WORDPRESS_CATEGORY_IDS = 25
MAX_WORDPRESS_LIMIT = 50
DEFAULT_WORDPRESS_LIMIT = 20


def _validate_tags(
    tags: list[str] | None,
) -> list[str]:
    """
    Validate and bound WordPress tags.

    Args:
        tags: Optional list of tags.

    Returns:
        A bounded list of normalized tags.

    Raises:
        ValidationError: If tags are malformed.
    """
    if tags is None:
        return []

    if not isinstance(tags, list):
        raise ValidationError(
            "tags must be a list of strings."
        )

    if len(tags) > MAX_WORDPRESS_TAGS:
        raise ValidationError(
            f"tags cannot contain more than {MAX_WORDPRESS_TAGS} items."
        )

    normalized_tags: list[str] = []

    for tag in tags:
        if not isinstance(tag, str):
            raise ValidationError(
                "Each tag must be a string."
            )

        normalized_tag = tag.strip()

        if not normalized_tag:
            continue

        normalized_tags.append(
            truncate_text(
                normalized_tag,
                MAX_WORDPRESS_TAG_CHARS,
            )
        )

    return normalized_tags


def _validate_category_ids(
    category_ids: list[str | UUID] | None,
) -> list[str]:
    """
    Validate and bound WordPress category identifiers.

    Args:
        category_ids: Optional list of category IDs.

    Returns:
        A bounded list of category ID strings.

    Raises:
        ValidationError: If category IDs are malformed.
    """
    if category_ids is None:
        return []

    if not isinstance(category_ids, list):
        raise ValidationError(
            "category_ids must be a list of IDs."
        )

    if len(category_ids) > MAX_WORDPRESS_CATEGORY_IDS:
        raise ValidationError(
            "category_ids cannot contain more than "
            f"{MAX_WORDPRESS_CATEGORY_IDS} items."
        )

    normalized_category_ids: list[str] = []

    for category_id in category_ids:
        if isinstance(category_id, UUID):
            normalized_category_ids.append(
                str(category_id)
            )
            continue

        if not isinstance(category_id, str):
            raise ValidationError(
                "Each category ID must be a string or UUID."
            )

        normalized_category_id = category_id.strip()

        if not normalized_category_id:
            continue

        normalized_category_ids.append(
            truncate_text(
                normalized_category_id,
                100,
            )
        )

    return normalized_category_ids


def register_wordpress_tools(
    mcp: FastMCP,
) -> None:
    """
    Register WordPress-related MCP tools.

    Args:
        mcp: MCP server instance.
    """

    @mcp.tool()
    @mcp_tool_error_boundary
    async def search_wordpress_content(
        context: Context,
        *,
        query: str,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Search tenant-scoped WordPress content.

        Args:
            context: MCP request context.
            query: Search query.
            limit: Optional maximum number of results.

        Returns:
            Tenant-safe WordPress search results, including bounded titles,
            excerpts, status, and content identifiers. Raw WordPress API
            responses, credentials, storage URLs, and full post content are
            never returned.

        Raises:
            AuthenticationError: If the caller is not authenticated.
            AuthorizationError: If the caller lacks required access.
            ValidationError: If arguments are invalid.
            RateLimitError: If the caller exceeds the configured limit.
        """
        principal = await get_authenticated_principal(
            context
        )

        validated_query = validate_required_string(
            query,
            field_name="query",
            max_length=MAX_WORDPRESS_QUERY_CHARS,
        )

        validated_limit = (
            validate_optional_positive_int(
                limit,
                field_name="limit",
                default_value=DEFAULT_WORDPRESS_LIMIT,
                max_value=MAX_WORDPRESS_LIMIT,
            )
        )

        request = WordPressContentSearchRequest(
            query=validated_query,
            limit=validated_limit,
        )

        async def search_content(
            *,
            tenant_id: str,
            user_id: str,
            query: str,
            limit: int,
        ) -> WordPressContentSearchResponse:
            wordpress_service = WordPressService()

            return await wordpress_service.search_content(
                tenant_id=tenant_id,
                user_id=user_id,
                request=WordPressContentSearchRequest(
                    query=query,
                    limit=limit,
                ),
            )

        response = await with_mcp_context(
            search_content,
            principal=principal,
            tool_name="search_wordpress_content",
            required_scope="wordpress:read",
            tenant_id=principal.tenant_id,
            query=validated_query,
            limit=validated_limit,
            user_id=principal.user_id,
        )

        return response.model_dump(
            mode="json",
            exclude_none=True,
        )

    @mcp.tool()
    @mcp_tool_error_boundary
    async def create_wordpress_draft(
        context: Context,
        *,
        title: str,
        content: str,
        excerpt: str | None = None,
        slug: str | None = None,
        tags: list[str] | None = None,
        category_ids: list[str | UUID] | None = None,
    ) -> dict[str, Any]:
        """
        Create a WordPress draft.

        Args:
            context: MCP request context.
            title: Draft title.
            content: Draft body content.
            excerpt: Optional draft excerpt.
            slug: Optional URL slug.
            tags: Optional list of tags.
            category_ids: Optional list of category IDs.

        Returns:
            A tenant-safe draft response containing the WordPress content ID,
            title, status, slug, and bounded metadata. Raw WordPress API
            responses, credentials, storage URLs, and customer PII are never
            returned.

        Raises:
            AuthenticationError: If the caller is not authenticated.
            AuthorizationError: If the caller lacks required access.
            ValidationError: If arguments are invalid.
            RateLimitError: If the caller exceeds the configured limit.
        """
        principal = await get_authenticated_principal(
            context
        )

        validated_title = validate_required_string(
            title,
            field_name="title",
            max_length=MAX_WORDPRESS_TITLE_CHARS,
        )

        validated_content = validate_required_string(
            content,
            field_name="content",
            max_length=MAX_WORDPRESS_CONTENT_CHARS,
        )

        validated_excerpt = None

        if excerpt is not None:
            validated_excerpt = validate_required_string(
                excerpt,
                field_name="excerpt",
                max_length=MAX_WORDPRESS_EXCERPT_CHARS,
            )

        validated_slug = None

        if slug is not None:
            validated_slug = validate_required_string(
                slug,
                field_name="slug",
                max_length=MAX_WORDPRESS_SLUG_CHARS,
            )

        validated_tags = _validate_tags(
            tags
        )

        validated_category_ids = (
            _validate_category_ids(
                category_ids
            )
        )

        request = WordPressCreateDraftRequest(
            title=validated_title,
            content=validated_content,
            excerpt=validated_excerpt,
            slug=validated_slug,
            tags=validated_tags,
            category_ids=validated_category_ids,
        )

        async def create_draft(
            *,
            tenant_id: str,
            user_id: str,
            title: str,
            content: str,
            excerpt: str | None,
            slug: str | None,
            tags: list[str],
            category_ids: list[str],
        ) -> WordPressCreateDraftResponse:
            wordpress_service = WordPressService()

            return await wordpress_service.create_draft(
                tenant_id=tenant_id,
                user_id=user_id,
                request=WordPressCreateDraftRequest(
                    title=title,
                    content=content,
                    excerpt=excerpt,
                    slug=slug,
                    tags=tags,
                    category_ids=category_ids,
                ),
            )

        response = await with_mcp_context(
            create_draft,
            principal=principal,
            tool_name="create_wordpress_draft",
            required_scope="wordpress:write",
            tenant_id=principal.tenant_id,
            title=validated_title,
            content="[REDACTED]",
            excerpt="[REDACTED]",
            slug=validated_slug,
            tags=validated_tags,
            category_ids=validated_category_ids,
            user_id=principal.user_id,
        )

        return response.model_dump(
            mode="json",
            exclude_none=True,
        )

    @mcp.tool()
    @mcp_tool_error_boundary
    async def update_wordpress_draft(
        context: Context,
        *,
        wordpress_content_id: str,
        title: str | None = None,
        content: str | None = None,
        excerpt: str | None = None,
        slug: str | None = None,
        tags: list[str] | None = None,
        category_ids: list[str | UUID] | None = None,
    ) -> dict[str, Any]:
        """
        Update an existing WordPress draft.

        Args:
            context: MCP request context.
            wordpress_content_id: WordPress content identifier.
            title: Optional updated title.
            content: Optional updated body content.
            excerpt: Optional updated excerpt.
            slug: Optional updated URL slug.
            tags: Optional updated tags.
            category_ids: Optional updated category IDs.

        Returns:
            A tenant-safe draft update response containing the WordPress
            content ID, title, status, slug, and bounded metadata. Raw
            WordPress API responses, credentials, storage URLs, and customer
            PII are never returned.

        Raises:
            AuthenticationError: If the caller is not authenticated.
            AuthorizationError: If the caller lacks required access.
            ValidationError: If arguments are invalid.
            NotFoundError: If the draft cannot be found.
            RateLimitError: If the caller exceeds the configured limit.
        """
        principal = await get_authenticated_principal(
            context
        )

        validated_content_id = validate_required_string(
            wordpress_content_id,
            field_name="wordpress_content_id",
            max_length=100,
        )

        validated_title = None

        if title is not None:
            validated_title = validate_required_string(
                title,
                field_name="title",
                max_length=MAX_WORDPRESS_TITLE_CHARS,
            )

        validated_content = None

        if content is not None:
            validated_content = validate_required_string(
                content,
                field_name="content",
                max_length=MAX_WORDPRESS_CONTENT_CHARS,
            )

        validated_excerpt = None

        if excerpt is not None:
            validated_excerpt = validate_required_string(
                excerpt,
                field_name="excerpt",
                max_length=MAX_WORDPRESS_EXCERPT_CHARS,
            )

        validated_slug = None

        if slug is not None:
            validated_slug = validate_required_string(
                slug,
                field_name="slug",
                max_length=MAX_WORDPRESS_SLUG_CHARS,
            )

        validated_tags = _validate_tags(
            tags
        )

        validated_category_ids = (
            _validate_category_ids(
                category_ids
            )
        )

        if (
            validated_title is None
            and validated_content is None
            and validated_excerpt is None
            and validated_slug is None
            and not validated_tags
            and not validated_category_ids
        ):
            raise ValidationError(
                "At least one field must be provided to update a draft."
            )

        request = WordPressUpdateDraftRequest(
            wordpress_content_id=validated_content_id,
            title=validated_title,
            content=validated_content,
            excerpt=validated_excerpt,
            slug=validated_slug,
            tags=validated_tags,
            category_ids=validated_category_ids,
        )

        async def update_draft(
            *,
            tenant_id: str,
            user_id: str,
            wordpress_content_id: str,
            title: str | None,
            content: str | None,
            excerpt: str | None,
            slug: str | None,
            tags: list[str],
            category_ids: list[str],
        ) -> WordPressUpdateDraftResponse:
            wordpress_service = WordPressService()

            return await wordpress_service.update_draft(
                tenant_id=tenant_id,
                user_id=user_id,
                request=WordPressUpdateDraftRequest(
                    wordpress_content_id=wordpress_content_id,
                    title=title,
                    content=content,
                    excerpt=excerpt,
                    slug=slug,
                    tags=tags,
                    category_ids=category_ids,
                ),
            )

        response = await with_mcp_context(
            update_draft,
            principal=principal,
            tool_name="update_wordpress_draft",
            required_scope="wordpress:write",
            tenant_id=principal.tenant_id,
            wordpress_content_id=validated_content_id,
            title=validated_title,
            content="[REDACTED]",
            excerpt="[REDACTED]",
            slug=validated_slug,
            tags=validated_tags,
            category_ids=validated_category_ids,
            user_id=principal.user_id,
        )

        return response.model_dump(
            mode="json",
            exclude_none=True,
        )

    @mcp.tool()
    @mcp_tool_error_boundary
    async def request_wordpress_publication(
        context: Context,
        *,
        wordpress_content_id: str,
    ) -> dict[str, Any]:
        """
        Request publication of a WordPress draft.

        Publication remains subject to the configured human-approval policy.
        This tool never publishes content directly.

        Args:
            context: MCP request context.
            wordpress_content_id: WordPress content identifier.

        Returns:
            A tenant-safe publication-request response containing the content
            ID, requested status, and approval state. Raw WordPress API
            responses, credentials, storage URLs, and customer PII are never
            returned.

        Raises:
            AuthenticationError: If the caller is not authenticated.
            AuthorizationError: If the caller lacks required access.
            ValidationError: If arguments are invalid.
            NotFoundError: If the draft cannot be found.
            WorkflowStateError: If the draft cannot be submitted for review.
            RateLimitError: If the caller exceeds the configured limit.
        """
        principal = await get_authenticated_principal(
            context
        )

        validated_content_id = validate_required_string(
            wordpress_content_id,
            field_name="wordpress_content_id",
            max_length=100,
        )

        request = WordPressPostRequest(
            wordpress_content_id=validated_content_id,
        )

        async def request_publication(
            *,
            tenant_id: str,
            user_id: str,
            wordpress_content_id: str,
        ) -> WordPressPostResponse:
            wordpress_service = WordPressService()

            return await wordpress_service.request_publication(
                tenant_id=tenant_id,
                user_id=user_id,
                request=WordPressPostRequest(
                    wordpress_content_id=wordpress_content_id,
                ),
            )

        response = await with_mcp_context(
            request_publication,
            principal=principal,
            tool_name="request_wordpress_publication",
            required_scope="wordpress:publish",
            tenant_id=principal.tenant_id,
            wordpress_content_id=validated_content_id,
            user_id=principal.user_id,
        )

        return response.model_dump(
            mode="json",
            exclude_none=True,
        )

    logger.info(
        "wordpress_tools_registered",
        tools=[
            "search_wordpress_content",
            "create_wordpress_draft",
            "update_wordpress_draft",
            "request_wordpress_publication",
        ],
    )