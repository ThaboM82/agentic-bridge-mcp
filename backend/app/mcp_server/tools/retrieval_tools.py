"""
MCP retrieval tools.

These tools expose bounded, tenant-scoped semantic retrieval. They return
workflow-safe matches and citations, never embeddings, raw document text,
storage URLs, signed URLs, credentials, or customer PII.

Security and architecture rules:
- Authenticate and authorize every retrieval request.
- Enforce tenant isolation through the authenticated principal.
- Treat query text, document IDs, and retrieval parameters as untrusted input.
- Validate and bound all arguments before calling application services.
- Rate-limit retrieval to protect embedding and vector-database capacity.
- Return only sanitized retrieval matches and citations.
- Never return or log embeddings, vectors, document contents, prompts,
  completions, storage paths, or credentials.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from app.core.enums import RetrievalPurpose
from app.core.exceptions import AuthorizationError
from app.mcp_server.auth import (
    get_authenticated_principal,
    require_scope,
)
from app.mcp_server.dependencies import get_workflow_service
from app.mcp_server.errors import mcp_tool_error_boundary
from app.mcp_server.middleware import with_mcp_context
from app.mcp_server.rate_limit import rate_limiter
from app.mcp_server.schemas import (
    RetrievalRequest,
    RetrievalResponse,
)


logger = logging.getLogger(
    "agentic_bridge.mcp_server.tools.retrieval"
)

REQUIRED_RETRIEVAL_SCOPE = "documents:read"

DEFAULT_RETRIEVAL_LIMIT = 20
MAX_RETRIEVAL_LIMIT = 100


def register_retrieval_tools(
    mcp: FastMCP,
) -> None:
    """Register retrieval MCP tools."""

    @mcp.tool()
    @mcp_tool_error_boundary
    async def retrieve_workflow_context(
        query: str,
        document_ids: list[str] | None = None,
        limit: int | None = None,
        retrieval_purpose: str | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve workflow-safe design or domain context.

        Args:
            query: Natural-language retrieval query.
            document_ids: Optional document IDs used to restrict retrieval.
            limit: Optional maximum number of matches.
            retrieval_purpose: Optional retrieval purpose label.
            context: MCP request context.

        Returns:
            Bounded retrieval matches, citations, and retrieval metadata.

        Raises:
            AuthenticationError: If the caller is not authenticated.
            AuthorizationError: If the caller lacks retrieval permission.
            ValidationError: If arguments are invalid.
            RateLimitError: If the retrieval rate limit is exceeded.
        """
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope=REQUIRED_RETRIEVAL_SCOPE,
        )

        request = RetrievalRequest(
            query=query,
            document_ids=document_ids or [],
            limit=limit or DEFAULT_RETRIEVAL_LIMIT,
        )

        normalized_purpose = _validate_retrieval_purpose(
            retrieval_purpose
        )

        workflow_service = get_workflow_service()

        async def retrieve() -> RetrievalResponse:
            return (
                await workflow_service.retrieve_workflow_context(
                    query=request.query,
                    tenant_id=principal.tenant_id,
                    user_id=principal.user_id,
                    document_ids=request.document_ids,
                    limit=request.limit,
                    retrieval_purpose=normalized_purpose,
                )
            )

        result = await with_mcp_context(
            retrieve,
            principal=principal,
            tool_name="retrieve_workflow_context",
        )

        logger.info(
            "mcp_retrieval_completed",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            match_count=len(result.matches),
            citation_count=len(result.citations),
            retrieval_purpose=normalized_purpose,
            document_filter_count=len(request.document_ids),
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )


def _validate_retrieval_purpose(
    retrieval_purpose: str | None,
) -> RetrievalPurpose:
    """
    Validate an optional retrieval-purpose label.

    Args:
        retrieval_purpose: Caller-supplied retrieval purpose.

    Returns:
        A supported retrieval purpose.

    Raises:
        ValidationError: If the purpose is not recognized.
    """
    if retrieval_purpose is None:
        return RetrievalPurpose.DESIGN_CONTEXT

    if not isinstance(retrieval_purpose, str):
        raise AuthorizationError(
            "Retrieval purpose must be a string."
        )

    normalized_purpose = retrieval_purpose.strip().lower()

    if not normalized_purpose:
        return RetrievalPurpose.DESIGN_CONTEXT

    try:
        return RetrievalPurpose(normalized_purpose)
    except ValueError as exc:
        raise AuthorizationError(
            "Retrieval purpose is not recognized."
        ) from exc


# Configure a conservative retrieval limit to protect vector-database and
# embedding capacity.
rate_limiter.configure_tool(
    tool_name="retrieve_workflow_context",
    requests_per_minute=30,
    burst=10,
)