"""
MCP document tools.

These tools expose safe, tenant-scoped document metadata and document-backed
workflow context. They never return file contents, raw extracted text, storage
paths, signed URLs, embeddings, credentials, or customer PII.

Security and architecture rules:
- Authenticate and authorize every document operation.
- Enforce tenant isolation through the authenticated principal.
- Treat document IDs, filenames, tags, and filters as untrusted input.
- Validate and bound every argument before it reaches an application service.
- Rate-limit document operations to protect ingestion and retrieval capacity.
- Return only sanitized document summaries, citations, and processing status.
- Never return or log file bytes, extracted text, embeddings, storage URLs,
  signed URLs, database URLs, credentials, or customer PII.
- Keep this module thin: document business rules belong in document services.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from mcp.server.fastmcp import Context, FastMCP

from app.core.enums import DocumentStatus, DocumentType
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
    DocumentListRequest,
    DocumentSummaryRequest,
    DocumentUploadRequest,
)


logger = logging.getLogger(
    "agentic_bridge.mcp_server.tools.documents"
)

REQUIRED_DOCUMENT_READ_SCOPE = "documents:read"
REQUIRED_DOCUMENT_WRITE_SCOPE = "documents:write"

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


def register_document_tools(
    mcp: FastMCP,
) -> None:
    """Register document MCP tools."""

    @mcp.tool()
    @mcp_tool_error_boundary
    async def get_document_summary(
        document_id: str,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """
        Get workflow-safe metadata and processing status for a document.

        Args:
            document_id: Document identifier.
            context: MCP request context.

        Returns:
            A bounded document summary. File contents, extracted text,
            storage URLs, signed URLs, and embeddings are never returned.

        Raises:
            AuthenticationError: If the caller is not authenticated.
            AuthorizationError: If the caller lacks document-read permission.
            ValidationError: If the document ID is invalid.
            NotFoundError: If the document does not exist or is inaccessible.
        """
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope=REQUIRED_DOCUMENT_READ_SCOPE,
        )

        request = DocumentSummaryRequest(
            document_id=document_id,
        )

        workflow_service = get_workflow_service()

        async def load_document() -> Any:
            return await workflow_service.get_document_summary(
                document_id=request.document_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )

        result = await with_mcp_context(
            load_document,
            principal=principal,
            tool_name="get_document_summary",
        )

        logger.info(
            "mcp_document_summary_retrieved",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            document_id=str(request.document_id),
            document_status=result.status,
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )

    @mcp.tool()
    @mcp_tool_error_boundary
    async def list_documents(
        document_status: str | None = None,
        document_type: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """
        List tenant-visible document summaries.

        Args:
            document_status: Optional document-status filter.
            document_type: Optional document-type filter.
            page: Optional one-based page number.
            page_size: Optional page size.
            context: MCP request context.

        Returns:
            A bounded, paginated list of workflow-safe document summaries.

        Raises:
            AuthenticationError: If the caller is not authenticated.
            AuthorizationError: If the caller lacks document-read permission.
            ValidationError: If filters or pagination arguments are invalid.
        """
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope=REQUIRED_DOCUMENT_READ_SCOPE,
        )

        request = DocumentListRequest(
            document_status=document_status,
            document_type=document_type,
            page=page or 1,
            page_size=page_size or DEFAULT_PAGE_SIZE,
        )

        workflow_service = get_workflow_service()

        async def load_documents() -> Any:
            return await workflow_service.list_documents(
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                document_status=request.document_status,
                document_type=request.document_type,
                page=request.page,
                page_size=request.page_size,
            )

        result = await with_mcp_context(
            load_documents,
            principal=principal,
            tool_name="list_documents",
        )

        logger.info(
            "mcp_documents_listed",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            page=request.page,
            page_size=request.page_size,
            document_status=request.document_status,
            document_type=request.document_type,
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )

    @mcp.tool()
    @mcp_tool_error_boundary
    async def request_document_ingestion(
        document_id: str,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """
        Request ingestion and indexing for an uploaded document.

        This tool requests ingestion; it does not process file contents
        directly. Ingestion remains subject to document policy and background
        worker capacity.

        Args:
            document_id: Document identifier.
            context: MCP request context.

        Returns:
            An ingestion-request acknowledgement.

        Raises:
            AuthenticationError: If the caller is not authenticated.
            AuthorizationError: If the caller lacks document-write permission.
            ValidationError: If the document ID is invalid.
        """
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope=REQUIRED_DOCUMENT_WRITE_SCOPE,
        )

        request = DocumentSummaryRequest(
            document_id=document_id,
        )

        workflow_service = get_workflow_service()

        async def request_ingestion() -> Any:
            return await workflow_service.request_document_ingestion(
                document_id=request.document_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )

        result = await with_mcp_context(
            request_ingestion,
            principal=principal,
            tool_name="request_document_ingestion",
        )

        logger.info(
            "mcp_document_ingestion_requested",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            document_id=str(request.document_id),
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )

    @mcp.tool()
    @mcp_tool_error_boundary
    async def get_document_citations(
        document_id: str,
        limit: int | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """
        Get workflow-safe citations associated with a document.

        Args:
            document_id: Document identifier.
            limit: Optional maximum number of citations.
            context: MCP request context.

        Returns:
            Bounded document citations. Embeddings, chunk text, storage URLs,
            and signed URLs are never returned.

        Raises:
            AuthenticationError: If the caller is not authenticated.
            AuthorizationError: If the caller lacks document-read permission.
            ValidationError: If arguments are invalid.
        """
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope=REQUIRED_DOCUMENT_READ_SCOPE,
        )

        request = DocumentCitationsRequest(
            document_id=document_id,
            limit=limit or DEFAULT_PAGE_SIZE,
        )

        workflow_service = get_workflow_service()

        async def load_citations() -> Any:
            return await workflow_service.get_document_citations(
                document_id=request.document_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                limit=request.limit,
            )

        result = await with_mcp_context(
            load_citations,
            principal=principal,
            tool_name="get_document_citations",
        )

        logger.info(
            "mcp_document_citations_retrieved",
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            document_id=str(request.document_id),
            citation_count=len(result.citations),
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )


def _validate_document_status(
    document_status: str | None,
) -> DocumentStatus | None:
    """
    Validate an optional document-status filter.

    Args:
        document_status: Caller-supplied status filter.

    Returns:
        A supported document status or `None`.

    Raises:
        AuthorizationError: If the status filter is invalid.
    """
    if document_status is None:
        return None

    if not isinstance(document_status, str):
        raise AuthorizationError(
            "Document status filter must be a string."
        )

    normalized_status = document_status.strip().lower()

    if not normalized_status:
        return None

    try:
        return DocumentStatus(normalized_status)
    except ValueError as exc:
        raise AuthorizationError(
            "Document status filter is not recognized."
        ) from exc


def _validate_document_type(
    document_type: str | None,
) -> DocumentType | None:
    """
    Validate an optional document-type filter.

    Args:
        document_type: Caller-supplied type filter.

    Returns:
        A supported document type or `None`.

    Raises:
        AuthorizationError: If the type filter is invalid.
    """
    if document_type is None:
        return None

    if not isinstance(document_type, str):
        raise AuthorizationError(
            "Document type filter must be a string."
        )

    normalized_type = document_type.strip().lower()

    if not normalized_type:
        return None

    try:
        return DocumentType(normalized_type)
    except ValueError as exc:
        raise AuthorizationError(
            "Document type filter is not recognized."
        ) from exc


# Configure conservative document-tool rate limits.
rate_limiter.configure_tool(
    tool_name="list_documents",
    requests_per_minute=60,
    burst=20,
)

rate_limiter.configure_tool(
    tool_name="get_document_summary",
    requests_per_minute=60,
    burst=20,
)

rate_limiter.configure_tool(
    tool_name="request_document_ingestion",
    requests_per_minute=10,
    burst=5,
)

rate_limiter.configure_tool(
    tool_name="get_document_citations",
    requests_per_minute=30,
    burst=10,
)