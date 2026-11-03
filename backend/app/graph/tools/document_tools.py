"""
Document retrieval tools for workflow nodes.

These helpers provide a narrow, node-safe interface to document persistence.
They enforce tenant scoping, validate identifiers, bound batch operations, and
return only the document fields that workflow nodes need.

Security and policy rules:
- Every lookup is tenant-scoped when a tenant identifier is supplied.
- Never expose raw file bytes, storage URLs, signed URLs, or internal storage
  metadata to workflow nodes.
- Never log document content, extracted text, customer PII, or storage paths.
- Treat document IDs as untrusted input and validate them before persistence.
- Keep this module free of workflow policy, retrieval ranking, embedding, and
  LLM logic.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import MAX_DOCUMENT_BATCH_SIZE
from app.core.exceptions import NotFoundError, ValidationError
from app.core.logging_context import bind_workflow_context
from app.core.utils import truncate_text
from app.crud.document_crud import get_document_or_404
from app.models.documents import DocumentModel


logger = logging.getLogger("agentic_bridge.graph.tools.document")

MAX_DOCUMENT_TITLE_CHARS = 500
MAX_DOCUMENT_SUMMARY_CHARS = 2_000
MAX_DOCUMENT_TAG_COUNT = 50
MAX_DOCUMENT_TAG_LENGTH = 100
MAX_DOCUMENT_METADATA_KEYS = 50
MAX_DOCUMENT_METADATA_VALUE_CHARS = 1_000


# ---------------------------------------------------------------------------
# Identifier validation
# ---------------------------------------------------------------------------


def _coerce_document_id(
    document_id: str | UUID,
) -> UUID:
    """
    Validate and normalize a document identifier.

    Args:
        document_id: Document identifier supplied by a workflow node.

    Returns:
        A validated UUID.

    Raises:
        ValidationError: If the identifier is missing or invalid.
    """
    if document_id is None:
        raise ValidationError("Document ID is required.")

    if isinstance(document_id, UUID):
        return document_id

    if not isinstance(document_id, str):
        raise ValidationError("Document ID must be a UUID or UUID string.")

    normalized_document_id = document_id.strip()

    if not normalized_document_id:
        raise ValidationError("Document ID is required.")

    try:
        return UUID(normalized_document_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValidationError(
            "Document ID must be a valid UUID."
        ) from exc


def _coerce_tenant_id(
    tenant_id: str | UUID | None,
) -> str | None:
    """
    Validate and normalize a tenant identifier.

    Args:
        tenant_id: Tenant identifier supplied by a workflow node.

    Returns:
        A normalized tenant identifier or `None`.

    Raises:
        ValidationError: If the tenant identifier is invalid.
    """
    if tenant_id is None:
        return None

    if isinstance(tenant_id, UUID):
        return str(tenant_id)

    if not isinstance(tenant_id, str):
        raise ValidationError(
            "Tenant ID must be a UUID, UUID string, or None."
        )

    normalized_tenant_id = tenant_id.strip()

    if not normalized_tenant_id:
        return None

    try:
        return str(UUID(normalized_tenant_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValidationError(
            "Tenant ID must be a valid UUID."
        ) from exc


def _coerce_document_ids(
    document_ids: Sequence[str | UUID],
) -> list[UUID]:
    """
    Validate, normalize, and deduplicate document identifiers.

    Args:
        document_ids: Document identifiers supplied by a workflow node.

    Returns:
        A bounded list of unique document UUIDs in request order.

    Raises:
        ValidationError: If no IDs are supplied, the batch is too large, or an
            identifier is invalid.
    """
    if not document_ids:
        raise ValidationError("At least one document ID is required.")

    if len(document_ids) > MAX_DOCUMENT_BATCH_SIZE:
        raise ValidationError(
            f"Cannot retrieve more than {MAX_DOCUMENT_BATCH_SIZE} "
            "documents in one workflow operation."
        )

    normalized_ids: list[UUID] = []
    seen_ids: set[UUID] = set()

    for document_id in document_ids:
        normalized_id = _coerce_document_id(document_id)

        if normalized_id in seen_ids:
            continue

        seen_ids.add(normalized_id)
        normalized_ids.append(normalized_id)

    if not normalized_ids:
        raise ValidationError(
            "At least one valid document ID is required."
        )

    return normalized_ids


# ---------------------------------------------------------------------------
# Safe document projections
# ---------------------------------------------------------------------------


def _safe_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Return a bounded, workflow-safe copy of document metadata.

    Removes common sensitive keys and bounds values. This is defense in depth;
    document persistence remains responsible for preventing sensitive metadata
    from being stored in the first place.
    """
    if not metadata:
        return {}

    sensitive_key_fragments = (
        "password",
        "secret",
        "token",
        "api_key",
        "apikey",
        "credential",
        "authorization",
        "cookie",
        "signed_url",
        "storage_url",
        "file_url",
        "download_url",
    )

    safe_metadata: dict[str, Any] = {}

    for key, value in list(metadata.items())[:MAX_DOCUMENT_METADATA_KEYS]:
        if not isinstance(key, str):
            continue

        normalized_key = key.strip().lower()

        if not normalized_key:
            continue

        if any(
            fragment in normalized_key
            for fragment in sensitive_key_fragments
        ):
            continue

        if isinstance(value, (str, int, float, bool)) or value is None:
            safe_metadata[key] = (
                truncate_text(
                    str(value),
                    MAX_DOCUMENT_METADATA_VALUE_CHARS,
                )
                if isinstance(value, str)
                else value
            )

    return safe_metadata


def _safe_tags(
    tags: Sequence[str] | None,
) -> list[str]:
    """Return bounded, normalized document tags."""
    if not tags:
        return []

    normalized_tags: list[str] = []
    seen_tags: set[str] = set()

    for tag in list(tags)[:MAX_DOCUMENT_TAG_COUNT]:
        if not isinstance(tag, str):
            continue

        normalized_tag = tag.strip()[:MAX_DOCUMENT_TAG_LENGTH]

        if not normalized_tag:
            continue

        tag_key = normalized_tag.lower()

        if tag_key not in seen_tags:
            seen_tags.add(tag_key)
            normalized_tags.append(normalized_tag)

    return normalized_tags


def get_document_summary(
    document: DocumentModel,
) -> dict[str, Any]:
    """
    Build a workflow-safe summary of a document.

    This projection intentionally excludes raw file bytes, extracted document
    text, storage paths, signed URLs, and internal storage metadata.

    Args:
        document: A tenant-scoped document model instance.

    Returns:
        A dictionary containing safe document metadata for workflow use.
    """
    return {
        "document_id": str(document.id),
        "title": truncate_text(
            str(document.title or ""),
            MAX_DOCUMENT_TITLE_CHARS,
        ),
        "summary": truncate_text(
            str(getattr(document, "summary", "") or ""),
            MAX_DOCUMENT_SUMMARY_CHARS,
        ),
        "status": getattr(document, "status", None),
        "document_type": getattr(document, "document_type", None),
        "mime_type": getattr(document, "mime_type", None),
        "file_size_bytes": getattr(document, "file_size_bytes", None),
        "page_count": getattr(document, "page_count", None),
        "tags": _safe_tags(getattr(document, "tags", None)),
        "metadata": _safe_metadata(
            getattr(document, "metadata", None)
        ),
        "created_at": getattr(document, "created_at", None),
        "updated_at": getattr(document, "updated_at", None),
    }


def get_document_summaries(
    documents: Sequence[DocumentModel],
) -> list[dict[str, Any]]:
    """
    Build workflow-safe summaries for multiple documents.

    Args:
        documents: Tenant-scoped document model instances.

    Returns:
        A list of safe document summaries.
    """
    return [
        get_document_summary(document)
        for document in documents
    ]


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------


async def get_document_for_workflow(
    db: AsyncSession,
    document_id: str | UUID,
    *,
    tenant_id: str | UUID | None = None,
    execution_id: str | None = None,
) -> DocumentModel:
    """
    Retrieve a tenant-scoped document for a workflow node.

    Args:
        db: Active database session.
        document_id: Document identifier.
        tenant_id: Tenant identifier used for row-level isolation.
        execution_id: Optional workflow execution ID for traceability.

    Returns:
        The requested document model.

    Raises:
        ValidationError: If identifiers are invalid.
        NotFoundError: If the document does not exist or is not accessible.
    """
    normalized_document_id = _coerce_document_id(document_id)
    normalized_tenant_id = _coerce_tenant_id(tenant_id)

    if execution_id:
        bind_workflow_context(
            execution_id,
            tenant_id=normalized_tenant_id,
        )

    document = await get_document_or_404(
        db,
        normalized_document_id,
        tenant_id=normalized_tenant_id,
    )

    logger.debug(
        "workflow_document_retrieved",
        document_id=str(document.id),
        tenant_id=normalized_tenant_id,
        execution_id=execution_id,
    )

    return document


async def get_documents_for_workflow(
    db: AsyncSession,
    document_ids: Sequence[str | UUID],
    *,
    tenant_id: str | UUID | None = None,
    execution_id: str | None = None,
) -> list[DocumentModel]:
    """
    Retrieve multiple tenant-scoped documents.

    Duplicate identifiers are removed while preserving request order. The
    function fails if any requested document cannot be found, preventing
    partial document sets from silently entering a workflow.

    Args:
        db: Active database session.
        document_ids: Document identifiers.
        tenant_id: Tenant identifier used for row-level isolation.
        execution_id: Optional workflow execution ID for traceability.

    Returns:
        A list of requested document models.

    Raises:
        ValidationError: If identifiers are invalid or the batch is too large.
        NotFoundError: If any requested document cannot be found.
    """
    normalized_document_ids = _coerce_document_ids(document_ids)
    normalized_tenant_id = _coerce_tenant_id(tenant_id)

    if execution_id:
        bind_workflow_context(
            execution_id,
            tenant_id=normalized_tenant_id,
        )

    documents: list[DocumentModel] = []

    for document_id in normalized_document_ids:
        documents.append(
            await get_document_for_workflow(
                db,
                document_id,
                tenant_id=normalized_tenant_id,
                execution_id=execution_id,
            )
        )

    logger.debug(
        "workflow_documents_retrieved",
        document_count=len(documents),
        tenant_id=normalized_tenant_id,
        execution_id=execution_id,
    )

    return documents


async def get_document_summary_for_workflow(
    db: AsyncSession,
    document_id: str | UUID,
    *,
    tenant_id: str | UUID | None = None,
    execution_id: str | None = None,
) -> dict[str, Any]:
    """
    Retrieve a workflow-safe summary for one document.

    Args:
        db: Active database session.
        document_id: Document identifier.
        tenant_id: Tenant identifier used for row-level isolation.
        execution_id: Optional workflow execution ID for traceability.

    Returns:
        A workflow-safe document summary.

    Raises:
        ValidationError: If identifiers are invalid.
        NotFoundError: If the document does not exist or is not accessible.
    """
    document = await get_document_for_workflow(
        db,
        document_id,
        tenant_id=tenant_id,
        execution_id=execution_id,
    )

    return get_document_summary(document)


async def get_document_summaries_for_workflow(
    db: AsyncSession,
    document_ids: Sequence[str | UUID],
    *,
    tenant_id: str | UUID | None = None,
    execution_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve workflow-safe summaries for multiple documents.

    Args:
        db: Active database session.
        document_ids: Document identifiers.
        tenant_id: Tenant identifier used for row-level isolation.
        execution_id: Optional workflow execution ID for traceability.

    Returns:
        A list of workflow-safe document summaries.

    Raises:
        ValidationError: If identifiers are invalid or the batch is too large.
        NotFoundError: If any requested document cannot be found.
    """
    documents = await get_documents_for_workflow(
        db,
        document_ids,
        tenant_id=tenant_id,
        execution_id=execution_id,
    )

    return get_document_summaries(documents)


def generate_document_tool_call_id() -> str:
    """
    Generate a stable, non-sensitive tool-call correlation ID.

    Workflow nodes can use this ID in logs and provider telemetry without
    exposing document content or customer identifiers.
    """
    return f"doc_tool_{uuid4()}"