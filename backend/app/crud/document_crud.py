"""
CRUD operations for documents.

This module owns all direct persistence operations for documents. API routes
and services must not construct SQLAlchemy queries directly; they should call
these functions so filtering, pagination, tenant isolation, transaction
boundaries, and error mapping remain consistent.

Responsibilities:
- Create, retrieve, update, archive, and delete documents.
- Apply tenant isolation where a tenant ID is present.
- Provide paginated listing and metadata search.
- Support bulk ingestion-ready document creation.
- Map database failures to stable application exceptions.

Non-goals:
- Do not generate embeddings, invoke LLMs, parse files, or execute workflows
  in this module. Those belong in service and worker layers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import String, and_, func, or_, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import DocumentStatus, DocumentType
from app.core.exceptions import (
    DatabaseIntegrityError,
    NotFoundError,
    PersistenceError,
    ValidationError,
)
from app.core.logging_context import bind_document_context
from app.core.pagination import Page, PaginationParams
from app.crud.base import (
    create_record,
    create_records_bulk,
    delete_record,
    get_by_id,
    get_by_id_or_404,
    update_record,
)
from app.models.documents import DocumentModel


logger = logging.getLogger("agentic_bridge.crud.document")

MAX_SEARCH_TERM_LENGTH = 200
MAX_TAG_COUNT = 25
MAX_TAG_LENGTH = 50


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _normalize_document_id(document_id: str | UUID) -> UUID:
    """Validate and normalize a document ID."""
    if isinstance(document_id, UUID):
        return document_id

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValidationError("document_id must be a non-empty string.")

    try:
        return UUID(document_id.strip())
    except ValueError as exc:
        raise ValidationError(
            "document_id must be a valid UUID."
        ) from exc


def _normalize_document_type(
    document_type: str | DocumentType | None,
) -> DocumentType | None:
    """Validate and normalize a document type."""
    if document_type is None:
        return None

    try:
        return DocumentType(document_type)
    except ValueError as exc:
        raise ValidationError(
            f"Unsupported document type: {document_type}."
        ) from exc


def _normalize_document_status(
    status: str | DocumentStatus | None,
) -> DocumentStatus | None:
    """Validate and normalize a document status."""
    if status is None:
        return None

    try:
        return DocumentStatus(status)
    except ValueError as exc:
        raise ValidationError(
            f"Unsupported document status: {status}."
        ) from exc


def _normalize_tags(tags: Sequence[str] | None) -> list[str]:
    """Normalize, validate, and deduplicate document tags."""
    if not tags:
        return []

    normalized: list[str] = []
    seen: set[str] = set()

    for tag in tags:
        if not isinstance(tag, str):
            raise ValidationError("Document tags must be strings.")

        clean_tag = tag.strip().lower()

        if not clean_tag:
            continue

        if len(clean_tag) > MAX_TAG_LENGTH:
            raise ValidationError(
                f"Document tags may not exceed {MAX_TAG_LENGTH} characters."
            )

        if clean_tag not in seen:
            seen.add(clean_tag)
            normalized.append(clean_tag)

    if len(normalized) > MAX_TAG_COUNT:
        raise ValidationError(
            f"A document may not have more than {MAX_TAG_COUNT} tags."
        )

    return normalized


def _normalize_search_term(search: str | None) -> str | None:
    """Validate and normalize a metadata search term."""
    if search is None:
        return None

    normalized = search.strip()

    if not normalized:
        return None

    if len(normalized) > MAX_SEARCH_TERM_LENGTH:
        raise ValidationError(
            f"Search term may not exceed {MAX_SEARCH_TERM_LENGTH} characters."
        )

    return normalized


def _apply_tenant_filter(
    query,
    tenant_id: str | None,
    *,
    require_tenant: bool = False,
):
    """
    Apply tenant isolation to a document query.

    Args:
        query: SQLAlchemy select query.
        tenant_id: Tenant identifier.
        require_tenant: Whether a missing tenant ID should be rejected.

    Returns:
        The filtered query.
    """
    if tenant_id:
        return query.where(DocumentModel.tenant_id == tenant_id)

    if require_tenant:
        raise ValidationError("tenant_id is required for this operation.")

    return query


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


async def create_document(
    db: AsyncSession,
    *,
    title: str,
    content: str,
    document_type: str | DocumentType = DocumentType.OTHER,
    status: str | DocumentStatus = DocumentStatus.UPLOADED,
    tenant_id: str | None = None,
    created_by: str | None = None,
    tags: Sequence[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> DocumentModel:
    """
    Create and persist a new document.

    Args:
        db: Active async database session.
        title: Human-readable document title.
        content: Raw document text or serialized content.
        document_type: Document category.
        status: Initial document lifecycle status.
        tenant_id: Owning tenant identifier, when multi-tenancy is enabled.
        created_by: Authenticated user ID, when available.
        tags: Optional normalized tags.
        extra_metadata: Optional non-sensitive document metadata.

    Returns:
        The persisted document.

    Raises:
        ValidationError: If required values or tags are invalid.
        DatabaseIntegrityError: If a uniqueness or foreign-key constraint fails.
        PersistenceError: If another database failure occurs.
    """
    if not title.strip():
        raise ValidationError("Document title is required.")

    if not content.strip():
        raise ValidationError("Document content is required.")

    safe_document_type = _normalize_document_type(document_type)

    if safe_document_type is None:
        raise ValidationError("Document type is required.")

    safe_status = _normalize_document_status(status)

    if safe_status is None:
        raise ValidationError("Document status is required.")

    safe_tags = _normalize_tags(tags)

    document = DocumentModel(
        title=title.strip(),
        content=content,
        document_type=safe_document_type.value,
        status=safe_status.value,
        tenant_id=tenant_id,
        created_by=created_by,
        tags=safe_tags,
        extra_metadata=extra_metadata or {},
    )

    document = await create_record(db, document)

    bind_document_context(str(document.id))

    logger.info(
        "document_created",
        document_id=str(document.id),
        document_type=document.document_type,
        status=document.status,
        tenant_id=tenant_id,
        content_length=len(content),
    )

    return document


async def create_documents_bulk(
    db: AsyncSession,
    documents: Sequence[dict[str, Any]],
) -> Sequence[DocumentModel]:
    """
    Create multiple documents in one transaction.

    Each dictionary must contain `title` and `content`. Optional keys include
    `document_type`, `status`, `tenant_id`, `created_by`, `tags`, and
    `extra_metadata`.

    Args:
        db: Active async database session.
        documents: Document creation payloads.

    Returns:
        The persisted documents.

    Raises:
        ValidationError: If no documents are supplied or a payload is invalid.
        DatabaseIntegrityError: If a constraint is violated.
        PersistenceError: If another database failure occurs.
    """
    if not documents:
        raise ValidationError("At least one document must be supplied.")

    models: list[DocumentModel] = []

    for index, payload in enumerate(documents):
        title = payload.get("title")
        content = payload.get("content")

        if not title or not content:
            raise ValidationError(
                "Each document must include a title and content.",
                details={"document_index": index},
            )

        safe_document_type = _normalize_document_type(
            payload.get("document_type", DocumentType.OTHER)
        )

        safe_status = _normalize_document_status(
            payload.get("status", DocumentStatus.UPLOADED)
        )

        if safe_document_type is None or safe_status is None:
            raise ValidationError(
                "Document type and status are required.",
                details={"document_index": index},
            )

        models.append(
            DocumentModel(
                title=str(title).strip(),
                content=str(content),
                document_type=safe_document_type.value,
                status=safe_status.value,
                tenant_id=payload.get("tenant_id"),
                created_by=payload.get("created_by"),
                tags=_normalize_tags(payload.get("tags")),
                extra_metadata=payload.get("extra_metadata") or {},
            )
        )

    created_documents = await create_records_bulk(db, models)

    logger.info(
        "documents_created_bulk",
        document_count=len(created_documents),
        tenant_id=created_documents[0].tenant_id
        if created_documents
        else None,
    )

    return created_documents


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


async def get_document(
    db: AsyncSession,
    document_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> DocumentModel | None:
    """
    Retrieve a document by ID.

    Args:
        db: Active async database session.
        document_id: Document UUID.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The document, or `None` when it does not exist or belongs to another
        tenant.
    """
    document_uuid = _normalize_document_id(document_id)

    query = select(DocumentModel).where(
        DocumentModel.id == document_uuid
    )

    query = _apply_tenant_filter(query, tenant_id)

    result = await db.execute(query)

    return result.scalar_one_or_none()


async def get_document_or_404(
    db: AsyncSession,
    document_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> DocumentModel:
    """
    Retrieve a document by ID or raise `NotFoundError`.

    Args:
        db: Active async database session.
        document_id: Document UUID.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The matching document.

    Raises:
        NotFoundError: If the document does not exist or is outside the tenant.
    """
    document = await get_document(
        db,
        document_id,
        tenant_id=tenant_id,
    )

    if document is None:
        raise NotFoundError(
            "Document was not found.",
            details={"document_id": str(document_id)},
        )

    return document


async def list_documents(
    db: AsyncSession,
    *,
    pagination: PaginationParams,
    document_type: str | DocumentType | None = None,
    status: str | DocumentStatus | None = None,
    tenant_id: str | None = None,
    created_by: str | None = None,
    tags: Sequence[str] | None = None,
    search: str | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
) -> Page[DocumentModel]:
    """
    Retrieve a paginated, filtered list of documents.

    Args:
        db: Active async database session.
        pagination: Validated pagination parameters.
        document_type: Optional document-type filter.
        status: Optional document-status filter.
        tenant_id: Optional tenant isolation filter.
        created_by: Optional creator filter.
        tags: Optional tags; a document must contain all supplied tags.
        search: Optional case-insensitive title/content search.
        created_after: Optional inclusive lower created-at bound.
        created_before: Optional inclusive upper created-at bound.

    Returns:
        A `Page` containing documents and pagination metadata.

    Raises:
        ValidationError: If filters or pagination values are invalid.
    """
    safe_document_type = _normalize_document_type(document_type)
    safe_status = _normalize_document_status(status)
    safe_search = _normalize_search_term(search)
    safe_tags = _normalize_tags(tags)

    filters = []

    if safe_document_type is not None:
        filters.append(
            DocumentModel.document_type == safe_document_type.value
        )

    if safe_status is not None:
        filters.append(DocumentModel.status == safe_status.value)

    if tenant_id:
        filters.append(DocumentModel.tenant_id == tenant_id)

    if created_by:
        filters.append(DocumentModel.created_by == created_by)

    if created_after is not None:
        filters.append(DocumentModel.created_at >= created_after)

    if created_before is not None:
        filters.append(DocumentModel.created_at <= created_before)

    if safe_tags:
        for tag in safe_tags:
            filters.append(
                DocumentModel.tags.contains([tag])
            )

    if safe_search:
        search_pattern = f"%{safe_search}%"

        filters.append(
            or_(
                DocumentModel.title.ilike(search_pattern),
                DocumentModel.content.ilike(search_pattern),
            )
        )

    count_query = select(func.count()).select_from(DocumentModel)

    if filters:
        count_query = count_query.where(and_(*filters))

    count_result = await db.execute(count_query)
    total = int(count_result.scalar_one())

    query = select(DocumentModel).order_by(
        DocumentModel.created_at.desc(),
        DocumentModel.id.desc(),
    )

    if filters:
        query = query.where(and_(*filters))

    result = await db.execute(
        query.offset(pagination.skip).limit(pagination.limit)
    )

    documents = list(result.scalars().all())

    return Page[DocumentModel].create(
        items=documents,
        total=total,
        skip=pagination.skip,
        limit=pagination.limit,
    )


async def count_documents(
    db: AsyncSession,
    *,
    document_type: str | DocumentType | None = None,
    status: str | DocumentStatus | None = None,
    tenant_id: str | None = None,
) -> int:
    """
    Count documents matching optional filters.

    Args:
        db: Active async database session.
        document_type: Optional document-type filter.
        status: Optional document-status filter.
        tenant_id: Optional tenant isolation filter.

    Returns:
        Total matching document count.
    """
    safe_document_type = _normalize_document_type(document_type)
    safe_status = _normalize_document_status(status)

    filters = []

    if safe_document_type is not None:
        filters.append(
            DocumentModel.document_type == safe_document_type.value
        )

    if safe_status is not None:
        filters.append(DocumentModel.status == safe_status.value)

    if tenant_id:
        filters.append(DocumentModel.tenant_id == tenant_id)

    query = select(func.count()).select_from(DocumentModel)

    if filters:
        query = query.where(and_(*filters))

    result = await db.execute(query)

    return int(result.scalar_one())


async def document_exists(
    db: AsyncSession,
    document_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> bool:
    """
    Check whether a document exists.

    Args:
        db: Active async database session.
        document_id: Document UUID.
        tenant_id: Optional tenant isolation filter.

    Returns:
        `True` when the document exists and is visible to the tenant.
    """
    document = await get_document(
        db,
        document_id,
        tenant_id=tenant_id,
    )

    return document is not None


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


async def update_document(
    db: AsyncSession,
    document: DocumentModel,
    *,
    title: str | None = None,
    content: str | None = None,
    document_type: str | DocumentType | None = None,
    tags: Sequence[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> DocumentModel:
    """
    Update editable document fields.

    Only supplied fields are changed. Use `update_document_status` for
    lifecycle transitions.

    Args:
        db: Active async database session.
        document: Existing document model.
        title: Optional new title.
        content: Optional new content.
        document_type: Optional new document type.
        tags: Optional replacement tag list.
        extra_metadata: Optional replacement metadata dictionary.

    Returns:
        The refreshed document.

    Raises:
        ValidationError: If supplied values are invalid.
        PersistenceError: If the update cannot be completed.
    """
    if title is not None:
        if not title.strip():
            raise ValidationError("Document title cannot be empty.")

        document.title = title.strip()

    if content is not None:
        if not content.strip():
            raise ValidationError("Document content cannot be empty.")

        document.content = content

    if document_type is not None:
        safe_document_type = _normalize_document_type(document_type)

        if safe_document_type is None:
            raise ValidationError("Document type is required.")

        document.document_type = safe_document_type.value

    if tags is not None:
        document.tags = _normalize_tags(tags)

    if extra_metadata is not None:
        document.extra_metadata = extra_metadata

    document.updated_at = datetime.utcnow()

    document = await update_record(db, document)

    bind_document_context(str(document.id))

    logger.info(
        "document_updated",
        document_id=str(document.id),
        updated_fields=[
            field
            for field, value in (
                ("title", title),
                ("content", content),
                ("document_type", document_type),
                ("tags", tags),
                ("extra_metadata", extra_metadata),
            )
            if value is not None
        ],
    )

    return document


async def update_document_status(
    db: AsyncSession,
    document_id: str | UUID,
    status: str | DocumentStatus,
    *,
    tenant_id: str | None = None,
) -> DocumentModel:
    """
    Update a document lifecycle status.

    Args:
        db: Active async database session.
        document_id: Document UUID.
        status: New document status.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The refreshed document.

    Raises:
        ValidationError: If the status is unsupported.
        NotFoundError: If the document does not exist.
        PersistenceError: If the update cannot be completed.
    """
    safe_status = _normalize_document_status(status)

    if safe_status is None:
        raise ValidationError("Document status is required.")

    document = await get_document_or_404(
        db,
        document_id,
        tenant_id=tenant_id,
    )

    previous_status = document.status
    document.status = safe_status.value
    document.updated_at = datetime.utcnow()

    document = await update_record(db, document)

    bind_document_context(str(document.id))

    logger.info(
        "document_status_updated",
        document_id=str(document.id),
        previous_status=previous_status,
        status=document.status,
    )

    return document


async def archive_document(
    db: AsyncSession,
    document_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> DocumentModel:
    """
    Archive a document without permanently deleting it.

    Args:
        db: Active async database session.
        document_id: Document UUID.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The archived document.
    """
    return await update_document_status(
        db,
        document_id,
        DocumentStatus.ARCHIVED,
        tenant_id=tenant_id,
    )


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


async def delete_document(
    db: AsyncSession,
    document_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> None:
    """
    Permanently delete a document by ID.

    Prefer `archive_document` for user-initiated deletions unless regulatory,
    data-subject, or retention requirements require hard deletion.

    Args:
        db: Active async database session.
        document_id: Document UUID.
        tenant_id: Optional tenant isolation filter.

    Raises:
        NotFoundError: If the document does not exist.
        DatabaseIntegrityError: If related records prevent deletion.
        PersistenceError: If another database failure occurs.
    """
    document = await get_document_or_404(
        db,
        document_id,
        tenant_id=tenant_id,
    )

    document_id_value = str(document.id)

    await delete_record(db, document)

    logger.warning(
        "document_deleted",
        document_id=document_id_value,
        tenant_id=tenant_id,
    )