"""
CRUD operations for document embeddings.

This module owns persistence and vector-search operations for document
embeddings stored in PostgreSQL with pgvector. API routes, ingestion services,
and workflow services should call these functions rather than constructing
SQLAlchemy queries directly.

Responsibilities:
- Create and bulk-create embeddings.
- Retrieve embeddings by ID, document, or source chunk.
- Perform pgvector cosine-distance semantic search.
- Update embedding content or vectors.
- Delete embeddings by ID or document.
- Map database failures to stable application exceptions.

Non-goals:
- Do not generate embeddings, invoke embedding providers, chunk documents, or
  implement retrieval ranking policy in this module.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import and_, delete, func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

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
from app.models.vector_model import DocumentEmbedding


logger = logging.getLogger("agentic_bridge.crud.vector_embedding")

MAX_SOURCE_CHUNK_LENGTH = 100_000
MAX_METADATA_JSON_CHARS = 16_000
MAX_VECTOR_DIMENSIONS = 10_000
DEFAULT_SEARCH_LIMIT = 5
MAX_SEARCH_LIMIT = 50
DEFAULT_SIMILARITY_THRESHOLD = 0.0
MAX_SIMILARITY_THRESHOLD = 1.0

SENSITIVE_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "secret",
        "secret_key",
        "api_key",
        "apikey",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "cookie",
        "database_url",
        "connection_string",
    }
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _normalize_embedding_id(
    embedding_id: str | UUID,
) -> UUID:
    """Validate and normalize an embedding ID."""
    if isinstance(embedding_id, UUID):
        return embedding_id

    if not isinstance(embedding_id, str) or not embedding_id.strip():
        raise ValidationError("embedding_id must be a non-empty string.")

    try:
        return UUID(embedding_id.strip())
    except ValueError as exc:
        raise ValidationError(
            "embedding_id must be a valid UUID."
        ) from exc


def _normalize_document_id(
    document_id: str | UUID,
) -> UUID:
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


def _validate_embedding_vector(
    embedding: Sequence[float],
) -> list[float]:
    """
    Validate an embedding vector.

    Args:
        embedding: Numeric embedding values.

    Returns:
        A validated list of finite float values.

    Raises:
        ValidationError: If the vector is empty, oversized, or non-numeric.
    """
    if embedding is None:
        raise ValidationError("Embedding vector is required.")

    if isinstance(embedding, (str, bytes)):
        raise ValidationError("Embedding vector must be a numeric array.")

    try:
        vector = [float(value) for value in embedding]
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "Embedding vector must contain only numeric values."
        ) from exc

    if not vector:
        raise ValidationError("Embedding vector cannot be empty.")

    if len(vector) > MAX_VECTOR_DIMENSIONS:
        raise ValidationError(
            f"Embedding vector may not exceed {MAX_VECTOR_DIMENSIONS} dimensions."
        )

    if any(value != value or value in (float("inf"), float("-inf")) for value in vector):
        raise ValidationError(
            "Embedding vector cannot contain NaN or infinite values."
        )

    return vector


def _validate_source_chunk(source_chunk: str | None) -> str | None:
    """Validate and bound embedded text content."""
    if source_chunk is None:
        return None

    if not isinstance(source_chunk, str):
        raise ValidationError("source_chunk must be a string.")

    normalized_chunk = source_chunk.strip()

    if not normalized_chunk:
        return None

    if len(normalized_chunk) > MAX_SOURCE_CHUNK_LENGTH:
        raise ValidationError(
            f"source_chunk may not exceed {MAX_SOURCE_CHUNK_LENGTH} characters."
        )

    return normalized_chunk


def _validate_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Validate embedding metadata and reject sensitive keys."""
    if metadata is None:
        return {}

    if not isinstance(metadata, dict):
        raise ValidationError("Embedding metadata must be an object.")

    for key in metadata:
        if str(key).lower() in SENSITIVE_METADATA_KEYS:
            raise ValidationError(
                f"Sensitive field '{key}' cannot be stored in embedding metadata."
            )

    if len(str(metadata)) > MAX_METADATA_JSON_CHARS:
        raise ValidationError(
            "Embedding metadata exceeds the maximum allowed size."
        )

    return metadata


def _normalize_search_limit(limit: int | None) -> int:
    """Normalize a semantic-search result limit."""
    if limit is None:
        return DEFAULT_SEARCH_LIMIT

    if limit < 1:
        raise ValidationError("Search limit must be at least 1.")

    return min(limit, MAX_SEARCH_LIMIT)


def _normalize_similarity_threshold(
    similarity_threshold: float | None,
) -> float:
    """
    Normalize a cosine-similarity threshold.

    This application uses a practical 0..1 similarity threshold.
    """
    if similarity_threshold is None:
        return DEFAULT_SIMILARITY_THRESHOLD

    if similarity_threshold < 0 or similarity_threshold > MAX_SIMILARITY_THRESHOLD:
        raise ValidationError(
            "Similarity threshold must be between 0 and 1."
        )

    return similarity_threshold


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _apply_tenant_filter(query, tenant_id: str | None):
    """Apply tenant isolation to an embedding query."""
    if tenant_id:
        return query.where(DocumentEmbedding.tenant_id == tenant_id)

    return query


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


async def create_embedding(
    db: AsyncSession,
    *,
    document_id: str | UUID,
    embedding: Sequence[float],
    source_chunk: str | None = None,
    chunk_index: int | None = None,
    model_name: str,
    model_version: str | None = None,
    tenant_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> DocumentEmbedding:
    """
    Create and persist a document embedding.

    Args:
        db: Active async database session.
        document_id: Parent document UUID.
        embedding: Embedding vector.
        source_chunk: Text chunk represented by the embedding.
        chunk_index: Zero-based chunk position within the document.
        model_name: Embedding model identifier.
        model_version: Optional embedding model version.
        tenant_id: Owning tenant identifier.
        metadata: Non-sensitive embedding metadata.

    Returns:
        The persisted embedding.

    Raises:
        ValidationError: If embedding values or metadata are invalid.
        DatabaseIntegrityError: If a constraint is violated.
        PersistenceError: If another database failure occurs.
    """
    document_uuid = _normalize_document_id(document_id)
    safe_vector = _validate_embedding_vector(embedding)
    safe_source_chunk = _validate_source_chunk(source_chunk)
    safe_metadata = _validate_metadata(metadata)

    if not model_name or not model_name.strip():
        raise ValidationError("model_name is required.")

    if chunk_index is not None and chunk_index < 0:
        raise ValidationError("chunk_index must be non-negative.")

    embedding_record = DocumentEmbedding(
        document_id=document_uuid,
        embedding=safe_vector,
        content=safe_source_chunk,
        chunk_index=chunk_index,
        model_name=model_name.strip(),
        model_version=model_version,
        tenant_id=tenant_id,
        extra_metadata=safe_metadata,
    )

    embedding_record = await create_record(db, embedding_record)

    bind_document_context(str(document_uuid))

    logger.info(
        "embedding_created",
        embedding_id=str(embedding_record.id),
        document_id=str(document_uuid),
        model_name=model_name,
        dimensions=len(safe_vector),
        chunk_index=chunk_index,
        tenant_id=tenant_id,
    )

    return embedding_record


async def create_embeddings_bulk(
    db: AsyncSession,
    embeddings: Sequence[dict[str, Any]],
) -> Sequence[DocumentEmbedding]:
    """
    Create multiple document embeddings in one transaction.

    Each dictionary must contain `document_id`, `embedding`, and `model_name`.

    Args:
        db: Active async database session.
        embeddings: Embedding creation payloads.

    Returns:
        The persisted embeddings.

    Raises:
        ValidationError: If no embeddings are supplied or a payload is invalid.
        DatabaseIntegrityError: If a constraint is violated.
        PersistenceError: If another database failure occurs.
    """
    if not embeddings:
        raise ValidationError("At least one embedding must be supplied.")

    models: list[DocumentEmbedding] = []

    for index, payload in enumerate(embeddings):
        document_id = payload.get("document_id")
        embedding = payload.get("embedding")
        model_name = payload.get("model_name")

        if document_id is None or embedding is None or not model_name:
            raise ValidationError(
                "Each embedding must include document_id, embedding, and model_name.",
                details={"embedding_index": index},
            )

        document_uuid = _normalize_document_id(document_id)
        safe_vector = _validate_embedding_vector(embedding)
        safe_source_chunk = _validate_source_chunk(
            payload.get("source_chunk")
        )
        safe_metadata = _validate_metadata(payload.get("metadata"))

        chunk_index = payload.get("chunk_index")

        if chunk_index is not None and chunk_index < 0:
            raise ValidationError(
                "chunk_index must be non-negative.",
                details={"embedding_index": index},
            )

        models.append(
            DocumentEmbedding(
                document_id=document_uuid,
                embedding=safe_vector,
                content=safe_source_chunk,
                chunk_index=chunk_index,
                model_name=str(model_name).strip(),
                model_version=payload.get("model_version"),
                tenant_id=payload.get("tenant_id"),
                extra_metadata=safe_metadata,
            )
        )

    created_embeddings = await create_records_bulk(db, models)

    logger.info(
        "embeddings_created_bulk",
        embedding_count=len(created_embeddings),
        document_id=str(created_embeddings[0].document_id)
        if created_embeddings
        else None,
        tenant_id=created_embeddings[0].tenant_id
        if created_embeddings
        else None,
    )

    return created_embeddings


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


async def get_embedding(
    db: AsyncSession,
    embedding_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> DocumentEmbedding | None:
    """
    Retrieve an embedding by ID.

    Args:
        db: Active async database session.
        embedding_id: Embedding UUID.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The embedding, or `None` when it does not exist or belongs to another
        tenant.
    """
    embedding_uuid = _normalize_embedding_id(embedding_id)

    query = select(DocumentEmbedding).where(
        DocumentEmbedding.id == embedding_uuid
    )

    query = _apply_tenant_filter(query, tenant_id)

    result = await db.execute(query)

    return result.scalar_one_or_none()


async def get_embedding_or_404(
    db: AsyncSession,
    embedding_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> DocumentEmbedding:
    """
    Retrieve an embedding by ID or raise `NotFoundError`.

    Args:
        db: Active async database session.
        embedding_id: Embedding UUID.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The matching embedding.

    Raises:
        NotFoundError: If the embedding does not exist or is outside the tenant.
    """
    embedding = await get_embedding(
        db,
        embedding_id,
        tenant_id=tenant_id,
    )

    if embedding is None:
        raise NotFoundError(
            "Embedding was not found.",
            details={"embedding_id": str(embedding_id)},
        )

    return embedding


async def get_embeddings_by_document(
    db: AsyncSession,
    document_id: str | UUID,
    *,
    tenant_id: str | None = None,
    model_name: str | None = None,
) -> Sequence[DocumentEmbedding]:
    """
    Retrieve embeddings associated with a document.

    Args:
        db: Active async database session.
        document_id: Parent document UUID.
        tenant_id: Optional tenant isolation filter.
        model_name: Optional embedding-model filter.

    Returns:
        Embeddings ordered by chunk index.
    """
    document_uuid = _normalize_document_id(document_id)

    query = select(DocumentEmbedding).where(
        DocumentEmbedding.document_id == document_uuid
    )

    query = _apply_tenant_filter(query, tenant_id)

    if model_name:
        query = query.where(
            DocumentEmbedding.model_name == model_name.strip()
        )

    query = query.order_by(DocumentEmbedding.chunk_index.asc())

    result = await db.execute(query)

    return result.scalars().all()


async def count_embeddings_by_document(
    db: AsyncSession,
    document_id: str | UUID,
    *,
    tenant_id: str | None = None,
    model_name: str | None = None,
) -> int:
    """
    Count embeddings associated with a document.

    Args:
        db: Active async database session.
        document_id: Parent document UUID.
        tenant_id: Optional tenant isolation filter.
        model_name: Optional embedding-model filter.

    Returns:
        Total embedding count for the document.
    """
    document_uuid = _normalize_document_id(document_id)

    filters = [DocumentEmbedding.document_id == document_uuid]

    if tenant_id:
        filters.append(DocumentEmbedding.tenant_id == tenant_id)

    if model_name:
        filters.append(DocumentEmbedding.model_name == model_name.strip())

    query = (
        select(func.count())
        .select_from(DocumentEmbedding)
        .where(and_(*filters))
    )

    result = await db.execute(query)

    return int(result.scalar_one())


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


async def search_embeddings_by_vector(
    db: AsyncSession,
    query_embedding: Sequence[float],
    *,
    tenant_id: str | None = None,
    document_id: str | UUID | None = None,
    model_name: str | None = None,
    limit: int | None = None,
    similarity_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """
    Perform cosine-similarity semantic search using pgvector.

    Args:
        db: Active async database session.
        query_embedding: Query vector.
        tenant_id: Optional tenant isolation filter.
        document_id: Optional document restriction.
        model_name: Optional embedding-model restriction.
        limit: Maximum number of matches to return.
        similarity_threshold: Minimum cosine similarity, from 0 to 1.

    Returns:
        A list of dictionaries containing embedding metadata, content,
        cosine distance, and cosine similarity.

    Raises:
        ValidationError: If search parameters are invalid.
        PersistenceError: If the search cannot be completed.
    """
    safe_query_vector = _validate_embedding_vector(query_embedding)
    safe_limit = _normalize_search_limit(limit)
    safe_threshold = _normalize_similarity_threshold(
        similarity_threshold
    )

    # pgvector cosine distance: 0 means identical, 2 means opposite.
    # Cosine similarity = 1 - cosine distance.
    maximum_distance = 1 - safe_threshold

    distance_expression = (
        DocumentEmbedding.embedding.cosine_distance(safe_query_vector)
    )

    filters = [distance_expression <= maximum_distance]

    if tenant_id:
        filters.append(DocumentEmbedding.tenant_id == tenant_id)

    if document_id is not None:
        filters.append(
            DocumentEmbedding.document_id
            == _normalize_document_id(document_id)
        )

    if model_name:
        filters.append(
            DocumentEmbedding.model_name == model_name.strip()
        )

    query = (
        select(
            DocumentEmbedding.id.label("embedding_id"),
            DocumentEmbedding.document_id.label("document_id"),
            DocumentEmbedding.content.label("content"),
            DocumentEmbedding.chunk_index.label("chunk_index"),
            DocumentEmbedding.model_name.label("model_name"),
            distance_expression.label("distance"),
            (1 - distance_expression).label("similarity"),
        )
        .where(and_(*filters))
        .order_by(distance_expression.asc())
        .limit(safe_limit)
    )

    try:
        result = await db.execute(query)
        rows = result.mappings().all()
    except SQLAlchemyError as exc:
        logger.exception("embedding_search_failed")

        raise PersistenceError(
            "Semantic search could not be completed."
        ) from exc

    matches = [
        {
            "embedding_id": str(row["embedding_id"]),
            "document_id": str(row["document_id"]),
            "content": row["content"],
            "chunk_index": row["chunk_index"],
            "model_name": row["model_name"],
            "distance": float(row["distance"]),
            "similarity": float(row["similarity"]),
        }
        for row in rows
    ]

    logger.info(
        "embedding_search_completed",
        result_count=len(matches),
        tenant_id=tenant_id,
        document_id=str(document_id) if document_id else None,
        model_name=model_name,
        limit=safe_limit,
        similarity_threshold=safe_threshold,
    )

    return matches


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


async def update_embedding(
    db: AsyncSession,
    embedding: DocumentEmbedding,
    *,
    source_chunk: str | None = None,
    embedding_vector: Sequence[float] | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> DocumentEmbedding:
    """
    Update an existing embedding.

    Args:
        db: Active async database session.
        embedding: Existing embedding model.
        source_chunk: Optional replacement text chunk.
        embedding_vector: Optional replacement vector.
        model_name: Optional replacement model name.
        model_version: Optional replacement model version.
        metadata: Optional replacement metadata.

    Returns:
        The refreshed embedding.

    Raises:
        ValidationError: If supplied values are invalid.
        PersistenceError: If the update cannot be completed.
    """
    if source_chunk is not None:
        embedding.content = _validate_source_chunk(source_chunk)

    if embedding_vector is not None:
        embedding.embedding = _validate_embedding_vector(
            embedding_vector
        )

    if model_name is not None:
        if not model_name.strip():
            raise ValidationError("model_name cannot be empty.")

        embedding.model_name = model_name.strip()

    if model_version is not None:
        embedding.model_version = model_version

    if metadata is not None:
        embedding.extra_metadata = _validate_metadata(metadata)

    embedding.updated_at = _utc_now()

    embedding = await update_record(db, embedding)

    bind_document_context(str(embedding.document_id))

    logger.info(
        "embedding_updated",
        embedding_id=str(embedding.id),
        document_id=str(embedding.document_id),
    )

    return embedding


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


async def delete_embedding(
    db: AsyncSession,
    embedding_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> None:
    """
    Permanently delete an embedding by ID.

    Args:
        db: Active async database session.
        embedding_id: Embedding UUID.
        tenant_id: Optional tenant isolation filter.

    Raises:
        NotFoundError: If the embedding does not exist.
        DatabaseIntegrityError: If related records prevent deletion.
        PersistenceError: If another database failure occurs.
    """
    embedding = await get_embedding_or_404(
        db,
        embedding_id,
        tenant_id=tenant_id,
    )

    embedding_id_value = str(embedding.id)
    document_id_value = str(embedding.document_id)

    await delete_record(db, embedding)

    logger.warning(
        "embedding_deleted",
        embedding_id=embedding_id_value,
        document_id=document_id_value,
        tenant_id=tenant_id,
    )


async def delete_embeddings_by_document(
    db: AsyncSession,
    document_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> int:
    """
    Delete all embeddings associated with a document.

    Use this when re-ingesting a document or enforcing a deletion request.

    Args:
        db: Active async database session.
        document_id: Parent document UUID.
        tenant_id: Optional tenant isolation filter.

    Returns:
        Number of deleted embeddings.

    Raises:
        DatabaseIntegrityError: If deletion violates a constraint.
        PersistenceError: If another database failure occurs.
    """
    document_uuid = _normalize_document_id(document_id)

    filters = [DocumentEmbedding.document_id == document_uuid]

    if tenant_id:
        filters.append(DocumentEmbedding.tenant_id == tenant_id)

    query = delete(DocumentEmbedding).where(and_(*filters))

    try:
        result = await db.execute(query)
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()

        logger.exception(
            "embedding_bulk_delete_integrity_error",
            document_id=str(document_uuid),
        )

        raise DatabaseIntegrityError(
            "Embeddings could not be deleted because related records exist."
        ) from exc

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "embedding_bulk_delete_failed",
            document_id=str(document_uuid),
        )

        raise PersistenceError(
            "Embeddings could not be deleted."
        ) from exc

    deleted_count = int(result.rowcount or 0)

    bind_document_context(str(document_uuid))

    logger.warning(
        "embeddings_deleted_by_document",
        document_id=str(document_uuid),
        deleted_count=deleted_count,
        tenant_id=tenant_id,
    )

    return deleted_count