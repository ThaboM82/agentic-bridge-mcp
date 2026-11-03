"""
Document ingestion and semantic-search API routes.

These routes provide tenant-scoped document ingestion, vector search,
document listing, retrieval, updates, deletion, statistics, and chunk-level
access. Documents are stored as chunks with embeddings for pgvector-based
semantic retrieval.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Any, List
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from sqlalchemy import (
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import (
    get_current_user,
)
from app.core.database import (
    get_db,
)
from app.models import (
    DocumentChunk,
)
from app.schemas.documents import (
    DocumentCreate,
    DocumentResponse,
)
from app.schemas.search import (
    SearchQuery,
    SearchResponse,
    SearchResultItem,
)
from app.services.document_service import (
    document_service,
)


logger = logging.getLogger(
    __name__
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents & Vector Search"],
    dependencies=[
        Depends(
            get_current_user
        )
    ],
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


async def get_pagination(
    skip: Annotated[
        int,
        Query(
            ge=0,
            le=100_000,
            description="Number of records to skip.",
        ),
    ] = 0,
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=100,
            description="Maximum number of records to return.",
        ),
    ] = 25,
) -> tuple[int, int]:
    """Return validated pagination parameters."""
    return skip, limit


Pagination = Annotated[
    tuple[int, int],
    Depends(get_pagination),
]

DatabaseSession = Annotated[
    AsyncSession,
    Depends(get_db),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_document_or_404(
    db: AsyncSession,
    document_id: UUID,
) -> DocumentChunk:
    """Retrieve a document chunk or raise a 404 error."""
    document = (
        await db.get(
            DocumentChunk,
            document_id,
        )
    )

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return document


def _apply_document_filters(
    documents: List[DocumentChunk],
    *,
    document_type: str | None,
    source: str | None,
    created_after: datetime | None,
    created_before: datetime | None,
) -> List[DocumentChunk]:
    """Apply optional in-memory filters to document results."""
    filtered_documents = documents

    if document_type:
        normalized_type = (
            document_type.strip().lower()
        )

        filtered_documents = [
            document
            for document in filtered_documents
            if document.document_type
            and document.document_type.lower()
            == normalized_type
        ]

    if source:
        normalized_source = (
            source.strip().lower()
        )

        filtered_documents = [
            document
            for document in filtered_documents
            if document.source
            and document.source.lower()
            == normalized_source
        ]

    if created_after:
        filtered_documents = [
            document
            for document in filtered_documents
            if document.created_at
            >= created_after
        ]

    if created_before:
        filtered_documents = [
            document
            for document in filtered_documents
            if document.created_at
            <= created_before
        ]

    return filtered_documents


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest document",
    description=(
        "Ingests a document, generates an embedding, and persists it in "
        "PostgreSQL for semantic retrieval."
    ),
)
async def ingest_document(
    payload: DocumentCreate,
    db: DatabaseSession,
    request: Request,
) -> DocumentResponse:
    """Create a document chunk and generate its vector embedding."""
    try:
        document = (
            await document_service.create_document(
                db,
                payload,
            )
        )

        await db.commit()

        logger.info(
            "Document ingested.",
            extra={
                "document_id": str(
                    document.id
                ),
                "title": payload.title,
                "request_id": request.headers.get(
                    "X-Request-ID"
                ),
            },
        )

        return document
    except ValueError as exc:
        await db.rollback()

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        await db.rollback()

        logger.exception(
            "Document ingestion failed request_id=%s title=%r",
            request.headers.get(
                "X-Request-ID"
            ),
            payload.title,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Document ingestion failed while generating or "
                "storing its embedding."
            ),
        ) from exc


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search",
    description=(
        "Performs semantic similarity search across stored document chunks "
        "using pgvector cosine distance."
    ),
)
async def semantic_search(
    payload: SearchQuery,
    db: DatabaseSession,
    request: Request,
) -> SearchResponse:
    """Search stored documents using vector similarity."""
    try:
        results: list[SearchResultItem] = (
            await document_service.search_documents(
                db,
                payload,
            )
        )

        await db.commit()

        logger.info(
            "Semantic search completed.",
            extra={
                "query": payload.query,
                "result_count": len(
                    results
                ),
                "request_id": request.headers.get(
                    "X-Request-ID"
                ),
            },
        )

        return SearchResponse(
            query=payload.query,
            results=results,
            total_results=len(
                results
            ),
        )
    except ValueError as exc:
        await db.rollback()

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        await db.rollback()

        logger.exception(
            "Semantic search failed request_id=%s query=%r",
            request.headers.get(
                "X-Request-ID"
            ),
            payload.query,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Semantic search failed while generating or "
                "executing the vector query."
            ),
        ) from exc


# ---------------------------------------------------------------------------
# Document collection
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=List[DocumentResponse],
    summary="List documents",
    description="Returns a paginated list of ingested document chunks.",
)
async def list_documents(
    db: DatabaseSession,
    pagination: Pagination,
    document_type: Annotated[
        str | None,
        Query(
            max_length=100,
            description="Filter by document type.",
        ),
    ] = None,
    source: Annotated[
        str | None,
        Query(
            max_length=200,
            description="Filter by document source.",
        ),
    ] = None,
    created_after: Annotated[
        datetime | None,
        Query(
            description=(
                "Return documents created on or after this time."
            ),
        ),
    ] = None,
    created_before: Annotated[
        datetime | None,
        Query(
            description=(
                "Return documents created on or before this time."
            ),
        ),
    ] = None,
) -> List[DocumentResponse]:
    """List document chunks with pagination and optional filters."""
    skip, limit = pagination

    statement = (
        select(DocumentChunk)
        .order_by(
            DocumentChunk.created_at.desc(),
            DocumentChunk.id.desc(),
        )
        .offset(skip)
        .limit(limit)
    )

    result = await db.execute(
        statement
    )

    documents = list(
        result.scalars().all()
    )

    documents = _apply_document_filters(
        documents,
        document_type=document_type,
        source=source,
        created_after=created_after,
        created_before=created_before,
    )

    await db.commit()

    return documents


@router.get(
    "/stats",
    response_model=dict,
    summary="Document statistics",
    description="Returns basic ingestion and vector-index statistics.",
)
async def get_document_stats(
    db: DatabaseSession,
) -> dict:
    """Return document count and vector-index metadata."""
    count_statement = (
        select(func.count())
        .select_from(DocumentChunk)
    )

    result = await db.execute(
        count_statement
    )

    total_documents = (
        result.scalar_one()
    )

    await db.commit()

    return {
        "total_documents": total_documents,
        "embedding_model": (
            "text-embedding-3-small"
        ),
        "vector_index": "pgvector",
    }


# ---------------------------------------------------------------------------
# Individual documents
# ---------------------------------------------------------------------------


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document",
    description="Retrieve a stored document chunk by ID.",
)
async def get_document(
    document_id: UUID,
    db: DatabaseSession,
) -> DocumentResponse:
    """Retrieve a document chunk by ID."""
    document = (
        await _get_document_or_404(
            db,
            document_id,
        )
    )

    await db.commit()

    return document


@router.delete(
    "/{document_id}",
    response_model=dict,
    summary="Delete document",
    description="Delete a stored document chunk by ID.",
)
async def delete_document(
    document_id: UUID,
    db: DatabaseSession,
) -> dict:
    """Delete a document chunk by ID."""
    document = (
        await _get_document_or_404(
            db,
            document_id,
        )
    )

    await db.delete(
        document
    )

    await db.commit()

    logger.info(
        "Document deleted.",
        extra={
            "document_id": str(
                document_id
            ),
        },
    )

    return {
        "document_id": str(
            document_id
        ),
        "deleted": True,
    }