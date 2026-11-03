"""
Knowledge retrieval API routes.

These routes expose tenant-scoped semantic search, hybrid retrieval, document
and chunk retrieval, retrieval previews, and search suggestions. Retrieval
operations combine vector similarity with optional metadata filters while
preserving tenant isolation.
"""

from __future__ import annotations

import logging
import time
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
from pydantic import (
    BaseModel,
    Field,
)

from app.api.dependencies import (
    CurrentPrincipal,
    DatabaseSession,
    Pagination,
    require_scoped_access,
)
from app.api.pagination import (
    build_paginated_response,
)
from app.core.exceptions import ValidationError
from app.repositories.document_repository import (
    DocumentRepository,
)
from app.repositories.audit_log_repository import (
    AuditLogRepository,
)
from app.services.retrieval_service import (
    retrieval_service,
)


logger = logging.getLogger(
    "agentic_bridge.api.retrieval"
)


router = APIRouter(
    prefix="/retrieval",
    tags=["Retrieval"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RetrievalQueryRequest(BaseModel):
    """Request for semantic or hybrid knowledge retrieval."""

    query: str = Field(
        min_length=1,
        max_length=2_000,
        description="Natural-language query to retrieve relevant knowledge.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of results to return.",
    )
    document_type: str | None = Field(
        default=None,
        max_length=100,
        description="Optional document-type filter.",
    )
    source: str | None = Field(
        default=None,
        max_length=200,
        description="Optional document-source filter.",
    )
    created_after: datetime | None = Field(
        default=None,
        description="Only include documents created on or after this time.",
    )
    created_before: datetime | None = Field(
        default=None,
        description="Only include documents created on or before this time.",
    )
    minimum_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum normalized relevance score.",
    )
    metadata_filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured metadata filters.",
    )


class HybridRetrievalQueryRequest(
    RetrievalQueryRequest
):
    """Request for hybrid semantic and keyword retrieval."""

    keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Relative weight applied to keyword matches.",
    )
    semantic_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Relative weight applied to vector similarity.",
    )


class RetrievalResultItem(BaseModel):
    """A single retrieved knowledge chunk."""

    chunk_id: UUID
    document_id: UUID
    title: str
    content: str
    score: float
    document_type: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class RetrievalResponse(BaseModel):
    """Response envelope for retrieval operations."""

    query: str
    results: list[RetrievalResultItem]
    total_results: int
    retrieval_mode: str
    elapsed_ms: float


class RetrievalSuggestionResponse(BaseModel):
    """Response envelope for query suggestions."""

    query: str
    suggestions: list[str]


# ---------------------------------------------------------------------------
# Semantic retrieval
# ---------------------------------------------------------------------------


@router.post(
    "/search",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic retrieval",
    description=(
        "Retrieve the most relevant knowledge chunks using vector similarity "
        "and optional metadata filters."
    ),
)
async def semantic_retrieval(
    payload: RetrievalQueryRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "retrieval:read"
            )
        ),
    ],
    db: DatabaseSession,
    request: Request,
) -> RetrievalResponse:
    """Perform tenant-scoped semantic retrieval."""
    started_at = time.perf_counter()

    try:
        raw_results = (
            await retrieval_service.semantic_search(
                db,
                tenant_id=principal.tenant_id,
                query=payload.query,
                top_k=payload.top_k,
                document_type=payload.document_type,
                source=payload.source,
                created_after=payload.created_after,
                created_before=payload.created_before,
                metadata_filters=payload.metadata_filters,
            )
        )
    except ValidationError as exc:
        await db.rollback()

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        await db.rollback()

        logger.exception(
            "Semantic retrieval failed.",
            extra={
                "tenant_id": str(
                    principal.tenant_id
                ),
                "user_id": str(
                    principal.user_id
                ),
                "request_id": request.headers.get(
                    "X-Request-ID"
                ),
                "query": payload.query,
            },
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Retrieval failed while generating or executing the "
                "semantic query."
            ),
        ) from exc

    results = [
        RetrievalResultItem(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            title=result.title,
            content=result.content,
            score=result.score,
            document_type=result.document_type,
            source=result.source,
            metadata=result.metadata,
        )
        for result in raw_results
        if result.score
        >= payload.minimum_score
    ]

    elapsed_ms = round(
        (
            time.perf_counter()
            - started_at
        )
        * 1_000,
        2,
    )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="retrieval_performed",
        actor_id=principal.user_id,
        resource_type="retrieval",
        action="semantic_search",
        outcome="completed",
        metadata={
            "query": payload.query,
            "result_count": len(
                results
            ),
            "elapsed_ms": elapsed_ms,
        },
    )

    await db.commit()

    logger.info(
        "Semantic retrieval completed.",
        extra={
            "tenant_id": str(
                principal.tenant_id
            ),
            "user_id": str(
                principal.user_id
            ),
            "result_count": len(
                results
            ),
            "elapsed_ms": elapsed_ms,
        },
    )

    return RetrievalResponse(
        query=payload.query,
        results=results,
        total_results=len(
            results
        ),
        retrieval_mode="semantic",
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------


@router.post(
    "/hybrid-search",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Hybrid retrieval",
    description=(
        "Retrieve knowledge using a weighted combination of semantic vector "
        "similarity and keyword matching."
    ),
)
async def hybrid_retrieval(
    payload: HybridRetrievalQueryRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "retrieval:read"
            )
        ),
    ],
    db: DatabaseSession,
    request: Request,
) -> RetrievalResponse:
    """Perform tenant-scoped hybrid semantic and keyword retrieval."""
    started_at = time.perf_counter()

    total_weight = (
        payload.keyword_weight
        + payload.semantic_weight
    )

    if total_weight <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "At least one of keyword_weight or semantic_weight "
                "must be greater than zero."
            ),
        )

    try:
        raw_results = (
            await retrieval_service.hybrid_search(
                db,
                tenant_id=principal.tenant_id,
                query=payload.query,
                top_k=payload.top_k,
                keyword_weight=payload.keyword_weight,
                semantic_weight=payload.semantic_weight,
                document_type=payload.document_type,
                source=payload.source,
                created_after=payload.created_after,
                created_before=payload.created_before,
                metadata_filters=payload.metadata_filters,
            )
        )
    except ValidationError as exc:
        await db.rollback()

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        await db.rollback()

        logger.exception(
            "Hybrid retrieval failed.",
            extra={
                "tenant_id": str(
                    principal.tenant_id
                ),
                "user_id": str(
                    principal.user_id
                ),
                "request_id": request.headers.get(
                    "X-Request-ID"
                ),
                "query": payload.query,
            },
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Retrieval failed while executing the hybrid query."
            ),
        ) from exc

    results = [
        RetrievalResultItem(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            title=result.title,
            content=result.content,
            score=result.score,
            document_type=result.document_type,
            source=result.source,
            metadata=result.metadata,
        )
        for result in raw_results
        if result.score
        >= payload.minimum_score
    ]

    elapsed_ms = round(
        (
            time.perf_counter()
            - started_at
        )
        * 1_000,
        2,
    )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="retrieval_performed",
        actor_id=principal.user_id,
        resource_type="retrieval",
        action="hybrid_search",
        outcome="completed",
        metadata={
            "query": payload.query,
            "result_count": len(
                results
            ),
            "keyword_weight": payload.keyword_weight,
            "semantic_weight": payload.semantic_weight,
            "elapsed_ms": elapsed_ms,
        },
    )

    await db.commit()

    logger.info(
        "Hybrid retrieval completed.",
        extra={
            "tenant_id": str(
                principal.tenant_id
            ),
            "user_id": str(
                principal.user_id
            ),
            "result_count": len(
                results
            ),
            "elapsed_ms": elapsed_ms,
        },
    )

    return RetrievalResponse(
        query=payload.query,
        results=results,
        total_results=len(
            results
        ),
        retrieval_mode="hybrid",
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Document and chunk retrieval
# ---------------------------------------------------------------------------


@router.get(
    "/documents",
    summary="List retrievable documents",
    description=(
        "List tenant-scoped documents available to the retrieval system."
    ),
)
async def list_retrievable_documents(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "retrieval:read"
            )
        ),
    ],
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
    search: Annotated[
        str | None,
        Query(
            max_length=200,
            description="Search document titles.",
        ),
    ] = None,
):
    """List tenant-scoped documents available for retrieval."""
    documents, total_count = (
        await DocumentRepository.list_documents(
            db,
            tenant_id=principal.tenant_id,
            skip=pagination.skip,
            limit=pagination.limit,
            document_type_filter=document_type,
            source_filter=source,
            search_term=search,
        )
    )

    await db.commit()

    return build_paginated_response(
        documents,
        skip=pagination.skip,
        limit=pagination.limit,
        total_count=total_count,
    )


@router.get(
    "/documents/{document_id}/chunks",
    summary="List document chunks",
    description=(
        "List retrievable chunks for a tenant-scoped document."
    ),
)
async def list_document_chunks(
    document_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "retrieval:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
):
    """List retrieval chunks associated with a document."""
    document = (
        await DocumentRepository.get_by_id(
            db,
            record_id=document_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    chunks, total_count = (
        await DocumentRepository.list_chunks_for_document(
            db,
            document_id=document_id,
            tenant_id=principal.tenant_id,
            skip=pagination.skip,
            limit=pagination.limit,
        )
    )

    await db.commit()

    return build_paginated_response(
        chunks,
        skip=pagination.skip,
        limit=pagination.limit,
        total_count=total_count,
    )


# ---------------------------------------------------------------------------
# Retrieval utilities
# ---------------------------------------------------------------------------


@router.post(
    "/preview",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieval preview",
    description=(
        "Preview retrieval results without recording a full retrieval event."
    ),
)
async def preview_retrieval(
    payload: RetrievalQueryRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "retrieval:read"
            )
        ),
    ],
    db: DatabaseSession,
) -> RetrievalResponse:
    """Preview retrieval results for query tuning and diagnostics."""
    started_at = time.perf_counter()

    try:
        raw_results = (
            await retrieval_service.semantic_search(
                db,
                tenant_id=principal.tenant_id,
                query=payload.query,
                top_k=payload.top_k,
                document_type=payload.document_type,
                source=payload.source,
                created_after=payload.created_after,
                created_before=payload.created_before,
                metadata_filters=payload.metadata_filters,
            )
        )
    except ValidationError as exc:
        await db.rollback()

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    results = [
        RetrievalResultItem(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            title=result.title,
            content=result.content,
            score=result.score,
            document_type=result.document_type,
            source=result.source,
            metadata=result.metadata,
        )
        for result in raw_results
        if result.score
        >= payload.minimum_score
    ]

    elapsed_ms = round(
        (
            time.perf_counter()
            - started_at
        )
        * 1_000,
        2,
    )

    await db.commit()

    return RetrievalResponse(
        query=payload.query,
        results=results,
        total_results=len(
            results
        ),
        retrieval_mode="preview",
        elapsed_ms=elapsed_ms,
    )


@router.get(
    "/suggestions",
    response_model=RetrievalSuggestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Query suggestions",
    description=(
        "Return lightweight query suggestions based on recent successful "
        "tenant retrieval activity."
    ),
)
async def get_retrieval_suggestions(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "retrieval:read"
            )
        ),
    ],
    db: DatabaseSession,
    query: Annotated[
        str,
        Query(
            min_length=1,
            max_length=200,
            description="Partial query used to generate suggestions.",
        ),
    ],
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=20,
            description="Maximum number of suggestions to return.",
        ),
    ] = 5,
) -> RetrievalSuggestionResponse:
    """Return tenant-scoped retrieval query suggestions."""
    try:
        suggestions = (
            await retrieval_service.suggest_queries(
                db,
                tenant_id=principal.tenant_id,
                query=query,
                limit=limit,
            )
        )
    except ValidationError as exc:
        await db.rollback()

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    await db.commit()

    return RetrievalSuggestionResponse(
        query=query,
        suggestions=suggestions,
    )