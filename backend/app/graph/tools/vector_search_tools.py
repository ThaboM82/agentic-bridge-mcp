"""
Vector-search tools for workflow nodes.

Provides a narrow interface around embedding retrieval. Nodes should use these
functions rather than importing CRUD modules directly.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.vector_embedding_crud import search_embeddings_by_vector
from app.core.exceptions import ValidationError


logger = logging.getLogger("agentic_bridge.graph.tools.vector_search")

MAX_QUERY_VECTOR_DIMENSIONS = 10_000


async def retrieve_design_context(
    db: AsyncSession,
    query_embedding: Sequence[float],
    *,
    tenant_id: str | None = None,
    document_ids: Sequence[str | UUID] | None = None,
    model_name: str | None = None,
    limit: int = 5,
    similarity_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant design-system context for a workflow.

    Args:
        db: Active async database session.
        query_embedding: Query embedding vector.
        tenant_id: Tenant isolation filter.
        document_ids: Optional document restriction.
        model_name: Optional embedding-model restriction.
        limit: Maximum number of matches.
        similarity_threshold: Minimum cosine similarity.

    Returns:
        A list of retrieval matches.
    """
    if not query_embedding:
        raise ValidationError("Query embedding vector is required.")

    if len(query_embedding) > MAX_QUERY_VECTOR_DIMENSIONS:
        raise ValidationError(
            "Query embedding vector exceeds the allowed dimensions."
        )

    matches: list[dict[str, Any]] = []

    if document_ids:
        for document_id in document_ids:
            document_matches = await search_embeddings_by_vector(
                db,
                query_embedding,
                tenant_id=tenant_id,
                document_id=document_id,
                model_name=model_name,
                limit=limit,
                similarity_threshold=similarity_threshold,
            )

            matches.extend(document_matches)

        matches.sort(
            key=lambda match: match.get("similarity", 0),
            reverse=True,
        )

        return matches[:limit]

    matches = await search_embeddings_by_vector(
        db,
        query_embedding,
        tenant_id=tenant_id,
        model_name=model_name,
        limit=limit,
        similarity_threshold=similarity_threshold,
    )

    logger.info(
        "workflow_vector_context_retrieved",
        match_count=len(matches),
        tenant_id=tenant_id,
        model_name=model_name,
    )

    return matches