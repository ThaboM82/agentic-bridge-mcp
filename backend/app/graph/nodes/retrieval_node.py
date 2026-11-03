"""
LangGraph context-retrieval node.

Retrieves bounded, tenant-scoped design and domain context for downstream
workflow nodes. This node does not generate content or make workflow decisions.
"""

from __future__ import annotations

import logging

from app.core.enums import WorkflowStep
from app.graph.state import AgenticState
from app.graph.tools.vector_search_tools import (
    retrieve_design_context,
)


logger = logging.getLogger(
    "agentic_bridge.graph.nodes.retrieval"
)


async def retrieval_node(
    state: AgenticState,
) -> AgenticState:
    """Retrieve relevant context for the workflow."""
    execution_id = state.get("execution_id")
    tenant_id = state.get("tenant_id")

    logger.info(
        "retrieval_started",
        execution_id=execution_id,
    )

    # A production implementation should create the query embedding through
    # the embedding service using a bounded, sanitized retrieval query.
    query_embedding = state.get("query_embedding")

    if not query_embedding:
        return {
            "current_step": WorkflowStep.RETRIEVAL,
            "vector_context_matches": [],
            "vector_context_document_ids": [],
            "retrieval_completed_at": None,
            "current_error": (
                "Context retrieval requires a query embedding."
            ),
            "error_step": WorkflowStep.RETRIEVAL,
            "error_code": "missing_query_embedding",
            "recoverable_error": False,
            "execution_logs": [
                "ERROR: Context retrieval failed because no query embedding "
                "was supplied."
            ],
        }

    matches = await retrieve_design_context(
        db=state["db"],
        query_embedding=query_embedding,
        tenant_id=tenant_id,
        document_ids=state.get("document_ids"),
        model_name=state.get("retrieval_model"),
        limit=state.get("vector_search_limit", 5),
        similarity_threshold=state.get(
            "vector_search_similarity_threshold",
            0.2,
        ),
        execution_id=execution_id,
        retrieval_purpose="design_context",
    )

    document_ids = list(
        {
            match.get("document_id")
            for match in matches
            if match.get("document_id")
        }
    )

    return {
        "current_step": WorkflowStep.RETRIEVAL,
        "vector_context_matches": matches,
        "vector_context_document_ids": document_ids,
        "retrieval_completed_at": None,
        "current_error": None,
        "error_step": None,
        "error_code": None,
        "recoverable_error": False,
        "execution_logs": [
            f"INFO: Retrieved {len(matches)} context matches."
        ],
    }