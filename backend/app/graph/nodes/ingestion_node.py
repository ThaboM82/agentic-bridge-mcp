"""
LangGraph ingestion node.

This node prepares workflow context by loading document summaries and relevant
retrieval context. It is intentionally defensive: ingestion failures should be
recorded in workflow state rather than crashing the graph.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.enums import (
    JobStatus,
    WorkflowStep,
)
from app.services.document_service import (
    get_document_summary_for_workflow,
)


logger = logging.getLogger(
    "agentic_bridge.graph.nodes.ingestion",
)


MAX_DOCUMENT_IDS = 25
MAX_SUMMARY_CHARS = 8_000
MAX_RETRIEVAL_MATCHES = 20


async def ingestion_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Prepare ingestion context for the workflow.

    Args:
        state: Current LangGraph workflow state.

    Returns:
        A partial state update containing ingestion status, document summaries,
        retrieval context, and error information.
    """
    execution_id = state.get("execution_id")
    tenant_id = state.get("tenant_id")
    document_ids = state.get("document_ids") or []

    logger.info(
        "ingestion_node_started",
        extra={
            "execution_id": execution_id,
            "tenant_id": tenant_id,
            "document_count": len(document_ids),
        },
    )

    if not document_ids:
        logger.warning(
            "ingestion_node_no_documents",
            extra={
                "execution_id": execution_id,
                "tenant_id": tenant_id,
            },
        )

        return {
            "current_step": WorkflowStep.INGESTION,
            "job_status": JobStatus.COMPLETED,
            "document_summaries": [],
            "retrieval_matches": [],
            "ingestion_completed": True,
            "recoverable_error": False,
            "current_error": None,
        }

    if len(document_ids) > MAX_DOCUMENT_IDS:
        document_ids = document_ids[
            :MAX_DOCUMENT_IDS
        ]

        logger.warning(
            "ingestion_node_document_ids_truncated",
            extra={
                "execution_id": execution_id,
                "tenant_id": tenant_id,
                "max_document_ids": (
                    MAX_DOCUMENT_IDS
                ),
            },
        )

    document_summaries: list[dict[str, Any]] = []
    retrieval_matches: list[dict[str, Any]] = []
    errors: list[str] = []

    for document_id in document_ids:
        try:
            summary = (
                await get_document_summary_for_workflow(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    execution_id=execution_id,
                )
            )

        except Exception as exc:
            logger.exception(
                "ingestion_node_document_failed",
                extra={
                    "execution_id": execution_id,
                    "tenant_id": tenant_id,
                    "document_id": str(
                        document_id,
                    ),
                },
            )

            errors.append(
                f"Document {document_id} could "
                "not be ingested.",
            )

            continue

        if not summary:
            errors.append(
                f"Document {document_id} returned "
                "no ingestion summary.",
            )

            continue

        normalized_summary = {
            "document_id": str(
                summary.get("document_id", document_id),
            ),
            "title": summary.get("title"),
            "summary": str(
                summary.get("summary", ""),
            )[:MAX_SUMMARY_CHARS],
            "source": summary.get("source"),
            "document_type": summary.get(
                "document_type",
            ),
            "metadata": summary.get(
                "metadata",
                {},
            ),
        }

        document_summaries.append(
            normalized_summary,
        )

        matches = summary.get(
            "retrieval_matches",
            [],
        )

        if isinstance(matches, list):
            retrieval_matches.extend(
                matches[:MAX_RETRIEVAL_MATCHES],
            )

    if errors and not document_summaries:
        logger.error(
            "ingestion_node_failed",
            extra={
                "execution_id": execution_id,
                "tenant_id": tenant_id,
                "error_count": len(errors),
            },
        )

        return {
            "current_step": WorkflowStep.INGESTION,
            "job_status": JobStatus.FAILED,
            "document_summaries": [],
            "retrieval_matches": [],
            "ingestion_completed": False,
            "recoverable_error": True,
            "current_error": (
                "No documents could be ingested."
            ),
            "ingestion_errors": errors,
        }

    if errors:
        logger.warning(
            "ingestion_node_partially_completed",
            extra={
                "execution_id": execution_id,
                "tenant_id": tenant_id,
                "successful_documents": len(
                    document_summaries,
                ),
                "failed_documents": len(errors),
            },
        )

        return {
            "current_step": WorkflowStep.INGESTION,
            "job_status": (
                JobStatus.PARTIALLY_COMPLETED
            ),
            "document_summaries": (
                document_summaries
            ),
            "retrieval_matches": retrieval_matches,
            "ingestion_completed": True,
            "recoverable_error": False,
            "current_error": None,
            "ingestion_errors": errors,
        }

    logger.info(
        "ingestion_node_completed",
        extra={
            "execution_id": execution_id,
            "tenant_id": tenant_id,
            "document_count": len(
                document_summaries,
            ),
            "retrieval_match_count": len(
                retrieval_matches,
            ),
        },
    )

    return {
        "current_step": WorkflowStep.INGESTION,
        "job_status": JobStatus.COMPLETED,
        "document_summaries": document_summaries,
        "retrieval_matches": retrieval_matches,
        "ingestion_completed": True,
        "recoverable_error": False,
        "current_error": None,
    }


__all__ = [
    "ingestion_node",
]