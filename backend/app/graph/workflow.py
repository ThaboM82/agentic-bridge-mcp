"""
LangGraph workflow orchestrator for AgenticBridge MCP.

This module defines the executable workflow graph. It coordinates context
retrieval, ingestion, requirements extraction, design analysis, code
generation, validation, human review, publishing, cancellation, and recovery.

Architecture rules:
- Nodes own one workflow responsibility each.
- Nodes return only state fields they change.
- Accumulating fields use LangGraph reducers and must return only new items.
- This module coordinates routing; it does not contain business rules.
- Services and nodes own LLM calls, persistence, approvals, and validation.
- Routing functions are imported from `graph.routing`; recovery policy is
  owned by `graph.recovery`; observability is owned by `graph.telemetry`.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import (
    END,
    StateGraph,
)

from app.core.config import settings
from app.core.enums import (
    ApprovalStatus,
    JobStatus,
    WorkflowStep,
)
from app.core.exceptions import (
    AppError,
    WorkflowStateError,
)
from app.graph.checkpoints import (
    create_checkpointer,
)
from app.graph.interrupts import (
    create_interrupt_payload,
    request_human_review,
)
from app.graph.nodes.code_generation_node import (
    code_generation_node,
)
from app.graph.nodes.design_node import (
    design_analysis_node,
)
from app.graph.nodes.ingestion_node import (
    ingestion_node,
)
from app.graph.nodes.publishing_node import (
    publishing_node,
)
from app.graph.nodes.requirements_node import (
    requirements_node,
)
from app.graph.nodes.retrieval_node import (
    retrieval_node,
)
from app.graph.nodes.review_node import (
    review_node,
)
from app.graph.nodes.validation_node import (
    validation_node,
)
from app.graph.recovery import (
    decide_recovery_action,
    prepare_retry,
)
from app.graph.routing import (
    ROUTE_AWAITING_REVIEW,
    ROUTE_CANCELLED,
    ROUTE_COMPLETED,
    ROUTE_EXTRACT_REQUIREMENTS,
    ROUTE_FAILED,
    ROUTE_GENERATE_CODE,
    ROUTE_GENERATE_DESIGN,
    ROUTE_HUMAN_REVIEW,
    ROUTE_PUBLISH_ARTIFACTS,
    ROUTE_RETRY,
    ROUTE_RETRIEVE_CONTEXT,
    ROUTE_VALIDATE_OUTPUT,
    route_after_design,
    route_after_generation,
    route_after_publication,
    route_after_requirements,
    route_after_retrieval,
    route_after_review,
    route_after_validation,
    route_on_error,
    route_on_interrupt,
    route_on_start,
)
from app.graph.state import (
    AgenticState,
    clear_error,
    request_cancellation,
    request_pause,
    resolve_active_interrupt,
    resume_workflow,
    set_current_step,
    set_error,
    set_workflow_status,
)
from app.graph.telemetry import (
    emit_human_review_requested,
    emit_human_review_resolved,
    emit_workflow_cancelled,
    emit_workflow_completed,
    emit_workflow_failed,
    emit_workflow_started,
)


logger = logging.getLogger(
    "agentic_bridge.graph.workflow",
)


WORKFLOW_NAME = (
    "agentic_bridge_generation_workflow"
)
WORKFLOW_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Compatibility routing wrappers
# ---------------------------------------------------------------------------


def route_after_ingestion(
    state: AgenticState,
) -> str:
    """
    Route after ingestion.

    Kept as a compatibility wrapper for existing tests and integrations. New
    workflow edges should use `route_after_retrieval` from `graph.routing`.
    """
    if state.get("current_error"):
        return "handle_error"

    if not state.get(
        "extracted_requirements",
    ):
        return "handle_error"

    return "extract_requirements"


# ---------------------------------------------------------------------------
# Coordination nodes
# ---------------------------------------------------------------------------


async def begin_workflow_node(
    state: AgenticState,
) -> AgenticState:
    """
    Initialize workflow execution state.
    """
    execution_id = state.get("execution_id")
    brief_id = state.get("brief_id")
    tenant_id = state.get("tenant_id")

    logger.info(
        "workflow_started",
        extra={
            "execution_id": execution_id,
            "brief_id": brief_id,
            "tenant_id": tenant_id,
        },
    )

    emit_workflow_started(
        execution_id=execution_id,
        brief_id=brief_id,
        tenant_id=tenant_id,
        user_id=state.get("user_id"),
        workflow_name=state.get(
            "workflow_name",
        ),
        workflow_version=state.get(
            "workflow_version",
        ),
    )

    return {
        **set_workflow_status(
            state,
            JobStatus.PROCESSING,
        ),
        **set_current_step(
            state,
            WorkflowStep.INGESTION,
        ),
        "started_at": state.get(
            "started_at",
        ),
        "execution_logs": [
            "INFO: Workflow execution started.",
        ],
        "current_error": None,
        "error_step": None,
        "error_code": None,
        "recoverable_error": False,
    }


async def cancellation_node(
    state: AgenticState,
) -> AgenticState:
    """
    Handle an explicitly cancelled workflow.
    """
    execution_id = state.get("execution_id")
    brief_id = state.get("brief_id")
    tenant_id = state.get("tenant_id")

    logger.warning(
        "workflow_cancelled",
        extra={
            "execution_id": execution_id,
            "brief_id": brief_id,
            "tenant_id": tenant_id,
        },
    )

    emit_workflow_cancelled(
        execution_id=execution_id,
        brief_id=brief_id,
        tenant_id=tenant_id,
        user_id=state.get("cancelled_by"),
        workflow_name=state.get(
            "workflow_name",
        ),
        cancelled_by=state.get(
            "cancelled_by",
        ),
    )

    cancellation_updates = (
        request_cancellation(
            state,
            cancelled_by=state.get(
                "cancelled_by",
            ),
            reason=state.get(
                "cancellation_reason",
            ),
        )
    )

    return {
        **cancellation_updates,
        **set_current_step(
            state,
            WorkflowStep.CANCELLED,
        ),
        "completed_at": cancellation_updates.get(
            "cancelled_at",
        ),
    }


async def pause_node(
    state: AgenticState,
) -> AgenticState:
    """
    Pause workflow execution at a safe boundary.
    """
    execution_id = state.get("execution_id")

    logger.warning(
        "workflow_paused",
        extra={
            "execution_id": execution_id,
            "reason": state.get(
                "pause_reason",
            ),
        },
    )

    pause_updates = request_pause(
        state,
        paused_by=state.get(
            "paused_by",
        ),
        reason=state.get(
            "pause_reason",
        ),
    )

    return {
        **pause_updates,
        **set_current_step(
            state,
            WorkflowStep.REVIEW,
        ),
    }


async def resume_node(
    state: AgenticState,
) -> AgenticState:
    """
    Resume a previously paused workflow.
    """
    execution_id = state.get("execution_id")

    logger.info(
        "workflow_resumed",
        extra={
            "execution_id": execution_id,
        },
    )

    resume_updates = resume_workflow(
        state,
    )

    return {
        **resume_updates,
        **set_current_step(
            state,
            WorkflowStep.INGESTION,
        ),
    }


async def review_node(
    state: AgenticState,
) -> AgenticState:
    """
    Prepare workflow output for human review.
    """
    execution_id = state.get("execution_id")
    workflow_step = WorkflowStep.REVIEW

    interrupt_payload = (
        create_interrupt_payload(
            interrupt_type=(
                "approval_required"
            ),
            reason=(
                "Generated artifacts require human "
                "review before publication."
            ),
            execution_id=execution_id,
            workflow_step=workflow_step.value,
            metadata={
                "validation_outcome": state.get(
                    "validation_outcome",
                ),
                "artifact_count": len(
                    state.get("artifacts", []),
                ),
            },
        )
    )

    interrupt_updates = request_human_review(
        state,
        reason=interrupt_payload["reason"],
        interrupt_type=interrupt_payload[
            "interrupt_type"
        ],
        workflow_step=workflow_step.value,
        metadata=interrupt_payload.get(
            "metadata",
        ),
    )

    emit_human_review_requested(
        execution_id=execution_id,
        reason=interrupt_payload["reason"],
        workflow_step=workflow_step.value,
        tenant_id=state.get("tenant_id"),
        interrupt_id=interrupt_payload[
            "interrupt_id"
        ],
    )

    logger.info(
        "workflow_awaiting_approval",
        extra={
            "execution_id": execution_id,
            "interrupt_id": interrupt_payload[
                "interrupt_id"
            ],
        },
    )

    return {
        **interrupt_updates,
        **set_current_step(
            state,
            workflow_step,
        ),
        "current_interrupt": interrupt_payload,
    }


async def resolve_review_node(
    state: AgenticState,
) -> AgenticState:
    """
    Apply a human review decision and resume or terminate the workflow.
    """
    execution_id = state.get("execution_id")

    review_decision = state.get(
        "review_decision",
    )
    reviewer_id = state.get("reviewer_id")
    review_comment = state.get(
        "review_comment",
    )

    if not review_decision:
        raise WorkflowStateError(
            "A human review decision is required "
            "to resume the workflow.",
        )

    current_interrupt = state.get(
        "current_interrupt",
    )

    if not isinstance(
        current_interrupt,
        dict,
    ):
        raise WorkflowStateError(
            "Workflow does not have an active "
            "human-review interrupt.",
        )

    resolved_updates = (
        resolve_active_interrupt(
            state,
            decision=review_decision,
            reviewer_id=reviewer_id,
            comment=review_comment,
        )
    )

    emit_human_review_resolved(
        execution_id=execution_id,
        decision=review_decision,
        reviewer_id=reviewer_id,
        workflow_step=(
            WorkflowStep.REVIEW.value
        ),
        tenant_id=state.get("tenant_id"),
        interrupt_id=current_interrupt.get(
            "interrupt_id",
        ),
    )

    logger.info(
        "workflow_review_resolved",
        extra={
            "execution_id": execution_id,
            "decision": review_decision,
        },
    )

    return resolved_updates


async def retry_node(
    state: AgenticState,
) -> AgenticState:
    """
    Prepare a recoverable workflow retry.
    """
    execution_id = state.get("execution_id")

    try:
        retry_updates = prepare_retry(
            state,
            error_step=state.get(
                "error_step",
            ),
            error_code=state.get(
                "error_code",
            ),
            message=state.get(
                "current_error",
            ),
        )

    except Exception:
        logger.warning(
            "workflow_retry_preparation_failed",
            extra={
                "execution_id": execution_id,
                "error_code": state.get(
                    "error_code",
                ),
            },
        )

        return set_error(
            state,
            message=(
                "Workflow retry allowance has "
                "been exhausted."
            ),
            error_step=state.get(
                "error_step",
            ),
            error_code=(
                "retry_limit_exceeded"
            ),
            recoverable=False,
        )

    cleared_state = clear_error(
        {
            **state,
            **retry_updates,
        },
    )

    logger.info(
        "workflow_retry_prepared",
        extra={
            "execution_id": execution_id,
            "retry_count": cleared_state.get(
                "retry_count",
            ),
            "retry_step": state.get(
                "error_step",
            ),
        },
    )

    return {
        **cleared_state,
        "execution_logs": [
            (
                "WARNING: Recoverable workflow "
                "failure detected. Retry "
                f"{cleared_state.get('retry_count')} of "
                f"{cleared_state.get('max_retries')} prepared."
            ),
        ],
    }


async def error_handler_node(
    state: AgenticState,
) -> AgenticState:
    """
    Terminal-safe error handler.
    """
    execution_id = state.get("execution_id")
    brief_id = state.get("brief_id")

    error_message = state.get(
        "current_error",
    ) or (
        "Workflow execution failed due to an "
        "unexpected error."
    )

    error_code = state.get("error_code") or (
        "workflow_execution_error"
    )

    error_step = state.get("error_step")

    logger.error(
        "workflow_failed",
        extra={
            "execution_id": execution_id,
            "error_step": (
                error_step.value
                if isinstance(
                    error_step,
                    WorkflowStep,
                )
                else error_step
            ),
            "error_code": error_code,
        },
    )

    emit_workflow_failed(
        execution_id=execution_id,
        brief_id=brief_id,
        tenant_id=state.get("tenant_id"),
        user_id=state.get("user_id"),
        workflow_name=state.get(
            "workflow_name",
        ),
        error_code=error_code,
        error_step=(
            error_step.value
            if isinstance(
                error_step,
                WorkflowStep,
            )
            else error_step
        ),
        recoverable=bool(
            state.get(
                "recoverable_error",
                False,
            ),
        ),
    )

    failed_state = set_error(
        state,
        message=error_message,
        error_step=error_step,
        error_code=error_code,
        recoverable=bool(
            state.get(
                "recoverable_error",
                False,
            ),
        ),
    )

    return {
        **failed_state,
        **set_current_step(
            state,
            WorkflowStep.ERROR,
        ),
        "execution_logs": [
            "ERROR: Workflow execution failed safely.",
        ],
    }


async def complete_workflow_node(
    state: AgenticState,
) -> AgenticState:
    """
    Mark a successfully published workflow as complete.
    """
    execution_id = state.get("execution_id")
    brief_id = state.get("brief_id")

    logger.info(
        "workflow_completed",
        extra={
            "execution_id": execution_id,
            "brief_id": brief_id,
            "artifact_count": len(
                state.get("artifacts", []),
            ),
        },
    )

    emit_workflow_completed(
        execution_id=execution_id,
        brief_id=brief_id,
        tenant_id=state.get("tenant_id"),
        user_id=state.get("user_id"),
        workflow_name=state.get(
            "workflow_name",
        ),
        duration_ms=state.get(
            "duration_ms",
        ),
        token_usage=state.get(
            "provider_token_usage",
        ),
        estimated_cost_usd=state.get(
            "estimated_cost_usd",
        ),
    )

    return {
        **set_workflow_status(
            state,
            JobStatus.COMPLETED,
        ),
        **set_current_step(
            state,
            WorkflowStep.COMPLETED,
        ),
        "completed_at": state.get(
            "publishing_completed_at",
        ),
        "execution_logs": [
            "INFO: Workflow execution completed "
            "successfully.",
        ],
        "current_error": None,
        "error_step": None,
        "error_code": None,
        "recoverable_error": False,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_workflow(
    *,
    checkpointer: Any | None = None,
    interrupt_before_review: bool = True,
) -> Any:
    """
    Build and compile the AgenticBridge workflow graph.
    """
    workflow = StateGraph(AgenticState)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    workflow.add_node(
        "begin",
        begin_workflow_node,
    )
    workflow.add_node(
        "retrieve_context",
        retrieval_node,
    )
    workflow.add_node(
        "ingest",
        ingestion_node,
    )
    workflow.add_node(
        "extract_requirements",
        requirements_node,
    )
    workflow.add_node(
        "design_analysis",
        design_analysis_node,
    )
    workflow.add_node(
        "generate_code",
        code_generation_node,
    )
    workflow.add_node(
        "validate_output",
        validation_node,
    )
    workflow.add_node(
        "review",
        review_node,
    )
    workflow.add_node(
        "resolve_review",
        resolve_review_node,
    )
    workflow.add_node(
        "await_approval",
        review_node,
    )
    workflow.add_node(
        "publish",
        publishing_node,
    )
    workflow.add_node(
        "complete",
        complete_workflow_node,
    )
    workflow.add_node(
        "pause",
        pause_node,
    )
    workflow.add_node(
        "resume",
        resume_node,
    )
    workflow.add_node(
        "cancelled",
        cancellation_node,
    )
    workflow.add_node(
        "retry",
        retry_node,
    )
    workflow.add_node(
        "handle_error",
        error_handler_node,
    )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    workflow.set_entry_point("begin")

    workflow.add_conditional_edges(
        "begin",
        route_on_start,
        {
            ROUTE_RETRIEVE_CONTEXT: (
                "retrieve_context"
            ),
            ROUTE_HUMAN_REVIEW: "pause",
            ROUTE_CANCELLED: "cancelled",
            ROUTE_FAILED: "handle_error",
        },
    )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "retrieve_context",
        route_after_retrieval,
        {
            ROUTE_EXTRACT_REQUIREMENTS: (
                "extract_requirements"
            ),
            ROUTE_RETRY: "retry",
            ROUTE_CANCELLED: "cancelled",
            ROUTE_FAILED: "handle_error",
        },
    )

    # ------------------------------------------------------------------
    # Ingestion compatibility edge
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "ingest",
        route_after_ingestion,
        {
            "extract_requirements": (
                "extract_requirements"
            ),
            "handle_error": "handle_error",
        },
    )

    # ------------------------------------------------------------------
    # Requirements
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "extract_requirements",
        route_after_requirements,
        {
            ROUTE_GENERATE_DESIGN: (
                "design_analysis"
            ),
            ROUTE_HUMAN_REVIEW: "review",
            ROUTE_RETRY: "retry",
            ROUTE_CANCELLED: "cancelled",
            ROUTE_FAILED: "handle_error",
        },
    )

    # ------------------------------------------------------------------
    # Design
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "design_analysis",
        route_after_design,
        {
            ROUTE_GENERATE_CODE: (
                "generate_code"
            ),
            ROUTE_HUMAN_REVIEW: "review",
            ROUTE_RETRY: "retry",
            ROUTE_CANCELLED: "cancelled",
            ROUTE_FAILED: "handle_error",
        },
    )

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "generate_code",
        route_after_generation,
        {
            ROUTE_VALIDATE_OUTPUT: (
                "validate_output"
            ),
            ROUTE_HUMAN_REVIEW: "review",
            ROUTE_RETRY: "retry",
            ROUTE_CANCELLED: "cancelled",
            ROUTE_FAILED: "handle_error",
        },
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "validate_output",
        route_after_validation,
        {
            ROUTE_HUMAN_REVIEW: "review",
            ROUTE_RETRY: "retry",
            ROUTE_CANCELLED: "cancelled",
            ROUTE_FAILED: "handle_error",
        },
    )

    # ------------------------------------------------------------------
    # Review and approval
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "review",
        route_after_review,
        {
            ROUTE_AWAITING_REVIEW: (
                "await_approval"
            ),
            ROUTE_PUBLISH_ARTIFACTS: "publish",
            ROUTE_RETRY: "retry",
            ROUTE_CANCELLED: "cancelled",
            ROUTE_FAILED: "handle_error",
        },
    )

    workflow.add_conditional_edges(
        "await_approval",
        route_on_interrupt,
        {
            ROUTE_AWAITING_REVIEW: (
                "await_approval"
            ),
            ROUTE_PUBLISH_ARTIFACTS: "publish",
            ROUTE_RETRY: "retry",
            ROUTE_CANCELLED: "cancelled",
            ROUTE_FAILED: "handle_error",
        },
    )

    workflow.add_conditional_edges(
        "resolve_review",
        route_after_review,
        {
            ROUTE_AWAITING_REVIEW: (
                "await_approval"
            ),
            ROUTE_PUBLISH_ARTIFACTS: "publish",
            ROUTE_RETRY: "retry",
            ROUTE_CANCELLED: "cancelled",
            ROUTE_FAILED: "handle_error",
        },
    )

    # ------------------------------------------------------------------
    # Publication
    # ------------------------------------------------------------------

    workflow.add_conditional_edges(
        "publish",
        route_after_publication,
        {
            ROUTE_COMPLETED: "complete",
            ROUTE_RETRY: "retry",
            ROUTE_CANCELLED: "cancelled",
            ROUTE_FAILED: "handle_error",
        },
    )

    # ------------------------------------------------------------------
    # Terminal edges
    # ------------------------------------------------------------------

    workflow.add_edge(
        "complete",
        END,
    )
    workflow.add_edge(
        "cancelled",
        END,
    )
    workflow.add_edge(
        "handle_error",
        END,
    )
    workflow.add_edge(
        "pause",
        END,
    )

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    compile_kwargs: dict[str, Any] = {}

    if checkpointer is not None:
        compile_kwargs["checkpointer"] = (
            checkpointer
        )

    if interrupt_before_review:
        compile_kwargs["interrupt_before"] = [
            "review",
        ]

    try:
        return workflow.compile(
            **compile_kwargs,
        )

    except Exception as exc:
        logger.exception(
            "workflow_graph_build_failed",
        )

        raise WorkflowStateError(
            "The workflow graph could not be built.",
        ) from exc


# ---------------------------------------------------------------------------
# Default workflow instance
# ---------------------------------------------------------------------------


def create_default_checkpointer() -> Any | None:
    """
    Create the configured default workflow checkpointer.

    Returns `None` when persistence is unavailable or disabled.
    """
    try:
        return create_checkpointer()

    except Exception:
        logger.warning(
            "workflow_checkpointer_unavailable",
            exc_info=True,
        )

        return None


workflow_app = build_workflow(
    checkpointer=create_default_checkpointer(),
    interrupt_before_review=True,
)


__all__ = [
    "WORKFLOW_NAME",
    "WORKFLOW_VERSION",
    "build_workflow",
    "create_default_checkpointer",
    "route_after_ingestion",
    "workflow_app",
]