"""
LangGraph conditional routing for the AgenticBridge workflow.

This module maps workflow state to explicit route names used by conditional
edges. Routing functions are pure, synchronous, exception-safe, and free of
side effects.

Security and reliability rules:
- Treat workflow state as untrusted input.
- Never mutate state, call providers, access databases, or execute generated
  code.
- Never log prompts, completions, generated code, document contents,
  embeddings, credentials, or customer PII.
- Always return a known route; unknown or invalid state must fail safe.
- Keep route names stable because workflow edges and tests depend on them.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from app.core.enums import JobStatus, WorkflowStep
from app.graph.recovery import decide_recovery_action


logger = logging.getLogger("agentic_bridge.graph.routing")

RouteName = Literal[
    "retrieve_context",
    "extract_requirements",
    "generate_design",
    "generate_code",
    "validate_output",
    "human_review",
    "publish_artifacts",
    "retry",
    "failed",
    "completed",
    "cancelled",
    "awaiting_review",
    "unknown",
]

ROUTE_RETRIEVE_CONTEXT: RouteName = "retrieve_context"
ROUTE_EXTRACT_REQUIREMENTS: RouteName = "extract_requirements"
ROUTE_GENERATE_DESIGN: RouteName = "generate_design"
ROUTE_GENERATE_CODE: RouteName = "generate_code"
ROUTE_VALIDATE_OUTPUT: RouteName = "validate_output"
ROUTE_HUMAN_REVIEW: RouteName = "human_review"
ROUTE_PUBLISH_ARTIFACTS: RouteName = "publish_artifacts"
ROUTE_RETRY: RouteName = "retry"
ROUTE_FAILED: RouteName = "failed"
ROUTE_COMPLETED: RouteName = "completed"
ROUTE_CANCELLED: RouteName = "cancelled"
ROUTE_AWAITING_REVIEW: RouteName = "awaiting_review"
ROUTE_UNKNOWN: RouteName = "unknown"

MAX_ROUTE_REASON_CHARS = 500


# ---------------------------------------------------------------------------
# Safe state helpers
# ---------------------------------------------------------------------------


def _safe_get(
    state: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Safely read a workflow-state value."""
    if not isinstance(state, dict):
        return default

    return state.get(key, default)


def _is_truthy(value: Any) -> bool:
    """Return whether a state value should be treated as true."""
    if value is None:
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    if isinstance(value, (int, float)):
        return value > 0

    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0

    return bool(value)


def _normalize_step(
    value: Any,
) -> str | None:
    """Normalize a workflow-step value."""
    if value is None:
        return None

    if isinstance(value, WorkflowStep):
        return value.value

    if isinstance(value, str):
        normalized_value = value.strip().lower()

        return normalized_value or None

    return None


def _normalize_status(
    value: Any,
) -> str | None:
    """Normalize a status-like value."""
    if value is None:
        return None

    if isinstance(value, JobStatus):
        return value.value

    if isinstance(value, str):
        normalized_value = value.strip().lower()

        return normalized_value or None

    return None


def _log_route(
    *,
    execution_id: str | None,
    route: RouteName,
    reason: str,
) -> None:
    """Log a bounded, non-sensitive routing decision."""
    logger.info(
        "workflow_route_selected",
        execution_id=execution_id,
        route=route,
        reason=reason[:MAX_ROUTE_REASON_CHARS],
    )


# ---------------------------------------------------------------------------
# Shared routing predicates
# ---------------------------------------------------------------------------


def is_cancelled(
    state: dict[str, Any],
) -> bool:
    """Return whether workflow cancellation was requested."""
    return _is_truthy(_safe_get(state, "cancel_requested"))


def is_paused(
    state: dict[str, Any],
) -> bool:
    """Return whether workflow pause was requested."""
    return _is_truthy(_safe_get(state, "pause_requested"))


def has_active_interrupt(
    state: dict[str, Any],
) -> bool:
    """Return whether the workflow has an active human-review interrupt."""
    current_interrupt = _safe_get(
        state,
        "current_interrupt",
    )

    if not isinstance(current_interrupt, dict):
        return False

    return current_interrupt.get("status") == "pending_review"


def has_blocking_error(
    state: dict[str, Any],
) -> bool:
    """Return whether the workflow has a non-recoverable error."""
    current_error = _safe_get(state, "current_error")

    if not current_error:
        return False

    return not _is_truthy(
        _safe_get(state, "recoverable_error")
    )


def has_recoverable_error(
    state: dict[str, Any],
) -> bool:
    """Return whether the workflow has a recoverable error."""
    return (
        bool(_safe_get(state, "current_error"))
        and _is_truthy(_safe_get(state, "recoverable_error"))
    )


def has_requirements(
    state: dict[str, Any],
) -> bool:
    """Return whether usable requirements exist."""
    requirements = _safe_get(
        state,
        "extracted_requirements",
        [],
    )

    return isinstance(requirements, list) and len(requirements) > 0


def has_design_context(
    state: dict[str, Any],
) -> bool:
    """Return whether design context exists."""
    design_tokens = _safe_get(
        state,
        "design_tokens",
        {},
    )

    return isinstance(design_tokens, dict) and len(design_tokens) > 0


def has_generated_files(
    state: dict[str, Any],
) -> bool:
    """Return whether generated code files exist."""
    generated_files = _safe_get(
        state,
        "generated_code_files",
        {},
    )

    return (
        isinstance(generated_files, dict)
        and len(generated_files) > 0
    )


def is_validation_passed(
    state: dict[str, Any],
) -> bool:
    """Return whether validation passed or completed with warnings."""
    validation_outcome = _normalize_status(
        _safe_get(state, "validation_outcome")
    )

    return validation_outcome in {
        "passed",
        "warning",
    }


def is_validation_failed(
    state: dict[str, Any],
) -> bool:
    """Return whether validation failed."""
    validation_outcome = _normalize_status(
        _safe_get(state, "validation_outcome")
    )

    validation_status = _normalize_status(
        _safe_get(state, "validation_status")
    )

    return (
        validation_outcome == "failed"
        or validation_status == JobStatus.FAILED.value
    )


def is_approved(
    state: dict[str, Any],
) -> bool:
    """Return whether a human reviewer approved the workflow."""
    review_status = _normalize_status(
        _safe_get(state, "review_status")
    )

    approval_status = _normalize_status(
        _safe_get(state, "approval_status")
    )

    return review_status == "approved" or approval_status == "approved"


def is_rejected(
    state: dict[str, Any],
) -> bool:
    """Return whether a human reviewer rejected the workflow."""
    review_status = _normalize_status(
        _safe_get(state, "review_status")
    )

    approval_status = _normalize_status(
        _safe_get(state, "approval_status")
    )

    return review_status == "rejected" or approval_status == "rejected"


def requires_human_approval(
    state: dict[str, Any],
) -> bool:
    """Return whether the workflow is waiting for human approval."""
    approval_required = _is_truthy(
        _safe_get(state, "approval_required")
    )

    review_status = _normalize_status(
        _safe_get(state, "review_status")
    )

    return approval_required or review_status == "pending_approval"


# ---------------------------------------------------------------------------
# Entry routing
# ---------------------------------------------------------------------------


def route_on_start(
    state: dict[str, Any],
) -> RouteName:
    """
    Route at workflow start.

    Args:
        state: Current workflow state.

    Returns:
        `retrieve_context` when the workflow can start, `cancelled` when
        cancellation was requested, or `failed` for invalid state.
    """
    execution_id = _safe_get(state, "execution_id")

    try:
        if is_cancelled(state):
            route: RouteName = ROUTE_CANCELLED
            reason = "Workflow cancellation requested before start."
        elif not _safe_get(state, "execution_id"):
            route = ROUTE_FAILED
            reason = "Workflow execution ID is missing."
        elif not _safe_get(state, "brief_id"):
            route = ROUTE_FAILED
            reason = "Workflow brief ID is missing."
        elif not _safe_get(state, "raw_brief_text"):
            route = ROUTE_FAILED
            reason = "Workflow source brief is missing."
        else:
            route = ROUTE_RETRIEVE_CONTEXT
            reason = "Workflow can begin context retrieval."

        _log_route(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_start_routing_failed",
            execution_id=execution_id,
        )

        return ROUTE_FAILED


# ---------------------------------------------------------------------------
# Node-exit routing
# ---------------------------------------------------------------------------


def route_after_retrieval(
    state: dict[str, Any],
) -> RouteName:
    """
    Route after context retrieval.

    Args:
        state: Current workflow state.

    Returns:
        `extract_requirements`, `retry`, `failed`, or `cancelled`.
    """
    execution_id = _safe_get(state, "execution_id")

    try:
        if is_cancelled(state):
            route: RouteName = ROUTE_CANCELLED
            reason = "Workflow cancellation requested."
        elif has_blocking_error(state):
            route = ROUTE_FAILED
            reason = "Retrieval failed permanently."
        elif has_recoverable_error(state):
            route = ROUTE_RETRY
            reason = "Retrieval failure is recoverable."
        else:
            route = ROUTE_EXTRACT_REQUIREMENTS
            reason = "Retrieval completed."

        _log_route(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_retrieval_routing_failed",
            execution_id=execution_id,
        )

        return ROUTE_FAILED


def route_after_requirements(
    state: dict[str, Any],
) -> RouteName:
    """
    Route after requirements extraction.

    Args:
        state: Current workflow state.

    Returns:
        `generate_design`, `retry`, `human_review`, `failed`, or `cancelled`.
    """
    execution_id = _safe_get(state, "execution_id")

    try:
        if is_cancelled(state):
            route: RouteName = ROUTE_CANCELLED
            reason = "Workflow cancellation requested."
        elif has_blocking_error(state):
            route = ROUTE_FAILED
            reason = "Requirements extraction failed permanently."
        elif has_recoverable_error(state):
            route = ROUTE_RETRY
            reason = "Requirements failure is recoverable."
        elif not has_requirements(state):
            route = ROUTE_HUMAN_REVIEW
            reason = "No usable requirements were extracted."
        else:
            route = ROUTE_GENERATE_DESIGN
            reason = "Requirements extraction completed."

        _log_route(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_requirements_routing_failed",
            execution_id=execution_id,
        )

        return ROUTE_FAILED


def route_after_design(
    state: dict[str, Any],
) -> RouteName:
    """
    Route after design generation.

    Args:
        state: Current workflow state.

    Returns:
        `generate_code`, `retry`, `human_review`, `failed`, or `cancelled`.
    """
    execution_id = _safe_get(state, "execution_id")

    try:
        if is_cancelled(state):
            route: RouteName = ROUTE_CANCELLED
            reason = "Workflow cancellation requested."
        elif has_blocking_error(state):
            route = ROUTE_FAILED
            reason = "Design generation failed permanently."
        elif has_recoverable_error(state):
            route = ROUTE_RETRY
            reason = "Design failure is recoverable."
        elif not has_design_context(state):
            route = ROUTE_HUMAN_REVIEW
            reason = "No usable design context was produced."
        else:
            route = ROUTE_GENERATE_CODE
            reason = "Design generation completed."

        _log_route(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_design_routing_failed",
            execution_id=execution_id,
        )

        return ROUTE_FAILED


def route_after_generation(
    state: dict[str, Any],
) -> RouteName:
    """
    Route after code generation.

    Args:
        state: Current workflow state.

    Returns:
        `validate_output`, `retry`, `human_review`, `failed`, or `cancelled`.
    """
    execution_id = _safe_get(state, "execution_id")

    try:
        if is_cancelled(state):
            route: RouteName = ROUTE_CANCELLED
            reason = "Workflow cancellation requested."
        elif has_blocking_error(state):
            route = ROUTE_FAILED
            reason = "Code generation failed permanently."
        elif has_recoverable_error(state):
            route = ROUTE_RETRY
            reason = "Code-generation failure is recoverable."
        elif not has_generated_files(state):
            route = ROUTE_HUMAN_REVIEW
            reason = "No generated artifacts were produced."
        else:
            route = ROUTE_VALIDATE_OUTPUT
            reason = "Code generation completed."

        _log_route(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_generation_routing_failed",
            execution_id=execution_id,
        )

        return ROUTE_FAILED


def route_after_validation(
    state: dict[str, Any],
) -> RouteName:
    """
    Route after output validation.

    Args:
        state: Current workflow state.

    Returns:
        `human_review`, `retry`, `failed`, or `cancelled`.
    """
    execution_id = _safe_get(state, "execution_id")

    try:
        if is_cancelled(state):
            route: RouteName = ROUTE_CANCELLED
            reason = "Workflow cancellation requested."
        elif has_blocking_error(state):
            route = ROUTE_FAILED
            reason = "Validation failed permanently."
        elif is_validation_failed(state):
            recovery_action = decide_recovery_action(
                state,
                error_code=_safe_get(state, "error_code"),
            )

            if recovery_action == "retry":
                route = ROUTE_RETRY
                reason = "Validation failure is recoverable."
            elif recovery_action == "human_review":
                route = ROUTE_HUMAN_REVIEW
                reason = (
                    "Validation failure requires human review."
                )
            else:
                route = ROUTE_FAILED
                reason = "Validation failure cannot be recovered."
        elif is_validation_passed(state):
            route = ROUTE_HUMAN_REVIEW
            reason = "Validation completed; approval is required."
        else:
            route = ROUTE_HUMAN_REVIEW
            reason = "Validation outcome is inconclusive."

        _log_route(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_validation_routing_failed",
            execution_id=execution_id,
        )

        return ROUTE_FAILED


def route_after_review(
    state: dict[str, Any],
) -> RouteName:
    """
    Route after human review.

    Args:
        state: Current workflow state.

    Returns:
        `publish_artifacts`, `retry`, `failed`, `awaiting_review`, or
        `cancelled`.
    """
    execution_id = _safe_get(state, "execution_id")

    try:
        if is_cancelled(state):
            route: RouteName = ROUTE_CANCELLED
            reason = "Workflow cancellation requested."
        elif is_approved(state):
            route = ROUTE_PUBLISH_ARTIFACTS
            reason = "Workflow approved for publication."
        elif is_rejected(state):
            route = ROUTE_FAILED
            reason = "Workflow rejected by human reviewer."
        elif requires_human_approval(state) or has_active_interrupt(state):
            route = ROUTE_AWAITING_REVIEW
            reason = "Workflow is awaiting human review."
        else:
            route = ROUTE_RETRY
            reason = "Reviewer requested changes."

        _log_route(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_review_routing_failed",
            execution_id=execution_id,
        )

        return ROUTE_FAILED


def route_after_publication(
    state: dict[str, Any],
) -> RouteName:
    """
    Route after artifact publication.

    Args:
        state: Current workflow state.

    Returns:
        `completed`, `retry`, `failed`, or `cancelled`.
    """
    execution_id = _safe_get(state, "execution_id")

    try:
        publishing_status = _normalize_status(
            _safe_get(state, "publishing_status")
        )

        if is_cancelled(state):
            route: RouteName = ROUTE_CANCELLED
            reason = "Workflow cancellation requested."
        elif has_blocking_error(state):
            route = ROUTE_FAILED
            reason = "Publication failed permanently."
        elif has_recoverable_error(state):
            route = ROUTE_RETRY
            reason = "Publication failure is recoverable."
        elif publishing_status == JobStatus.COMPLETED.value:
            route = ROUTE_COMPLETED
            reason = "Workflow publication completed."
        else:
            route = ROUTE_RETRY
            reason = "Publication has not completed."

        _log_route(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_publication_routing_failed",
            execution_id=execution_id,
        )

        return ROUTE_FAILED


# ---------------------------------------------------------------------------
# Error and interrupt routing
# ---------------------------------------------------------------------------


def route_on_error(
    state: dict[str, Any],
) -> RouteName:
    """
    Route after a workflow error.

    Args:
        state: Current workflow state.

    Returns:
        `retry`, `human_review`, `failed`, or `cancelled`.
    """
    execution_id = _safe_get(state, "execution_id")

    try:
        recovery_action = decide_recovery_action(
            state,
            error_code=_safe_get(state, "error_code"),
        )

        if recovery_action == "retry":
            route: RouteName = ROUTE_RETRY
            reason = "Recoverable workflow error can be retried."
        elif recovery_action == "human_review":
            route = ROUTE_HUMAN_REVIEW
            reason = "Workflow error requires human review."
        elif recovery_action == "cancel":
            route = ROUTE_CANCELLED
            reason = "Workflow cancellation was requested."
        else:
            route = ROUTE_FAILED
            reason = "Workflow error cannot be recovered."

        _log_route(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_error_routing_failed",
            execution_id=execution_id,
        )

        return ROUTE_FAILED


def route_on_interrupt(
    state: dict[str, Any],
) -> RouteName:
    """
    Route when a workflow interrupt is active.

    Args:
        state: Current workflow state.

    Returns:
        `awaiting_review` when a review is pending, `retry` after requested
        changes, `failed` after rejection, or `cancelled` when cancelled.
    """
    execution_id = _safe_get(state, "execution_id")

    try:
        if is_cancelled(state):
            route: RouteName = ROUTE_CANCELLED
            reason = "Workflow cancellation requested."
        elif has_active_interrupt(state) or requires_human_approval(state):
            route = ROUTE_AWAITING_REVIEW
            reason = "Workflow is awaiting human review."
        elif is_approved(state):
            route = ROUTE_PUBLISH_ARTIFACTS
            reason = "Workflow approved after review."
        elif is_rejected(state):
            route = ROUTE_FAILED
            reason = "Workflow rejected after review."
        else:
            route = ROUTE_RETRY
            reason = "Reviewer requested workflow changes."

        _log_route(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_interrupt_routing_failed",
            execution_id=execution_id,
        )

        return ROUTE_FAILED


# ---------------------------------------------------------------------------
# Route registry
# ---------------------------------------------------------------------------


def get_workflow_routes() -> dict[str, str]:
    """
    Return the canonical workflow route registry.

    Returns:
        A mapping of route constant names to route values.
    """
    return {
        "RETRIEVE_CONTEXT": ROUTE_RETRIEVE_CONTEXT,
        "EXTRACT_REQUIREMENTS": ROUTE_EXTRACT_REQUIREMENTS,
        "GENERATE_DESIGN": ROUTE_GENERATE_DESIGN,
        "GENERATE_CODE": ROUTE_GENERATE_CODE,
        "VALIDATE_OUTPUT": ROUTE_VALIDATE_OUTPUT,
        "HUMAN_REVIEW": ROUTE_HUMAN_REVIEW,
        "PUBLISH_ARTIFACTS": ROUTE_PUBLISH_ARTIFACTS,
        "RETRY": ROUTE_RETRY,
        "FAILED": ROUTE_FAILED,
        "COMPLETED": ROUTE_COMPLETED,
        "CANCELLED": ROUTE_CANCELLED,
        "AWAITING_REVIEW": ROUTE_AWAITING_REVIEW,
        "UNKNOWN": ROUTE_UNKNOWN,
    }