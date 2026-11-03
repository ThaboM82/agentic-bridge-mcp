"""
LangGraph workflow guards and routing conditions.

This module contains pure, deterministic guard functions used by conditional
edges. Guards inspect workflow state and return routing decisions; they do not
mutate state, invoke providers, access databases, or implement business policy
beyond workflow routing.

Security and reliability rules:
- Treat workflow state as untrusted input.
- Never raise from a guard; return a safe fallback route instead.
- Never log prompt text, generated code, document contents, embeddings, or
  credentials.
- Keep guards synchronous, fast, and side-effect free.
- Use explicit route constants so conditional edges remain auditable.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from app.core.enums import (
    JobStatus,
    WorkflowStep,
)


logger = logging.getLogger(
    "agentic_bridge.graph.guards",
)


RouteName = Literal[
    "continue",
    "retry",
    "human_review",
    "failed",
    "end",
    "unknown",
]

MAX_ROUTE_REASON_CHARS = 500


# ---------------------------------------------------------------------------
# Route constants
# ---------------------------------------------------------------------------

ROUTE_CONTINUE: RouteName = "continue"
ROUTE_RETRY: RouteName = "retry"
ROUTE_HUMAN_REVIEW: RouteName = (
    "human_review"
)
ROUTE_FAILED: RouteName = "failed"
ROUTE_END: RouteName = "end"
ROUTE_UNKNOWN: RouteName = "unknown"


# ---------------------------------------------------------------------------
# Safe state access
# ---------------------------------------------------------------------------


def _safe_get(
    state: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Safely read a workflow-state value."""
    if not isinstance(state, dict):
        return default

    return state.get(
        key,
        default,
    )


def _is_truthy(
    value: Any,
) -> bool:
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

    if isinstance(
        value,
        (list, tuple, set, dict),
    ):
        return len(value) > 0

    return bool(value)


def _normalize_status(
    value: Any,
) -> str | None:
    """Normalize a status-like state value."""
    if value is None:
        return None

    if isinstance(value, JobStatus):
        return value.value

    if isinstance(value, str):
        normalized_value = (
            value.strip().lower()
        )

        return normalized_value or None

    return None


def _normalize_step(
    value: Any,
) -> str | None:
    """Normalize a workflow-step state value."""
    if value is None:
        return None

    if isinstance(value, WorkflowStep):
        return value.value

    if isinstance(value, str):
        normalized_value = (
            value.strip().lower()
        )

        return normalized_value or None

    return None


def _log_route_decision(
    *,
    execution_id: str | None,
    route: RouteName,
    reason: str,
) -> None:
    """Log a bounded, non-sensitive routing decision."""
    logger.info(
        "workflow_route_selected",
        extra={
            "execution_id": execution_id,
            "route": route,
            "reason": reason[
                :MAX_ROUTE_REASON_CHARS
            ],
        },
    )


# ---------------------------------------------------------------------------
# Core guards
# ---------------------------------------------------------------------------


def should_retry(
    state: dict[str, Any],
) -> bool:
    """
    Return whether the current failed step is recoverable and retries remain.
    """
    recoverable_error = _is_truthy(
        _safe_get(
            state,
            "recoverable_error",
        ),
    )

    retry_count = _safe_get(
        state,
        "retry_count",
        0,
    )

    if not isinstance(
        retry_count,
        int,
    ) or isinstance(
        retry_count,
        bool,
    ):
        retry_count = 0

    max_retries = _safe_get(
        state,
        "max_retries",
        3,
    )

    if not isinstance(
        max_retries,
        int,
    ) or isinstance(
        max_retries,
        bool,
    ):
        max_retries = 3

    return (
        recoverable_error
        and retry_count < max_retries
    )


def has_blocking_error(
    state: dict[str, Any],
) -> bool:
    """
    Return whether the workflow has a non-recoverable error.
    """
    current_error = _safe_get(
        state,
        "current_error",
    )

    if not current_error:
        return False

    recoverable_error = _is_truthy(
        _safe_get(
            state,
            "recoverable_error",
        ),
    )

    return not recoverable_error


def requires_human_approval(
    state: dict[str, Any],
) -> bool:
    """
    Return whether workflow output requires human approval.
    """
    approval_required = _is_truthy(
        _safe_get(
            state,
            "approval_required",
        ),
    )

    review_status = _normalize_status(
        _safe_get(
            state,
            "review_status",
        ),
    )

    return (
        approval_required
        or review_status
        == "pending_approval"
    )


def is_validation_passed(
    state: dict[str, Any],
) -> bool:
    """
    Return whether validation completed without blocking errors.
    """
    validation_outcome = _normalize_status(
        _safe_get(
            state,
            "validation_outcome",
        ),
    )

    validation_status = _normalize_status(
        _safe_get(
            state,
            "validation_status",
        ),
    )

    return (
        validation_outcome
        in {
            "passed",
            "warning",
        }
        or validation_status
        == JobStatus.COMPLETED.value
    )


def is_validation_failed(
    state: dict[str, Any],
) -> bool:
    """
    Return whether validation failed.
    """
    validation_outcome = _normalize_status(
        _safe_get(
            state,
            "validation_outcome",
        ),
    )

    validation_status = _normalize_status(
        _safe_get(
            state,
            "validation_status",
        ),
    )

    return (
        validation_outcome == "failed"
        or validation_status
        == JobStatus.FAILED.value
    )


def has_generated_artifacts(
    state: dict[str, Any],
) -> bool:
    """
    Return whether the workflow produced generated artifacts.
    """
    generated_files = _safe_get(
        state,
        "generated_code_files",
        {},
    )

    if not isinstance(
        generated_files,
        dict,
    ):
        return False

    return len(generated_files) > 0


def has_retrieval_context(
    state: dict[str, Any],
) -> bool:
    """
    Return whether the workflow retrieved usable context.
    """
    retrieval_matches = _safe_get(
        state,
        "retrieval_matches",
        [],
    )

    if not isinstance(
        retrieval_matches,
        list,
    ):
        return False

    return len(retrieval_matches) > 0


def has_requirements(
    state: dict[str, Any],
) -> bool:
    """
    Return whether the workflow extracted usable requirements.
    """
    requirements = _safe_get(
        state,
        "extracted_requirements",
        [],
    )

    if not isinstance(
        requirements,
        list,
    ):
        return False

    return len(requirements) > 0


def is_workflow_cancelled(
    state: dict[str, Any],
) -> bool:
    """
    Return whether the workflow was cancelled by a user or operator.
    """
    return _is_truthy(
        _safe_get(
            state,
            "cancel_requested",
        ),
    )


def is_workflow_paused(
    state: dict[str, Any],
) -> bool:
    """
    Return whether the workflow is explicitly paused.
    """
    return _is_truthy(
        _safe_get(
            state,
            "pause_requested",
        ),
    )


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def route_after_retrieval(
    state: dict[str, Any],
) -> RouteName:
    """
    Route after the retrieval node.
    """
    execution_id = _safe_get(
        state,
        "execution_id",
    )

    try:
        if is_workflow_cancelled(state):
            route: RouteName = ROUTE_END
            reason = (
                "Workflow cancellation requested."
            )

        elif has_blocking_error(state):
            route = ROUTE_FAILED
            reason = (
                "Retrieval failed with a "
                "non-recoverable error."
            )

        elif should_retry(state):
            route = ROUTE_RETRY
            reason = (
                "Retrieval failed with a "
                "recoverable error."
            )

        else:
            route = ROUTE_CONTINUE
            reason = (
                "Retrieval completed."
            )

        _log_route_decision(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_route_after_retrieval_failed",
            extra={
                "execution_id": execution_id,
            },
        )

        return ROUTE_FAILED


def route_after_requirements(
    state: dict[str, Any],
) -> RouteName:
    """
    Route after the requirements node.
    """
    execution_id = _safe_get(
        state,
        "execution_id",
    )

    try:
        if is_workflow_cancelled(state):
            route: RouteName = ROUTE_END
            reason = (
                "Workflow cancellation requested."
            )

        elif has_blocking_error(state):
            route = ROUTE_FAILED
            reason = (
                "Requirements extraction failed "
                "permanently."
            )

        elif should_retry(state):
            route = ROUTE_RETRY
            reason = (
                "Requirements extraction can be "
                "retried."
            )

        elif not has_requirements(state):
            route = ROUTE_HUMAN_REVIEW
            reason = (
                "No usable requirements were "
                "extracted."
            )

        else:
            route = ROUTE_CONTINUE
            reason = (
                "Requirements extraction completed."
            )

        _log_route_decision(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_route_after_requirements_failed",
            extra={
                "execution_id": execution_id,
            },
        )

        return ROUTE_FAILED


def route_after_design(
    state: dict[str, Any],
) -> RouteName:
    """
    Route after the design node.
    """
    execution_id = _safe_get(
        state,
        "execution_id",
    )

    try:
        if is_workflow_cancelled(state):
            route: RouteName = ROUTE_END
            reason = (
                "Workflow cancellation requested."
            )

        elif has_blocking_error(state):
            route = ROUTE_FAILED
            reason = (
                "Design generation failed "
                "permanently."
            )

        elif should_retry(state):
            route = ROUTE_RETRY
            reason = (
                "Design generation can be retried."
            )

        else:
            route = ROUTE_CONTINUE
            reason = (
                "Design generation completed."
            )

        _log_route_decision(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_route_after_design_failed",
            extra={
                "execution_id": execution_id,
            },
        )

        return ROUTE_FAILED


def route_after_generation(
    state: dict[str, Any],
) -> RouteName:
    """
    Route after the code-generation node.
    """
    execution_id = _safe_get(
        state,
        "execution_id",
    )

    try:
        if is_workflow_cancelled(state):
            route: RouteName = ROUTE_END
            reason = (
                "Workflow cancellation requested."
            )

        elif has_blocking_error(state):
            route = ROUTE_FAILED
            reason = (
                "Code generation failed "
                "permanently."
            )

        elif should_retry(state):
            route = ROUTE_RETRY
            reason = (
                "Code generation can be retried."
            )

        elif not has_generated_artifacts(
            state,
        ):
            route = ROUTE_HUMAN_REVIEW
            reason = (
                "No generated artifacts were "
                "produced."
            )

        else:
            route = ROUTE_CONTINUE
            reason = (
                "Code generation completed."
            )

        _log_route_decision(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_route_after_generation_failed",
            extra={
                "execution_id": execution_id,
            },
        )

        return ROUTE_FAILED


def route_after_validation(
    state: dict[str, Any],
) -> RouteName:
    """
    Route after the validation node.
    """
    execution_id = _safe_get(
        state,
        "execution_id",
    )

    try:
        if is_workflow_cancelled(state):
            route: RouteName = ROUTE_END
            reason = (
                "Workflow cancellation requested."
            )

        elif has_blocking_error(state):
            route = ROUTE_FAILED
            reason = (
                "Validation failed with a "
                "non-recoverable error."
            )

        elif is_validation_failed(state):
            if should_retry(state):
                route = ROUTE_RETRY
                reason = (
                    "Validation failed but is "
                    "recoverable."
                )

            else:
                route = ROUTE_HUMAN_REVIEW
                reason = (
                    "Validation failed and retry "
                    "attempts are exhausted."
                )

        elif is_validation_passed(state):
            route = ROUTE_HUMAN_REVIEW
            reason = (
                "Validation completed; human "
                "approval is required."
            )

        else:
            route = ROUTE_HUMAN_REVIEW
            reason = (
                "Validation outcome is inconclusive."
            )

        _log_route_decision(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_route_after_validation_failed",
            extra={
                "execution_id": execution_id,
            },
        )

        return ROUTE_FAILED


def route_after_review(
    state: dict[str, Any],
) -> RouteName:
    """
    Route after human review.
    """
    execution_id = _safe_get(
        state,
        "execution_id",
    )

    try:
        review_status = _normalize_status(
            _safe_get(
                state,
                "review_status",
            ),
        )

        if review_status == "approved":
            route: RouteName = (
                ROUTE_CONTINUE
            )
            reason = (
                "Workflow approved by human reviewer."
            )

        elif (
            review_status
            == "changes_requested"
        ):
            route = ROUTE_RETRY
            reason = (
                "Human reviewer requested changes."
            )

        elif review_status == "rejected":
            route = ROUTE_FAILED
            reason = (
                "Workflow rejected by human reviewer."
            )

        else:
            route = ROUTE_HUMAN_REVIEW
            reason = (
                "Workflow is still awaiting human "
                "review."
            )

        _log_route_decision(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_route_after_review_failed",
            extra={
                "execution_id": execution_id,
            },
        )

        return ROUTE_FAILED


def route_after_publication(
    state: dict[str, Any],
) -> RouteName:
    """
    Route after artifact publication.
    """
    execution_id = _safe_get(
        state,
        "execution_id",
    )

    try:
        if has_blocking_error(state):
            route: RouteName = ROUTE_FAILED
            reason = (
                "Publication failed permanently."
            )

        elif should_retry(state):
            route = ROUTE_RETRY
            reason = (
                "Publication can be retried."
            )

        else:
            route = ROUTE_END
            reason = (
                "Workflow publication completed."
            )

        _log_route_decision(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_route_after_publication_failed",
            extra={
                "execution_id": execution_id,
            },
        )

        return ROUTE_FAILED


def route_on_workflow_error(
    state: dict[str, Any],
) -> RouteName:
    """
    Route when a workflow node raises or records an error.
    """
    execution_id = _safe_get(
        state,
        "execution_id",
    )

    try:
        if should_retry(state):
            route: RouteName = ROUTE_RETRY
            reason = (
                "Recoverable workflow error can "
                "be retried."
            )

        elif has_blocking_error(state):
            route = ROUTE_FAILED
            reason = (
                "Non-recoverable workflow error "
                "occurred."
            )

        else:
            route = ROUTE_HUMAN_REVIEW
            reason = (
                "Recoverable workflow error "
                "exhausted retry attempts."
            )

        _log_route_decision(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_error_route_failed",
            extra={
                "execution_id": execution_id,
            },
        )

        return ROUTE_FAILED


def route_on_start(
    state: dict[str, Any],
) -> RouteName:
    """
    Route at workflow startup.

    Routes to:
    - `end` when cancellation is requested.
    - `failed` when a blocking error is already present.
    - `retry` when a recoverable startup error can be retried.
    - `continue` when the workflow can begin normally.
    """
    execution_id = _safe_get(
        state,
        "execution_id",
    )

    try:
        if is_workflow_cancelled(state):
            route: RouteName = ROUTE_END
            reason = (
                "Workflow cancellation requested "
                "before startup."
            )

        elif has_blocking_error(state):
            route = ROUTE_FAILED
            reason = (
                "Workflow cannot start because a "
                "blocking error is present."
            )

        elif should_retry(state):
            route = ROUTE_RETRY
            reason = (
                "Workflow startup can be retried."
            )

        else:
            route = ROUTE_CONTINUE
            reason = (
                "Workflow startup completed."
            )

        _log_route_decision(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_route_on_start_failed",
            extra={
                "execution_id": execution_id,
            },
        )

        return ROUTE_FAILED


def route_on_pause(
    state: dict[str, Any],
) -> RouteName:
    """
    Route when a pause request is evaluated.
    """
    execution_id = _safe_get(
        state,
        "execution_id",
    )

    try:
        if is_workflow_paused(state):
            route: RouteName = ROUTE_END
            reason = (
                "Workflow pause requested."
            )

        elif is_workflow_cancelled(state):
            route = ROUTE_END
            reason = (
                "Workflow cancellation requested."
            )

        else:
            route = ROUTE_CONTINUE
            reason = (
                "No pause requested."
            )

        _log_route_decision(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_route_on_pause_failed",
            extra={
                "execution_id": execution_id,
            },
        )

        return ROUTE_FAILED


def route_on_completion(
    state: dict[str, Any],
) -> RouteName:
    """
    Route when workflow completion is evaluated.
    """
    execution_id = _safe_get(
        state,
        "execution_id",
    )

    try:
        current_step = _normalize_step(
            _safe_get(
                state,
                "current_step",
            ),
        )

        if is_workflow_cancelled(state):
            route: RouteName = ROUTE_END
            reason = (
                "Workflow cancellation requested."
            )

        elif has_blocking_error(state):
            route = ROUTE_FAILED
            reason = (
                "Workflow completed with a blocking "
                "error."
            )

        elif (
            current_step
            == WorkflowStep.COMPLETED.value
        ):
            route = ROUTE_END
            reason = (
                "Workflow completed successfully."
            )

        else:
            route = ROUTE_CONTINUE
            reason = (
                "Workflow has not yet completed."
            )

        _log_route_decision(
            execution_id=execution_id,
            route=route,
            reason=reason,
        )

        return route

    except Exception:
        logger.exception(
            "workflow_route_on_completion_failed",
            extra={
                "execution_id": execution_id,
            },
        )

        return ROUTE_FAILED


__all__ = [
    "ROUTE_CONTINUE",
    "ROUTE_END",
    "ROUTE_FAILED",
    "ROUTE_HUMAN_REVIEW",
    "ROUTE_RETRY",
    "ROUTE_UNKNOWN",
    "RouteName",
    "has_blocking_error",
    "has_generated_artifacts",
    "has_retrieval_context",
    "has_requirements",
    "is_validation_failed",
    "is_validation_passed",
    "is_workflow_cancelled",
    "is_workflow_paused",
    "requires_human_approval",
    "route_after_design",
    "route_after_generation",
    "route_after_publication",
    "route_after_requirements",
    "route_after_retrieval",
    "route_after_review",
    "route_after_validation",
    "route_on_completion",
    "route_on_pause",
    "route_on_start",
    "route_on_workflow_error",
    "should_retry",
]