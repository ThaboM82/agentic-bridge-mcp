"""
LangGraph workflow runner for AgenticBridge.

This module owns workflow execution orchestration: it validates initial state,
creates checkpoint-backed graph execution, applies workflow callbacks, records
telemetry, handles cancellation and pause requests, and maps execution outcomes
to a safe API response.

Security and reliability rules:
- Never log prompts, completions, generated code, document contents,
  embeddings, credentials, database URLs, API keys, or customer PII.
- Treat workflow state and external service responses as untrusted input.
- Use checkpoint thread IDs bound to the workflow execution ID.
- Enforce execution timeouts and bounded concurrency.
- Never expose internal stack traces, provider payloads, or raw state in errors.
- Human approval is required before publication when configured.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langgraph.graph.state import CompiledStateGraph

from app.core.config import settings
from app.core.enums import JobStatus, WorkflowStep
from app.core.exceptions import (
    ConfigurationError,
    ExternalProviderError,
    ProviderTimeoutError,
    WorkflowStateError,
)
from app.core.logging_context import bind_workflow_context
from app.graph.callbacks import create_workflow_callbacks
from app.graph.checkpoints import create_checkpointer
from app.graph.recovery import (
    classify_exception,
    prepare_cancellation,
    record_blocking_failure,
    record_recoverable_failure,
)
from app.graph.state import (
    AgenticState,
    create_initial_state,
    set_workflow_status,
)
from app.graph.telemetry import (
    emit_workflow_cancelled,
    emit_workflow_completed,
    emit_workflow_failed,
    emit_workflow_started,
)


logger = logging.getLogger("agentic_bridge.graph.runner")

MAX_CONCURRENT_EXECUTIONS = 100
MIN_EXECUTION_TIMEOUT_SECONDS = 30
MAX_EXECUTION_TIMEOUT_SECONDS = 7_200


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkflowRunResult:
    """
    Safe workflow execution result returned to API and service layers.

    This result intentionally excludes generated code contents, raw provider
    responses, prompts, completions, embeddings, and internal checkpoint data.
    """

    execution_id: str
    status: JobStatus
    current_step: WorkflowStep | None
    review_status: str | None
    approval_required: bool
    error_code: str | None
    error_message: str | None
    recoverable_error: bool
    duration_ms: float
    token_usage: dict[str, int]
    estimated_cost_usd: float | None
    artifact_count: int
    citation_count: int
    requirement_count: int
    interrupt_pending: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, API-safe result dictionary."""
        return {
            "execution_id": self.execution_id,
            "status": self.status.value,
            "current_step": (
                self.current_step.value
                if self.current_step
                else None
            ),
            "review_status": self.review_status,
            "approval_required": self.approval_required,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "recoverable_error": self.recoverable_error,
            "duration_ms": self.duration_ms,
            "token_usage": self.token_usage,
            "estimated_cost_usd": self.estimated_cost_usd,
            "artifact_count": self.artifact_count,
            "citation_count": self.citation_count,
            "requirement_count": self.requirement_count,
            "interrupt_pending": self.interrupt_pending,
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class WorkflowRunner:
    """
    Execute a compiled LangGraph workflow with production safeguards.

    The runner is responsible for execution lifecycle only. Workflow policy,
    routing, recovery decisions, and node behavior remain in the graph and its
    dedicated modules.
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        *,
        max_concurrent_executions: int | None = None,
        execution_timeout_seconds: int | None = None,
    ) -> None:
        """
        Initialize the workflow runner.

        Args:
            graph: Compiled LangGraph workflow.
            max_concurrent_executions: Optional concurrency override.
            execution_timeout_seconds: Optional timeout override.

        Raises:
            ConfigurationError: If concurrency or timeout values are invalid.
        """
        if not graph:
            raise ConfigurationError(
                "A compiled LangGraph workflow is required."
            )

        configured_concurrency = (
            max_concurrent_executions
            or settings.WORKFLOW_MAX_CONCURRENT_EXECUTIONS
        )

        if not isinstance(
            configured_concurrency,
            int,
        ) or configured_concurrency < 1:
            raise ConfigurationError(
                "Workflow concurrency must be a positive integer."
            )

        self._max_concurrent_executions = min(
            configured_concurrency,
            MAX_CONCURRENT_EXECUTIONS,
        )

        configured_timeout = (
            execution_timeout_seconds
            or settings.WORKFLOW_EXECUTION_TIMEOUT_SECONDS
        )

        if not isinstance(
            configured_timeout,
            int,
        ) or configured_timeout < MIN_EXECUTION_TIMEOUT_SECONDS:
            raise ConfigurationError(
                "Workflow execution timeout is too low."
            )

        self._execution_timeout_seconds = min(
            configured_timeout,
            MAX_EXECUTION_TIMEOUT_SECONDS,
        )

        self._graph = graph
        self._semaphore = asyncio.Semaphore(
            self._max_concurrent_executions
        )

        logger.info(
            "workflow_runner_initialized",
            max_concurrent_executions=(
                self._max_concurrent_executions
            ),
            execution_timeout_seconds=(
                self._execution_timeout_seconds
            ),
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def run(
        self,
        *,
        execution_id: str | UUID,
        brief_id: str | UUID,
        raw_brief_text: str,
        tenant_id: str | UUID | None = None,
        user_id: str | UUID | None = None,
        request_id: str | UUID | None = None,
        correlation_id: str | UUID | None = None,
        workflow_name: str | None = None,
        workflow_version: str | None = "1.0.0",
        max_retries: int = 3,
        approval_required: bool = False,
        initial_state_overrides: dict[str, Any] | None = None,
    ) -> WorkflowRunResult:
        """
        Execute the workflow from an initial state.

        Args:
            execution_id: Unique workflow execution identifier.
            brief_id: Source brief identifier.
            raw_brief_text: Initial brief or requirements text.
            tenant_id: Tenant identifier.
            user_id: Requesting user identifier.
            request_id: HTTP request correlation identifier.
            correlation_id: Business-operation correlation identifier.
            workflow_name: Human-readable workflow name.
            workflow_version: Workflow definition version.
            max_retries: Maximum retry attempts for recoverable failures.
            approval_required: Whether human approval is required.
            initial_state_overrides: Optional bounded initial-state overrides.

        Returns:
            A safe `WorkflowRunResult`.

        Raises:
            WorkflowStateError: If initial workflow state is invalid.
            ExternalProviderError: If workflow execution fails due to provider
                availability.
        """
        started_at = time.perf_counter()

        normalized_execution_id = str(execution_id)
        normalized_brief_id = str(brief_id)
        normalized_tenant_id = (
            str(tenant_id) if tenant_id else None
        )
        normalized_user_id = (
            str(user_id) if user_id else None
        )

        bind_workflow_context(
            normalized_execution_id,
            tenant_id=normalized_tenant_id,
            user_id=normalized_user_id,
        )

        try:
            initial_state = create_initial_state(
                execution_id=normalized_execution_id,
                brief_id=normalized_brief_id,
                raw_brief_text=raw_brief_text,
                tenant_id=normalized_tenant_id,
                user_id=normalized_user_id,
                request_id=(
                    str(request_id) if request_id else None
                ),
                correlation_id=(
                    str(correlation_id)
                    if correlation_id
                    else None
                ),
                workflow_name=workflow_name,
                workflow_version=workflow_version,
                max_retries=max_retries,
                approval_required=approval_required,
            )

            if initial_state_overrides:
                initial_state = {
                    **initial_state,
                    **initial_state_overrides,
                }

            emit_workflow_started(
                execution_id=normalized_execution_id,
                brief_id=normalized_brief_id,
                tenant_id=normalized_tenant_id,
                user_id=normalized_user_id,
                workflow_name=workflow_name,
                workflow_version=workflow_version,
            )

            async with self._semaphore:
                final_state = await self._execute_graph(
                    initial_state=initial_state,
                    execution_id=normalized_execution_id,
                    tenant_id=normalized_tenant_id,
                    user_id=normalized_user_id,
                    workflow_name=workflow_name,
                )

            duration_ms = (
                time.perf_counter() - started_at
            ) * 1_000

            return self._build_result(
                final_state=final_state,
                duration_ms=duration_ms,
            )

        except asyncio.TimeoutError:
            duration_ms = (
                time.perf_counter() - started_at
            ) * 1_000

            logger.error(
                "workflow_execution_timeout",
                execution_id=normalized_execution_id,
                brief_id=normalized_brief_id,
                timeout_seconds=(
                    self._execution_timeout_seconds
                ),
                duration_ms=round(duration_ms, 3),
            )

            emit_workflow_failed(
                execution_id=normalized_execution_id,
                brief_id=normalized_brief_id,
                tenant_id=normalized_tenant_id,
                user_id=normalized_user_id,
                workflow_name=workflow_name,
                error_code="workflow_timeout",
                error_step=None,
                recoverable=False,
            )

            raise WorkflowStateError(
                "Workflow execution exceeded the allowed time limit."
            ) from None

        except (ProviderTimeoutError, ExternalProviderError) as exc:
            duration_ms = (
                time.perf_counter() - started_at
            ) * 1_000

            error_code, recoverable, safe_message = (
                classify_exception(exc)
            )

            logger.error(
                "workflow_execution_provider_failure",
                execution_id=normalized_execution_id,
                brief_id=normalized_brief_id,
                error_code=error_code,
                duration_ms=round(duration_ms, 3),
            )

            emit_workflow_failed(
                execution_id=normalized_execution_id,
                brief_id=normalized_brief_id,
                tenant_id=normalized_tenant_id,
                user_id=normalized_user_id,
                workflow_name=workflow_name,
                error_code=error_code,
                recoverable=recoverable,
            )

            raise ExternalProviderError(
                safe_message
            ) from None

        except WorkflowStateError:
            duration_ms = (
                time.perf_counter() - started_at
            ) * 1_000

            logger.warning(
                "workflow_execution_invalid_state",
                execution_id=normalized_execution_id,
                brief_id=normalized_brief_id,
                duration_ms=round(duration_ms, 3),
            )

            raise

        except Exception as exc:
            duration_ms = (
                time.perf_counter() - started_at
            ) * 1_000

            error_code, recoverable, safe_message = (
                classify_exception(exc)
            )

            logger.exception(
                "workflow_execution_unexpected_failure",
                execution_id=normalized_execution_id,
                brief_id=normalized_brief_id,
                error_code=error_code,
                duration_ms=round(duration_ms, 3),
            )

            emit_workflow_failed(
                execution_id=normalized_execution_id,
                brief_id=normalized_brief_id,
                tenant_id=normalized_tenant_id,
                user_id=normalized_user_id,
                workflow_name=workflow_name,
                error_code=error_code,
                recoverable=recoverable,
            )

            raise WorkflowStateError(
                safe_message
            ) from exc

    async def cancel(
        self,
        state: AgenticState,
        *,
        cancelled_by: str | UUID | None = None,
        reason: str | None = None,
    ) -> AgenticState:
        """
        Mark a workflow as cancellation-requested.

        This method updates workflow state and emits telemetry. It does not
        forcibly terminate an already-running graph invocation; nodes and
        guards must honor `cancel_requested` at safe boundaries.

        Args:
            state: Current workflow state.
            cancelled_by: User or operator requesting cancellation.
            reason: Optional cancellation reason.

        Returns:
            Updated workflow state.
        """
        execution_id = str(
            state.get("execution_id") or "unknown-execution"
        )
        brief_id = str(
            state.get("brief_id") or "unknown-brief"
        )
        tenant_id = state.get("tenant_id")

        cancellation_updates = prepare_cancellation(
            state,
            cancelled_by=(
                str(cancelled_by) if cancelled_by else None
            ),
            reason=reason,
        )

        updated_state = {
            **state,
            **cancellation_updates,
        }

        emit_workflow_cancelled(
            execution_id=execution_id,
            brief_id=brief_id,
            tenant_id=(
                str(tenant_id) if tenant_id else None
            ),
            user_id=(
                str(cancelled_by) if cancelled_by else None
            ),
            workflow_name=state.get("workflow_name"),
        )

        logger.warning(
            "workflow_cancellation_requested",
            execution_id=execution_id,
            brief_id=brief_id,
        )

        return updated_state

    # -----------------------------------------------------------------
    # Internal execution
    # -----------------------------------------------------------------

    async def _execute_graph(
        self,
        *,
        initial_state: AgenticState,
        execution_id: str,
        tenant_id: str | None,
        user_id: str | None,
        workflow_name: str | None,
    ) -> AgenticState:
        """
        Execute the compiled graph with checkpointing and callbacks.

        Args:
            initial_state: Validated initial workflow state.
            execution_id: Workflow execution identifier.
            tenant_id: Tenant identifier.
            user_id: User identifier.
            workflow_name: Workflow name.

        Returns:
            The final workflow state.

        Raises:
            WorkflowStateError: If graph execution fails.
        """
        checkpointer = create_checkpointer()

        callbacks = create_workflow_callbacks(
            execution_id=execution_id,
            tenant_id=tenant_id,
        )

        runnable_config = {
            "configurable": {
                "thread_id": execution_id,
            },
            "callbacks": callbacks,
            "tags": [
                f"execution:{execution_id}",
                f"tenant:{tenant_id}" if tenant_id else None,
                f"workflow:{workflow_name}" if workflow_name else None,
            ],
        }

        runnable_config["tags"] = [
            tag
            for tag in runnable_config["tags"]
            if tag
        ]

        try:
            final_state = await asyncio.wait_for(
                self._graph.ainvoke(
                    initial_state,
                    config=runnable_config,
                    checkpointer=checkpointer,
                ),
                timeout=self._execution_timeout_seconds,
            )

        except TypeError:
            # Some LangGraph versions accept the checkpointer at compile time
            # rather than through `ainvoke`.
            final_state = await asyncio.wait_for(
                self._graph.ainvoke(
                    initial_state,
                    config=runnable_config,
                ),
                timeout=self._execution_timeout_seconds,
            )

        if not isinstance(final_state, dict):
            raise WorkflowStateError(
                "Workflow execution returned an invalid state."
            )

        return final_state

    # -----------------------------------------------------------------
    # Result projection
    # -----------------------------------------------------------------

    def _build_result(
        self,
        *,
        final_state: AgenticState,
        duration_ms: float,
    ) -> WorkflowRunResult:
        """
        Build a safe workflow result from final graph state.

        Args:
            final_state: Final workflow state.
            duration_ms: Total execution duration in milliseconds.

        Returns:
            A safe `WorkflowRunResult`.
        """
        execution_id = str(
            final_state.get("execution_id") or "unknown-execution"
        )

        brief_id = str(
            final_state.get("brief_id") or "unknown-brief"
        )

        tenant_id = final_state.get("tenant_id")
        user_id = final_state.get("user_id")
        workflow_name = final_state.get("workflow_name")

        workflow_status = final_state.get(
            "workflow_status",
            JobStatus.FAILED,
        )

        current_step = final_state.get("current_step")

        current_error = final_state.get("current_error")
        error_code = final_state.get("error_code")

        recoverable_error = bool(
            final_state.get("recoverable_error", False)
        )

        token_usage = final_state.get(
            "provider_token_usage",
            {},
        )

        if not isinstance(token_usage, dict):
            token_usage = {}

        current_interrupt = final_state.get(
            "current_interrupt"
        )

        interrupt_pending = (
            isinstance(current_interrupt, dict)
            and current_interrupt.get("status")
            == "pending_review"
        )

        if workflow_status == JobStatus.COMPLETED:
            emit_workflow_completed(
                execution_id=execution_id,
                brief_id=brief_id,
                tenant_id=(
                    str(tenant_id) if tenant_id else None
                ),
                user_id=(
                    str(user_id) if user_id else None
                ),
                workflow_name=workflow_name,
                duration_ms=round(duration_ms, 3),
                token_usage=token_usage,
                estimated_cost_usd=final_state.get(
                    "estimated_cost_usd"
                ),
            )

        elif workflow_status == JobStatus.CANCELLED:
            emit_workflow_cancelled(
                execution_id=execution_id,
                brief_id=brief_id,
                tenant_id=(
                    str(tenant_id) if tenant_id else None
                ),
                user_id=(
                    str(user_id) if user_id else None
                ),
                workflow_name=workflow_name,
                cancelled_by=final_state.get(
                    "cancelled_by"
                ),
            )

        elif workflow_status == JobStatus.FAILED:
            emit_workflow_failed(
                execution_id=execution_id,
                brief_id=brief_id,
                tenant_id=(
                    str(tenant_id) if tenant_id else None
                ),
                user_id=(
                    str(user_id) if user_id else None
                ),
                workflow_name=workflow_name,
                error_code=error_code,
                error_step=(
                    final_state.get("error_step").value
                    if final_state.get("error_step")
                    else None
                ),
                recoverable=recoverable_error,
            )

        return WorkflowRunResult(
            execution_id=execution_id,
            status=workflow_status,
            current_step=current_step,
            review_status=final_state.get("review_status"),
            approval_required=bool(
                final_state.get("approval_required", False)
            ),
            error_code=error_code,
            error_message=(
                str(current_error)
                if current_error
                else None
            ),
            recoverable_error=recoverable_error,
            duration_ms=round(duration_ms, 3),
            token_usage={
                key: int(value)
                for key, value in token_usage.items()
                if isinstance(value, (int, float))
            },
            estimated_cost_usd=final_state.get(
                "estimated_cost_usd"
            ),
            artifact_count=len(
                final_state.get("artifacts", [])
            ),
            citation_count=len(
                final_state.get("citations", [])
            ),
            requirement_count=len(
                final_state.get(
                    "extracted_requirements",
                    [],
                )
            ),
            interrupt_pending=interrupt_pending,
        )


# ---------------------------------------------------------------------------
# Runner factory
# ---------------------------------------------------------------------------


def create_workflow_runner(
    graph: CompiledStateGraph,
    *,
    max_concurrent_executions: int | None = None,
    execution_timeout_seconds: int | None = None,
) -> WorkflowRunner:
    """
    Create a configured workflow runner.

    Args:
        graph: Compiled LangGraph workflow.
        max_concurrent_executions: Optional concurrency override.
        execution_timeout_seconds: Optional timeout override.

    Returns:
        A configured `WorkflowRunner`.

    Raises:
        ConfigurationError: If runner configuration is invalid.
    """
    return WorkflowRunner(
        graph,
        max_concurrent_executions=(
            max_concurrent_executions
        ),
        execution_timeout_seconds=(
            execution_timeout_seconds
        ),
    )