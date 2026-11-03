"""
Centralized LangGraph workflow state schema for AgenticBridge MCP.


This module defines the typed contract passed between workflow nodes. Nodes
should read only the fields they need and return only the fields they change.


Design principles:
- Use enums and typed fields instead of free-form strings where possible.
- Use reducers only for fields that legitimately accumulate across nodes.
- Bound large artifacts, logs, requirements, and retrieved context.
- Keep secrets, raw provider responses, passwords, tokens, and unbounded
  customer content out of workflow state.
- Preserve enough context for observability, recovery, human review,
  cost tracking, and audit.
"""


from __future__ import annotations


import json
from datetime import datetime, timezone
from operator import add
from typing import Annotated, Any, Literal


from typing_extensions import TypedDict


from app.core.constants import (
    MAX_DESIGN_COMPONENTS,
    MAX_DESIGN_CONTEXT_DOCUMENTS,
    MAX_GENERATED_FILE_COUNT,
    MAX_WORKFLOW_LOG_ENTRIES,
    MAX_WORKFLOW_REQUIREMENTS,
)
from app.core.enums import ApprovalStatus, JobStatus, WorkflowStep



# ---------------------------------------------------------------------------
# Bounded type aliases
# ---------------------------------------------------------------------------


RequirementList = Annotated[list[str], add]
ExecutionLogList = Annotated[list[str], add]
ValidationIssueList = Annotated[list[dict[str, Any]], add]
RetrievedContextList = Annotated[list[dict[str, Any]], add]
ArtifactList = Annotated[list[dict[str, Any]], add]
InterruptHistoryList = Annotated[list[dict[str, Any]], add]


GeneratedFileMap = dict[str, str]
DesignTokenMap = dict[str, Any]
TestResultMap = dict[str, Any]
ProviderUsageMap = dict[str, int]
CheckpointMap = dict[str, Any]
InterruptPayloadMap = dict[str, Any]


ReviewDecision = Literal[
    "pending_approval",
    "approved",
    "rejected",
    "changes_requested",
    "cancelled",
]


ValidationOutcome = Literal[
    "pending",
    "passed",
    "failed",
    "warning",
    "skipped",
]


InterruptStatus = Literal[
    "pending_review",
    "resolved",
    "expired",
    "cancelled",
]


InterruptType = Literal[
    "human_review",
    "clarification_required",
    "approval_required",
    "validation_failed",
    "policy_review",
]



# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------



class AgenticState(TypedDict, total=False):
    """
    Centralized state schema passed across LangGraph workflow nodes.


    All fields are optional because workflow nodes initialize state
    progressively. The workflow service should validate required fields before
    execution begins.
    """


    # -----------------------------------------------------------------
    # Execution identity and correlation
    # -----------------------------------------------------------------


    execution_id: str
    brief_id: str
    tenant_id: str | None
    user_id: str | None
    request_id: str | None
    correlation_id: str | None


    workflow_name: str | None
    workflow_version: str | None
    workflow_status: JobStatus
    current_step: WorkflowStep
    previous_step: WorkflowStep | None


    started_at: str | None
    completed_at: str | None
    duration_ms: float | None


    # -----------------------------------------------------------------
    # Workflow control
    # -----------------------------------------------------------------


    cancel_requested: bool
    cancelled_by: str | None
    cancellation_reason: str | None
    cancelled_at: str | None


    pause_requested: bool
    paused_by: str | None
    pause_reason: str | None
    paused_at: str | None
    resumed_at: str | None


    # -----------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------


    raw_brief_text: str
    document_ids: list[str]
    ingestion_job_id: str | None
    ingestion_status: JobStatus
    ingestion_started_at: str | None
    ingestion_completed_at: str | None


    extracted_requirements: RequirementList
    requirement_confidence: float | None
    requirement_metadata: dict[str, Any]
    requirements_finalized: bool


    # -----------------------------------------------------------------
    # Retrieval and design context
    # -----------------------------------------------------------------


    retrieval_query: str | None
    retrieval_model: str | None
    retrieval_score_threshold: float | None
    retrieval_completed_at: str | None


    vector_context_matches: RetrievedContextList
    vector_context_document_ids: list[str]


    design_tokens: DesignTokenMap
    design_rationale: str | None
    accessibility_requirements: list[str]
    design_completed_at: str | None


    # -----------------------------------------------------------------
    # Code generation
    # -----------------------------------------------------------------


    generated_code_files: GeneratedFileMap
    generated_file_manifest: list[dict[str, Any]]
    code_generation_model: str | None
    code_generation_prompt_version: str | None
    code_generation_completed_at: str | None


    # -----------------------------------------------------------------
    # Validation and quality gates
    # -----------------------------------------------------------------


    validation_status: JobStatus
    validation_outcome: ValidationOutcome
    validation_issues: ValidationIssueList
    validation_score: float | None
    static_analysis_passed: bool | None
    security_scan_passed: bool | None
    accessibility_check_passed: bool | None
    tests_generated: bool | None
    test_results: TestResultMap
    validation_completed_at: str | None


    # -----------------------------------------------------------------
    # Human-in-the-loop governance
    # -----------------------------------------------------------------


    review_status: ReviewDecision
    approval_required: bool
    approval_request_id: str | None
    approval_status: ApprovalStatus
    approval_comment: str | None
    approved_by: str | None
    approved_at: str | None
    rejected_by: str | None
    rejected_at: str | None
    reviewer_role: str | None


    # -----------------------------------------------------------------
    # Human-in-the-loop interrupts
    # -----------------------------------------------------------------


    current_interrupt: InterruptPayloadMap | None
    interrupt_history: InterruptHistoryList
    interrupt_reason: str | None
    interrupt_type: InterruptType | None
    interrupt_created_at: str | None
    interrupt_resolved_at: str | None


    # -----------------------------------------------------------------
    # Publishing
    # -----------------------------------------------------------------


    publishing_status: JobStatus
    publication_target: str | None
    publication_url: str | None
    published_artifact_ids: ArtifactList
    publishing_completed_at: str | None


    # -----------------------------------------------------------------
    # Artifacts and citations
    # -----------------------------------------------------------------


    artifacts: ArtifactList
    citations: list[dict[str, Any]]


    # -----------------------------------------------------------------
    # Resilience, recovery, and observability
    # -----------------------------------------------------------------


    current_error: str | None
    error_step: WorkflowStep | None
    error_code: str | None
    error_occurred_at: str | None
    recoverable_error: bool


    retry_count: int
    max_retries: int
    last_retry_at: str | None


    checkpoint: CheckpointMap
    resumed_from_checkpoint: bool


    execution_logs: ExecutionLogList


    # -----------------------------------------------------------------
    # Provider telemetry
    # -----------------------------------------------------------------


    provider_name: str | None
    provider_model: str | None
    provider_latency_ms: float | None
    provider_token_usage: ProviderUsageMap
    estimated_cost_usd: float | None



# ---------------------------------------------------------------------------
# State factories
# ---------------------------------------------------------------------------



def create_initial_state(
    *,
    execution_id: str,
    brief_id: str,
    raw_brief_text: str,
    tenant_id: str | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
    correlation_id: str | None = None,
    workflow_name: str | None = None,
    workflow_version: str | None = "1.0.0",
    max_retries: int = 3,
    approval_required: bool = False,
) -> AgenticState:
    """
    Create a validated initial workflow state.


    Args:
        execution_id: Unique workflow execution identifier.
        brief_id: Source brief identifier.
        raw_brief_text: Initial brief or requirements text.
        tenant_id: Optional tenant identifier.
        user_id: Optional requesting user identifier.
        request_id: Optional HTTP request correlation identifier.
        correlation_id: Optional business-operation correlation identifier.
        workflow_name: Optional human-readable workflow name.
        workflow_version: Workflow definition version.
        max_retries: Maximum retry attempts for recoverable failures.
        approval_required: Whether human approval is required before publishing.


    Returns:
        A complete initial `AgenticState`.


    Raises:
        ValueError: If required identifiers or brief text are missing.
    """
    if not execution_id or not execution_id.strip():
        raise ValueError("execution_id is required.")


    if not brief_id or not brief_id.strip():
        raise ValueError("brief_id is required.")


    if not raw_brief_text or not raw_brief_text.strip():
        raise ValueError("raw_brief_text is required.")


    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")


    return AgenticState(
        execution_id=execution_id,
        brief_id=brief_id,
        raw_brief_text=raw_brief_text,
        tenant_id=tenant_id,
        user_id=user_id,
        request_id=request_id,
        correlation_id=correlation_id,
        workflow_name=workflow_name,
        workflow_version=workflow_version,
        workflow_status=JobStatus.QUEUED,
        current_step=WorkflowStep.INITIALIZED,
        previous_step=None,
        started_at=_utc_now_iso(),
        completed_at=None,
        duration_ms=None,


        cancel_requested=False,
        cancelled_by=None,
        cancellation_reason=None,
        cancelled_at=None,


        pause_requested=False,
        paused_by=None,
        pause_reason=None,
        paused_at=None,
        resumed_at=None,


        document_ids=[],
        ingestion_job_id=None,
        ingestion_status=JobStatus.QUEUED,
        ingestion_started_at=None,
        ingestion_completed_at=None,


        extracted_requirements=[],
        requirement_confidence=None,
        requirement_metadata={},
        requirements_finalized=False,


        retrieval_query=None,
        retrieval_model=None,
        retrieval_score_threshold=None,
        retrieval_completed_at=None,


        vector_context_matches=[],
        vector_context_document_ids=[],


        design_tokens={},
        design_rationale=None,
        accessibility_requirements=[],
        design_completed_at=None,


        generated_code_files={},
        generated_file_manifest=[],
        code_generation_model=None,
        code_generation_prompt_version=None,
        code_generation_completed_at=None,


        validation_status=JobStatus.QUEUED,
        validation_outcome="pending",
        validation_issues=[],
        validation_score=None,
        static_analysis_passed=None,
        security_scan_passed=None,
        accessibility_check_passed=None,
        tests_generated=None,
        test_results={},
        validation_completed_at=None,


        review_status="pending_approval",
        approval_required=approval_required,
        approval_request_id=None,
        approval_status=ApprovalStatus.PENDING,
        approval_comment=None,
        approved_by=None,
        approved_at=None,
        rejected_by=None,
        rejected_at=None,
        reviewer_role=None,


        current_interrupt=None,
        interrupt_history=[],
        interrupt_reason=None,
        interrupt_type=None,
        interrupt_created_at=None,
        interrupt_resolved_at=None,


        publishing_status=JobStatus.QUEUED,
        publication_target=None,
        publication_url=None,
        published_artifact_ids=[],
        publishing_completed_at=None,


        artifacts=[],
        citations=[],


        current_error=None,
        error_step=None,
        error_code=None,
        error_occurred_at=None,
        recoverable_error=False,


        retry_count=0,
        max_retries=max_retries,
        last_retry_at=None,


        checkpoint={},
        resumed_from_checkpoint=False,


        execution_logs=[
            f"INFO: Workflow execution initialized for brief {brief_id}."
        ],


        provider_name=None,
        provider_model=None,
        provider_latency_ms=None,
        provider_token_usage={},
        estimated_cost_usd=None,
    )



# ---------------------------------------------------------------------------
# Immutable update helpers
# ---------------------------------------------------------------------------



def _merge_state(
    state: AgenticState,
    updates: dict[str, Any],
) -> AgenticState:
    """Create a new state dictionary with the supplied updates applied."""
    return AgenticState(
        **{
            **state,
            **updates,
        }
    )



def append_execution_log(
    state: AgenticState,
    message: str,
    *,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
) -> AgenticState:
    """
    Return a new state with one bounded execution-log entry appended.


    Because `execution_logs` uses LangGraph's `add` reducer, return only the
    new entry; do not append it to the existing list.
    """
    normalized_message = message.strip()


    if not normalized_message:
        return state


    bounded_message = normalized_message[:2_000]


    return _merge_state(
        state,
        {
            "execution_logs": [
                f"{level}: {bounded_message}"
            ],
        },
    )



def set_current_step(
    state: AgenticState,
    workflow_step: WorkflowStep,
) -> AgenticState:
    """Return a new state with the current and previous workflow steps set."""
    previous_step = state.get("current_step")


    return _merge_state(
        state,
        {
            "previous_step": previous_step,
            "current_step": workflow_step,
        },
    )



def set_workflow_status(
    state: AgenticState,
    workflow_status: JobStatus,
) -> AgenticState:
    """Return a new state with the workflow status updated."""
    return _merge_state(
        state,
        {
            "workflow_status": workflow_status,
        },
    )



def set_error(
    state: AgenticState,
    *,
    message: str,
    error_step: WorkflowStep | None = None,
    error_code: str | None = None,
    recoverable: bool = False,
) -> AgenticState:
    """
    Return a new state with a bounded workflow error recorded.


    Avoid raw exception text, stack traces, provider responses, credentials, or
    full customer content in this field.
    """
    bounded_message = message.strip()[:2_000]


    return _merge_state(
        state,
        {
            "current_error": bounded_message,
            "error_step": error_step,
            "error_code": error_code,
            "error_occurred_at": _utc_now_iso(),
            "recoverable_error": recoverable,
            "workflow_status": JobStatus.FAILED,
            "execution_logs": [
                f"ERROR: {bounded_message}"
            ],
        },
    )



def clear_error(state: AgenticState) -> AgenticState:
    """Return a new state with error fields cleared."""
    return _merge_state(
        state,
        {
            "current_error": None,
            "error_step": None,
            "error_code": None,
            "error_occurred_at": None,
            "recoverable_error": False,
        },
    )



def increment_retry_count(state: AgenticState) -> AgenticState:
    """
    Return a new state with the retry counter incremented.


    Raises:
        ValueError: If the workflow has exhausted its retry allowance.
    """
    current_retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 0)


    next_retry_count = current_retry_count + 1


    if next_retry_count > max_retries:
        raise ValueError(
            "Workflow retry allowance has been exhausted."
        )


    return _merge_state(
        state,
        {
            "retry_count": next_retry_count,
            "last_retry_at": _utc_now_iso(),
        },
    )



def save_checkpoint(
    state: AgenticState,
    *,
    checkpoint_step: WorkflowStep,
    checkpoint_data: dict[str, Any] | None = None,
) -> AgenticState:
    """
    Save a recovery checkpoint.


    Checkpoint data must contain only serializable, non-sensitive values.
    """
    return _merge_state(
        state,
        {
            "checkpoint": {
                "step": checkpoint_step.value,
                "saved_at": _utc_now_iso(),
                "data": checkpoint_data or {},
            },
        },
    )



def mark_resumed_from_checkpoint(state: AgenticState) -> AgenticState:
    """Mark the workflow as resumed from a saved checkpoint."""
    return _merge_state(
        state,
        {
            "resumed_from_checkpoint": True,
            "resumed_at": _utc_now_iso(),
        },
    )



def add_provider_usage(
    state: AgenticState,
    *,
    provider_name: str,
    provider_model: str,
    latency_ms: float | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int | None = None,
    estimated_cost_usd: float | None = None,
) -> AgenticState:
    """
    Record non-sensitive language-model usage telemetry.


    Token values are accumulated across workflow nodes.
    """
    existing_usage = dict(state.get("provider_token_usage", {}))


    existing_usage["prompt_tokens"] = (
        existing_usage.get("prompt_tokens", 0) + prompt_tokens
    )
    existing_usage["completion_tokens"] = (
        existing_usage.get("completion_tokens", 0) + completion_tokens
    )


    calculated_total = (
        existing_usage["prompt_tokens"]
        + existing_usage["completion_tokens"]
    )


    existing_usage["total_tokens"] = (
        total_tokens
        if total_tokens is not None
        else calculated_total
    )


    current_cost = state.get("estimated_cost_usd") or 0.0


    return _merge_state(
        state,
        {
            "provider_name": provider_name,
            "provider_model": provider_model,
            "provider_latency_ms": latency_ms,
            "provider_token_usage": existing_usage,
            "estimated_cost_usd": (
                current_cost + estimated_cost_usd
                if estimated_cost_usd is not None
                else current_cost
            ),
        },
    )



# ---------------------------------------------------------------------------
# Workflow control helpers
# ---------------------------------------------------------------------------



def request_cancellation(
    state: AgenticState,
    *,
    cancelled_by: str | None = None,
    reason: str | None = None,
) -> AgenticState:
    """
    Mark a workflow as cancellation-requested.


    This does not terminate a running graph invocation by itself. Workflow
    guards and nodes must check `cancel_requested` at safe boundaries.
    """
    bounded_reason = (reason or "").strip()[:1_000] or None


    return _merge_state(
        state,
        {
            "cancel_requested": True,
            "cancelled_by": cancelled_by,
            "cancellation_reason": bounded_reason,
            "cancelled_at": _utc_now_iso(),
            "workflow_status": JobStatus.CANCELLED,
            "execution_logs": [
                "WARNING: Workflow cancellation requested."
            ],
        },
    )



def request_pause(
    state: AgenticState,
    *,
    paused_by: str | None = None,
    reason: str | None = None,
) -> AgenticState:
    """
    Mark a workflow as pause-requested.


    This does not suspend a running graph invocation by itself. Workflow
    guards and nodes must check `pause_requested` at safe boundaries.
    """
    bounded_reason = (reason or "").strip()[:1_000] or None


    return _merge_state(
        state,
        {
            "pause_requested": True,
            "paused_by": paused_by,
            "pause_reason": bounded_reason,
            "paused_at": _utc_now_iso(),
            "execution_logs": [
                "WARNING: Workflow pause requested."
            ],
        },
    )



def resume_workflow(state: AgenticState) -> AgenticState:
    """Clear pause-requested state and record workflow resumption."""
    return _merge_state(
        state,
        {
            "pause_requested": False,
            "paused_by": None,
            "pause_reason": None,
            "paused_at": None,
            "resumed_at": _utc_now_iso(),
            "execution_logs": [
                "INFO: Workflow resumed."
            ],
        },
    )



# ---------------------------------------------------------------------------
# Interrupt helpers
# ---------------------------------------------------------------------------



def set_active_interrupt(
    state: AgenticState,
    interrupt_payload: dict[str, Any],
) -> AgenticState:
    """
    Set the active interrupt and append it to interrupt history.


    The interrupt payload must already be sanitized by `interrupts.py`.
    """
    interrupt_history = list(state.get("interrupt_history", []))
    interrupt_history.append(interrupt_payload)


    return _merge_state(
        state,
        {
            "current_interrupt": interrupt_payload,
            "interrupt_history": interrupt_history,
            "interrupt_type": interrupt_payload.get("interrupt_type"),
            "interrupt_reason": interrupt_payload.get("reason"),
            "interrupt_created_at": interrupt_payload.get("created_at"),
            "interrupt_resolved_at": None,
            "approval_required": True,
            "approval_status": ApprovalStatus.PENDING,
            "review_status": "pending_approval",
            "execution_logs": [
                (
                    "INFO: Workflow paused for "
                    f"{interrupt_payload.get('interrupt_type', 'review')}."
                )
            ],
        },
    )



def resolve_active_interrupt(
    state: AgenticState,
    *,
    decision: Literal[
        "approved",
        "changes_requested",
        "rejected",
    ],
    reviewer_id: str | None = None,
    comment: str | None = None,
    resolved_interrupt: dict[str, Any] | None = None,
) -> AgenticState:
    """
    Resolve the active interrupt and record the human decision.


    Args:
        state: Current workflow state.
        decision: Human review decision.
        reviewer_id: Optional reviewer identifier.
        comment: Optional bounded reviewer comment.
        resolved_interrupt: Optional resolved interrupt payload.


    Returns:
        Updated workflow state.
    """
    current_interrupt = state.get("current_interrupt") or {}
    resolved_payload = resolved_interrupt or {
        **current_interrupt,
        "status": "resolved",
        "resolved_at": _utc_now_iso(),
        "decision": decision,
        "reviewer_id": reviewer_id,
        "comment": (comment or "").strip()[:2_000] or None,
    }


    interrupt_history = list(state.get("interrupt_history", []))
    interrupt_history.append(resolved_payload)


    updates: dict[str, Any] = {
        "current_interrupt": None,
        "interrupt_history": interrupt_history,
        "interrupt_resolved_at": _utc_now_iso(),
        "approval_required": False,
        "approval_status": (
            ApprovalStatus.APPROVED
            if decision == "approved"
            else ApprovalStatus.REJECTED
        ),
        "review_status": decision,
        "approval_comment": (comment or "").strip()[:2_000] or None,
        "execution_logs": [
            f"INFO: Human review resolved with decision '{decision}'."
        ],
    }


    if decision == "approved":
        updates.update(
            {
                "approved_by": reviewer_id,
                "approved_at": _utc_now_iso(),
                "current_error": None,
                "error_step": None,
                "error_code": None,
                "recoverable_error": False,
            }
        )
    elif decision == "changes_requested":
        updates.update(
            {
                "current_error": "Human reviewer requested changes.",
                "error_step": state.get("current_step"),
                "error_code": "review_changes_requested",
                "recoverable_error": True,
            }
        )
    else:
        updates.update(
            {
                "rejected_by": reviewer_id,
                "rejected_at": _utc_now_iso(),
                "current_error": "Workflow rejected by human reviewer.",
                "error_step": state.get("current_step"),
                "error_code": "workflow_rejected",
                "recoverable_error": False,
                "workflow_status": JobStatus.FAILED,
            }
        )


    return _merge_state(state, updates)



def clear_active_interrupt(state: AgenticState) -> AgenticState:
    """Clear the active interrupt without recording a human decision."""
    return _merge_state(
        state,
        {
            "current_interrupt": None,
            "interrupt_resolved_at": _utc_now_iso(),
            "approval_required": False,
        },
    )



# ---------------------------------------------------------------------------
# Artifact and citation helpers
# ---------------------------------------------------------------------------



def add_artifact(
    state: AgenticState,
    *,
    artifact_id: str,
    artifact_type: str,
    uri: str | None = None,
    checksum: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgenticState:
    """
    Add a generated or published artifact reference to state.


    Do not store file contents in artifacts; store an ID, URI, checksum, and
    small metadata object instead.
    """
    artifact = {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "uri": uri,
        "checksum": checksum,
        "created_at": _utc_now_iso(),
        "metadata": metadata or {},
    }


    return _merge_state(
        state,
        {
            "artifacts": [artifact],
        },
    )



def add_citation(
    state: AgenticState,
    *,
    document_id: str,
    embedding_id: str | None = None,
    title: str | None = None,
    chunk_index: int | None = None,
    similarity: float | None = None,
    uri: str | None = None,
) -> AgenticState:
    """
    Add a retrieval citation to state.


    Citations make generated recommendations traceable to source documents.
    """
    citation = {
        "document_id": document_id,
        "embedding_id": embedding_id,
        "title": title,
        "chunk_index": chunk_index,
        "similarity": similarity,
        "uri": uri,
        "retrieved_at": _utc_now_iso(),
    }


    return _merge_state(
        state,
        {
            "citations": [citation],
        },
    )



def set_generated_files(
    state: AgenticState,
    files: dict[str, str],
) -> AgenticState:
    """
    Set generated code files with a bounded file count.


    Args:
        state: Current workflow state.
        files: Mapping of file path to source-code content.


    Raises:
        ValueError: If the file count exceeds the configured maximum.
    """
    if len(files) > MAX_GENERATED_FILE_COUNT:
        raise ValueError(
            f"Workflow cannot generate more than "
            f"{MAX_GENERATED_FILE_COUNT} files."
        )


    return _merge_state(
        state,
        {
            "generated_code_files": files,
        },
    )



def set_validation_result(
    state: AgenticState,
    *,
    outcome: ValidationOutcome,
    score: float | None = None,
    issues: list[dict[str, Any]] | None = None,
    static_analysis_passed: bool | None = None,
    security_scan_passed: bool | None = None,
    accessibility_check_passed: bool | None = None,
) -> AgenticState:
    """Record validation outcome and quality-gate results."""
    return _merge_state(
        state,
        {
            "validation_status": (
                JobStatus.COMPLETED
                if outcome in {"passed", "failed", "warning"}
                else JobStatus.PROCESSING
            ),
            "validation_outcome": outcome,
            "validation_score": score,
            "validation_issues": issues or [],
            "static_analysis_passed": static_analysis_passed,
            "security_scan_passed": security_scan_passed,
            "accessibility_check_passed": accessibility_check_passed,
            "validation_completed_at": _utc_now_iso(),
        },
    )



# ---------------------------------------------------------------------------
# State predicates
# ---------------------------------------------------------------------------



def can_retry(state: AgenticState) -> bool:
    """Determine whether the workflow may retry after a recoverable failure."""
    return (
        state.get("recoverable_error", False)
        and state.get("retry_count", 0) < state.get("max_retries", 0)
    )



def is_terminal(state: AgenticState) -> bool:
    """Determine whether the workflow has reached a terminal state."""
    workflow_status = state.get("workflow_status")


    if workflow_status in {
        JobStatus.COMPLETED,
        JobStatus.PARTIALLY_COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }:
        return True


    return state.get("current_step") in {
        WorkflowStep.COMPLETED,
        WorkflowStep.ERROR,
    }



def requires_approval(state: AgenticState) -> bool:
    """Determine whether the workflow is waiting for human approval."""
    return (
        state.get("approval_required", False)
        and state.get("review_status") == "pending_approval"
    )



def is_approved(state: AgenticState) -> bool:
    """Determine whether the workflow has been approved for publishing."""
    return (
        state.get("review_status") == "approved"
        or state.get("approval_status") is ApprovalStatus.APPROVED
    )



def is_rejected(state: AgenticState) -> bool:
    """Determine whether the workflow has been rejected by a reviewer."""
    return (
        state.get("review_status") == "rejected"
        or state.get("approval_status") is ApprovalStatus.REJECTED
    )



def has_validation_failures(state: AgenticState) -> bool:
    """Determine whether validation produced blocking failures."""
    return (
        state.get("validation_outcome") == "failed"
        or bool(state.get("validation_issues"))
    )



def has_generated_code(state: AgenticState) -> bool:
    """Determine whether the workflow contains generated code files."""
    return bool(state.get("generated_code_files"))



def has_retrieved_context(state: AgenticState) -> bool:
    """Determine whether the workflow contains retrieved design context."""
    return bool(state.get("vector_context_matches"))



def is_cancel_requested(state: AgenticState) -> bool:
    """Determine whether workflow cancellation has been requested."""
    return bool(state.get("cancel_requested", False))



def is_pause_requested(state: AgenticState) -> bool:
    """Determine whether workflow pause has been requested."""
    return bool(state.get("pause_requested", False))



def has_active_interrupt(state: AgenticState) -> bool:
    """Determine whether the workflow has an active human-review interrupt."""
    current_interrupt = state.get("current_interrupt")


    if not isinstance(current_interrupt, dict):
        return False


    return current_interrupt.get("status") == "pending_review"



# ---------------------------------------------------------------------------
# Count and serialization helpers
# ---------------------------------------------------------------------------



def get_generated_file_count(state: AgenticState) -> int:
    """Return the number of generated code files in state."""
    return len(state.get("generated_code_files", {}))



def get_requirement_count(state: AgenticState) -> int:
    """Return the number of extracted requirements in state."""
    return len(state.get("extracted_requirements", []))



def get_context_match_count(state: AgenticState) -> int:
    """Return the number of retrieved context matches in state."""
    return len(state.get("vector_context_matches", []))



def get_artifact_count(state: AgenticState) -> int:
    """Return the number of tracked workflow artifacts."""
    return len(state.get("artifacts", []))



def get_citation_count(state: AgenticState) -> int:
    """Return the number of retrieval citations in state."""
    return len(state.get("citations", []))



def get_interrupt_count(state: AgenticState) -> int:
    """Return the number of recorded workflow interrupts."""
    return len(state.get("interrupt_history", []))



def get_total_token_usage(state: AgenticState) -> int:
    """Return total recorded provider token usage."""
    return int(state.get("provider_token_usage", {}).get("total_tokens", 0))



def serialize_state(state: AgenticState) -> str:
    """
    Serialize workflow state to JSON for logging, checkpoints, or diagnostics.


    Never send the serialized state to an external provider or log it in full
    if it may contain sensitive customer content.
    """
    return json.dumps(
        dict(state),
        default=str,
        ensure_ascii=False,
    )



def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()