"""
Comprehensive tests for workflow graph state transitions.

These tests cover initial-state construction and validation, execution-log
append behavior, step transitions, error and retry lifecycle, provider usage
accounting, artifacts and citations, generated-file limits, validation
results, approval predicates, terminal states, and JSON serialization.
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest

from app.core.enums import ApprovalStatus, JobStatus, WorkflowStep
from app.core.exceptions import ValidationError
from app.graph.state import (
    add_artifact,
    add_citation,
    add_provider_usage,
    append_execution_log,
    can_retry,
    clear_error,
    create_initial_state,
    get_artifact_count,
    get_citation_count,
    get_generated_file_count,
    get_requirement_count,
    get_total_token_usage,
    has_generated_code,
    has_retrieved_context,
    has_validation_failures,
    increment_retry_count,
    is_approved,
    is_rejected,
    is_terminal,
    requires_approval,
    serialize_state,
    set_current_step,
    set_error,
    set_generated_files,
    set_validation_result,
    set_workflow_status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state():
    return create_initial_state(
        execution_id="wf_123",
        brief_id="brief_456",
        raw_brief_text="Build an internal dashboard.",
        tenant_id="tenant_789",
        user_id="user_001",
        approval_required=True,
    )


@pytest.fixture
def recoverable_error_state(state):
    return set_error(
        state,
        message="Provider timeout.",
        error_step=WorkflowStep.DESIGN,
        error_code="provider_timeout",
        recoverable=True,
    )


# ---------------------------------------------------------------------------
# Initial state creation
# ---------------------------------------------------------------------------


def test_create_initial_state(state):
    assert state["execution_id"] == "wf_123"
    assert state["workflow_status"] == JobStatus.QUEUED
    assert state["current_step"] == WorkflowStep.INITIALIZED
    assert state["approval_required"] is True
    assert state["execution_logs"]


def test_create_initial_state_sets_core_identity(state):
    assert state["brief_id"] == "brief_456"
    assert state["tenant_id"] == "tenant_789"
    assert state["user_id"] == "user_001"
    assert (
        state["raw_brief_text"]
        == "Build an internal dashboard."
    )


def test_create_initial_state_sets_defaults(state):
    assert state["retry_count"] == 0
    assert state["max_retries"] == 3
    assert state["current_error"] is None
    assert state["recoverable_error"] is False
    assert state["cancel_requested"] is False
    assert state["artifacts"] == []
    assert state["citations"] == []
    assert state["generated_files"] == {}
    assert state["provider_usage"] == {}
    assert state["interrupt_history"] == []
    assert state["current_interrupt"] is None


def test_initial_state_validation():
    with pytest.raises(ValueError):
        create_initial_state(
            execution_id="",
            brief_id="brief_456",
            raw_brief_text="Build an app.",
        )


@pytest.mark.parametrize(
    "invalid_field",
    [
        "execution_id",
        "brief_id",
        "raw_brief_text",
        "tenant_id",
        "user_id",
    ],
)
def test_initial_state_rejects_missing_required_fields(
    invalid_field: str,
):
    valid_values = {
        "execution_id": "wf_123",
        "brief_id": "brief_456",
        "raw_brief_text": "Build an app.",
        "tenant_id": "tenant_789",
        "user_id": "user_001",
    }

    invalid_values = {
        **valid_values,
        invalid_field: "",
    }

    with pytest.raises(ValueError):
        create_initial_state(
            **invalid_values
        )


def test_initial_state_does_not_mutate_inputs():
    execution_id = "wf_123"

    state = create_initial_state(
        execution_id=execution_id,
        brief_id="brief_456",
        raw_brief_text="Build an app.",
    )

    state["execution_id"] = "changed"

    assert execution_id == "wf_123"


# ---------------------------------------------------------------------------
# Execution logs
# ---------------------------------------------------------------------------


def test_append_execution_log_returns_only_new_entry(state):
    updated = append_execution_log(
        state,
        "Ingestion completed.",
    )

    assert updated["execution_logs"] == [
        "INFO: Ingestion completed."
    ]


def test_append_execution_log_does_not_mutate_original(state):
    original_log_count = len(
        state["execution_logs"]
    )

    append_execution_log(
        state,
        "Ingestion completed.",
    )

    assert (
        len(state["execution_logs"])
        == original_log_count
    )


def test_append_execution_log_accumulates(state):
    first = append_execution_log(
        state,
        "Ingestion completed.",
    )

    second = append_execution_log(
        first,
        "Requirements extracted.",
    )

    assert len(
        second["execution_logs"]
    ) == 2
    assert (
        second["execution_logs"][0]
        == "INFO: Ingestion completed."
    )
    assert (
        second["execution_logs"][1]
        == "INFO: Requirements extracted."
    )


def test_append_execution_log_rejects_empty_message(state):
    with pytest.raises(ValueError):
        append_execution_log(state, "")


def test_append_execution_log_rejects_whitespace_message(state):
    with pytest.raises(ValueError):
        append_execution_log(state, "   ")


def test_append_execution_log_bounds_log_count(state):
    updated = state

    for index in range(150):
        updated = append_execution_log(
            updated,
            f"Log entry {index}",
        )

    assert (
        len(updated["execution_logs"])
        <= 100
    )


# ---------------------------------------------------------------------------
# Step transitions
# ---------------------------------------------------------------------------


def test_step_transition_tracks_previous_step(state):
    updated = set_current_step(
        state,
        WorkflowStep.INGESTION,
    )

    assert (
        updated["previous_step"]
        == WorkflowStep.INITIALIZED
    )
    assert (
        updated["current_step"]
        == WorkflowStep.INGESTION
    )


def test_step_transition_does_not_mutate_original(state):
    set_current_step(
        state,
        WorkflowStep.INGESTION,
    )

    assert (
        state["current_step"]
        == WorkflowStep.INITIALIZED
    )


def test_step_transition_rejects_invalid_step(state):
    with pytest.raises(ValueError):
        set_current_step(
            state,
            "not-a-workflow-step",
        )


def test_step_transition_rejects_none_step(state):
    with pytest.raises(ValueError):
        set_current_step(
            state,
            None,
        )


def test_step_transition_supports_sequential_steps(state):
    ingestion_state = set_current_step(
        state,
        WorkflowStep.INGESTION,
    )

    requirements_state = set_current_step(
        ingestion_state,
        WorkflowStep.REQUIREMENTS,
    )

    assert (
        requirements_state["previous_step"]
        == WorkflowStep.INGESTION
    )
    assert (
        requirements_state["current_step"]
        == WorkflowStep.REQUIREMENTS
    )


# ---------------------------------------------------------------------------
# Error and retry lifecycle
# ---------------------------------------------------------------------------


def test_error_and_recovery(state):
    errored = set_error(
        state,
        message="Provider timeout.",
        error_step=WorkflowStep.DESIGN,
        error_code="provider_timeout",
        recoverable=True,
    )

    assert (
        errored["workflow_status"]
        == JobStatus.FAILED
    )
    assert can_retry(errored) is True

    retried = increment_retry_count(
        errored
    )

    assert retried["retry_count"] == 1

    cleared = clear_error(retried)

    assert (
        cleared["current_error"]
        is None
    )


def test_set_error_records_error_context(state):
    errored = set_error(
        state,
        message="Provider timeout.",
        error_step=WorkflowStep.DESIGN,
        error_code="provider_timeout",
        recoverable=True,
    )

    assert (
        errored["current_error"]
        == "Provider timeout."
    )
    assert (
        errored["error_step"]
        == WorkflowStep.DESIGN
    )
    assert (
        errored["error_code"]
        == "provider_timeout"
    )
    assert (
        errored["recoverable_error"]
        is True
    )


def test_set_error_rejects_empty_message(state):
    with pytest.raises(ValueError):
        set_error(
            state,
            message="",
            error_step=WorkflowStep.DESIGN,
        )


def test_set_error_rejects_whitespace_message(state):
    with pytest.raises(ValueError):
        set_error(
            state,
            message="   ",
            error_step=WorkflowStep.DESIGN,
        )


def test_set_error_rejects_none_message(state):
    with pytest.raises(ValueError):
        set_error(
            state,
            message=None,
            error_step=WorkflowStep.DESIGN,
        )


def test_clear_error_resets_error_metadata(
    recoverable_error_state,
):
    cleared = clear_error(
        recoverable_error_state
    )

    assert (
        cleared["current_error"]
        is None
    )
    assert (
        cleared["error_step"]
        is None
    )
    assert (
        cleared["error_code"]
        is None
    )
    assert (
        cleared["recoverable_error"]
        is False
    )


def test_retry_allowance(state):
    with pytest.raises(ValueError):
        state_with_no_retries = {
            **state,
            "retry_count": 3,
            "max_retries": 3,
        }

        increment_retry_count(
            state_with_no_retries
        )


def test_increment_retry_count_accumulates(state):
    first = increment_retry_count(state)
    second = increment_retry_count(first)

    assert first["retry_count"] == 1
    assert second["retry_count"] == 2


def test_increment_retry_count_does_not_mutate_original(state):
    increment_retry_count(state)

    assert state["retry_count"] == 0


def test_can_retry_rejects_non_recoverable_error(state):
    errored = set_error(
        state,
        message="Invalid workflow state.",
        error_step=WorkflowStep.VALIDATION,
        error_code="invalid_state",
        recoverable=False,
    )

    assert can_retry(errored) is False


def test_can_retry_rejects_cancelled_workflow(state):
    cancelled_state = {
        **state,
        "cancel_requested": True,
    }

    assert (
        can_retry(cancelled_state)
        is False
    )


# ---------------------------------------------------------------------------
# Provider usage
# ---------------------------------------------------------------------------


def test_provider_usage_accumulates(state):
    once = add_provider_usage(
        state,
        provider_name="openai",
        provider_model="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
    )

    twice = add_provider_usage(
        once,
        provider_name="openai",
        provider_model="gpt-4o-mini",
        prompt_tokens=20,
        completion_tokens=30,
    )

    assert (
        get_total_token_usage(twice)
        == 200
    )


def test_provider_usage_tracks_prompt_and_completion_tokens(state):
    updated = add_provider_usage(
        state,
        provider_name="openai",
        provider_model="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
    )

    usage = updated["provider_usage"][
        "openai:gpt-4o-mini"
    ]

    assert usage["prompt_tokens"] == 100
    assert (
        usage["completion_tokens"] == 50
    )
    assert usage["total_tokens"] == 150


def test_provider_usage_separates_models(state):
    mini_usage = add_provider_usage(
        state,
        provider_name="openai",
        provider_model="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
    )

    full_usage = add_provider_usage(
        mini_usage,
        provider_name="openai",
        provider_model="gpt-4o",
        prompt_tokens=200,
        completion_tokens=100,
    )

    assert (
        get_total_token_usage(full_usage)
        == 450
    )


def test_provider_usage_rejects_negative_tokens(state):
    with pytest.raises(ValueError):
        add_provider_usage(
            state,
            provider_name="openai",
            provider_model="gpt-4o-mini",
            prompt_tokens=-1,
            completion_tokens=0,
        )


def test_provider_usage_rejects_empty_provider(state):
    with pytest.raises(ValueError):
        add_provider_usage(
            state,
            provider_name="",
            provider_model="gpt-4o-mini",
            prompt_tokens=1,
            completion_tokens=1,
        )


def test_provider_usage_rejects_empty_model(state):
    with pytest.raises(ValueError):
        add_provider_usage(
            state,
            provider_name="openai",
            provider_model="",
            prompt_tokens=1,
            completion_tokens=1,
        )


# ---------------------------------------------------------------------------
# Artifacts and citations
# ---------------------------------------------------------------------------


def test_artifacts_and_citations(state):
    with_artifact = add_artifact(
        state,
        artifact_id="artifact_1",
        artifact_type="code_bundle",
        uri="s3://bucket/artifact_1.zip",
    )

    with_citation = add_citation(
        with_artifact,
        document_id="doc_123",
        embedding_id="emb_456",
        similarity=0.92,
    )

    assert (
        get_artifact_count(with_citation)
        == 1
    )
    assert (
        get_citation_count(with_citation)
        == 1
    )


def test_add_artifact_records_metadata(state):
    updated = add_artifact(
        state,
        artifact_id="artifact_1",
        artifact_type="code_bundle",
        uri="s3://bucket/artifact_1.zip",
    )

    artifact = updated["artifacts"][0]

    assert (
        artifact["artifact_id"]
        == "artifact_1"
    )
    assert (
        artifact["artifact_type"]
        == "code_bundle"
    )
    assert (
        artifact["uri"]
        == "s3://bucket/artifact_1.zip"
    )


def test_add_artifact_rejects_empty_id(state):
    with pytest.raises(ValueError):
        add_artifact(
            state,
            artifact_id="",
            artifact_type="code_bundle",
            uri="s3://bucket/artifact_1.zip",
        )


def test_add_artifact_rejects_empty_uri(state):
    with pytest.raises(ValueError):
        add_artifact(
            state,
            artifact_id="artifact_1",
            artifact_type="code_bundle",
            uri="",
        )


def test_add_citation_records_similarity(state):
    updated = add_citation(
        state,
        document_id="doc_123",
        embedding_id="emb_456",
        similarity=0.92,
    )

    citation = updated["citations"][0]

    assert (
        citation["document_id"]
        == "doc_123"
    )
    assert (
        citation["embedding_id"]
        == "emb_456"
    )
    assert citation["similarity"] == 0.92


def test_add_citation_rejects_invalid_similarity(state):
    with pytest.raises(ValueError):
        add_citation(
            state,
            document_id="doc_123",
            embedding_id="emb_456",
            similarity=1.5,
        )


def test_add_citation_rejects_empty_document_id(state):
    with pytest.raises(ValueError):
        add_citation(
            state,
            document_id="",
            embedding_id="emb_456",
            similarity=0.9,
        )


# ---------------------------------------------------------------------------
# Generated files
# ---------------------------------------------------------------------------


def test_generated_files_limit(state):
    files = {
        f"file_{index}.py": "print('hello')"
        for index in range(101)
    }

    with pytest.raises(ValueError):
        set_generated_files(
            state,
            files,
        )


def test_set_generated_files_stores_files(state):
    files = {
        "dashboard.py": "print('hello')",
        "README.md": "# Dashboard",
    }

    updated = set_generated_files(
        state,
        files,
    )

    assert (
        updated["generated_files"]
        == files
    )
    assert (
        get_generated_file_count(
            updated
        )
        == 2
    )


def test_set_generated_files_rejects_empty_file_name(state):
    with pytest.raises(ValueError):
        set_generated_files(
            state,
            {"": "print('hello')"},
        )


def test_set_generated_files_rejects_empty_content(state):
    with pytest.raises(ValueError):
        set_generated_files(
            state,
            {"dashboard.py": ""},
        )


def test_set_generated_files_rejects_whitespace_content(state):
    with pytest.raises(ValueError):
        set_generated_files(
            state,
            {"dashboard.py": "   "},
        )


def test_has_generated_code_false_initially(state):
    assert (
        has_generated_code(state)
        is False
    )


def test_has_generated_code_true_after_files(state):
    updated = set_generated_files(
        state,
        {"dashboard.py": "print('hello')"},
    )

    assert (
        has_generated_code(updated)
        is True
    )


# ---------------------------------------------------------------------------
# Requirements and validation
# ---------------------------------------------------------------------------


def test_get_requirement_count_empty(state):
    assert (
        get_requirement_count(state)
        == 0
    )


def test_get_requirement_count_counts_requirements(state):
    updated = {
        **state,
        "extracted_requirements": [
            "Build a dashboard",
            "Show key metrics",
        ],
    }

    assert (
        get_requirement_count(updated)
        == 2
    )


def test_validation_result(state):
    validated = set_validation_result(
        state,
        outcome="failed",
        score=0.4,
        issues=[
            {
                "severity": "error",
                "message": (
                    "Missing accessibility label."
                ),
            }
        ],
        static_analysis_passed=False,
    )

    assert (
        has_validation_failures(validated)
        is True
    )
    assert (
        validated["validation_status"]
        == JobStatus.COMPLETED
    )


def test_set_validation_result_records_outcome(state):
    validated = set_validation_result(
        state,
        outcome="passed",
        score=0.95,
        issues=[],
        static_analysis_passed=True,
    )

    assert (
        validated["validation_outcome"]
        == "passed"
    )
    assert (
        validated["validation_score"]
        == 0.95
    )
    assert (
        validated["validation_issues"]
        == []
    )
    assert (
        validated[
            "static_analysis_passed"
        ]
        is True
    )


def test_set_validation_result_rejects_invalid_outcome(state):
    with pytest.raises(ValueError):
        set_validation_result(
            state,
            outcome="unknown",
            score=0.9,
            issues=[],
            static_analysis_passed=True,
        )


def test_set_validation_result_rejects_invalid_score(state):
    with pytest.raises(ValueError):
        set_validation_result(
            state,
            outcome="passed",
            score=1.5,
            issues=[],
            static_analysis_passed=True,
        )


def test_has_validation_failures_false_when_passed(state):
    validated = set_validation_result(
        state,
        outcome="passed",
        score=0.95,
        issues=[],
        static_analysis_passed=True,
    )

    assert (
        has_validation_failures(validated)
        is False
    )


# ---------------------------------------------------------------------------
# Retrieved context
# ---------------------------------------------------------------------------


def test_has_retrieved_context_false_initially(state):
    assert (
        has_retrieved_context(state)
        is False
    )


def test_has_retrieved_context_true_after_citation(state):
    updated = add_citation(
        state,
        document_id="doc_123",
        embedding_id="emb_456",
        similarity=0.92,
    )

    assert (
        has_retrieved_context(updated)
        is True
    )


# ---------------------------------------------------------------------------
# Approval predicates
# ---------------------------------------------------------------------------


def test_state_predicates(state):
    assert requires_approval(state) is True
    assert has_generated_code(state) is False
    assert has_retrieved_context(state) is False


    approved_state = {
        **state,
        "review_status": "approved",
        "approval_status": ApprovalStatus.APPROVED,
    }


    assert is_approved(approved_state) is True
    assert is_rejected(approved_state) is False


def test_requires_approval_false_when_not_configured(state):
    updated = {
        **state,
        "approval_required": False,
    }

    assert (
        requires_approval(updated)
        is False
    )


def test_is_approved_false_initially(state):
    assert is_approved(state) is False


def test_is_rejected_false_initially(state):
    assert is_rejected(state) is False


def test_is_rejected_true_after_rejection(state):
    rejected_state = {
        **state,
        "review_status": "rejected",
        "approval_status": ApprovalStatus.REJECTED,
    }

    assert (
        is_rejected(rejected_state)
        is True
    )
    assert (
        is_approved(rejected_state)
        is False
    )


# ---------------------------------------------------------------------------
# Workflow status and terminal states
# ---------------------------------------------------------------------------


def test_terminal_state(state):
    completed = set_workflow_status(
        state,
        JobStatus.COMPLETED,
    )

    assert is_terminal(completed) is True


def test_set_workflow_status_updates_status(state):
    updated = set_workflow_status(
        state,
        JobStatus.PROCESSING,
    )

    assert (
        updated["workflow_status"]
        == JobStatus.PROCESSING
    )


def test_set_workflow_status_rejects_invalid_status(state):
    with pytest.raises(ValueError):
        set_workflow_status(
            state,
            "not-a-status",
        )


def test_terminal_states_are_recognized(state):
    assert (
        is_terminal(
            set_workflow_status(
                state,
                JobStatus.COMPLETED,
            )
        )
        is True
    )
    assert (
        is_terminal(
            set_workflow_status(
                state,
                JobStatus.FAILED,
            )
        )
        is True
    )
    assert (
        is_terminal(
            set_workflow_status(
                state,
                JobStatus.CANCELLED,
            )
        )
        is True
    )


def test_non_terminal_states_are_not_terminal(state):
    assert is_terminal(state) is False
    assert (
        is_terminal(
            set_workflow_status(
                state,
                JobStatus.QUEUED,
            )
        )
        is False
    )
    assert (
        is_terminal(
            set_workflow_status(
                state,
                JobStatus.PROCESSING,
            )
        )
        is False
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_serialize_state(state):
    serialized = serialize_state(state)

    assert isinstance(serialized, str)
    assert (
        json.loads(serialized)[
            "execution_id"
        ]
        == "wf_123"
    )


def test_serialize_state_preserves_enums(state):
    updated = set_current_step(
        state,
        WorkflowStep.INGESTION,
    )

    serialized = serialize_state(updated)
    parsed = json.loads(serialized)

    assert (
        parsed["current_step"]
        == WorkflowStep.INGESTION.value
    )


def test_serialize_state_is_json_safe(state):
    updated = add_provider_usage(
        state,
        provider_name="openai",
        provider_model="gpt-4o-mini",
        prompt_tokens=10,
        completion_tokens=5,
    )

    serialized = serialize_state(updated)

    assert isinstance(
        json.loads(serialized),
        dict,
    )


def test_serialize_state_does_not_mutate_original(state):
    serialize_state(state)

    assert (
        state["workflow_status"]
        == JobStatus.QUEUED
    )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    asyncio.run(
        pytest.main(
            [
                __file__,
                "-v",
            ]
        )
    )
