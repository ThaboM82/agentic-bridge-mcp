"""
Comprehensive tests for AgenticBridge core enums.

These tests cover environment classification, job lifecycle states, workflow
steps, interrupts, recovery actions, approval and review states, validation
outcomes, quality gates, error codes, user roles, telemetry events, workflow
control actions, and workflow triggers.
"""

from __future__ import annotations

import pytest

from app.core.enums import (
    ApprovalDecision,
    ApprovalStatus,
    DocumentStatus,
    Environment,
    ErrorCode,
    IngestionJobStatus,
    InterruptStatus,
    InterruptType,
    JobStatus,
    QualityGateStatus,
    RecoveryAction,
    ReviewStatus,
    TelemetryEvent,
    UserRole,
    ValidationOutcome,
    WorkflowControlAction,
    WorkflowStep,
    WorkflowTrigger,
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_environment_values():
    assert Environment.LOCAL.value == "local"
    assert Environment.TEST.value == "test"
    assert Environment.DEVELOPMENT.value == "development"
    assert Environment.STAGING.value == "staging"
    assert Environment.PRODUCTION.value == "production"


def test_environment_properties():
    assert Environment.PRODUCTION.is_production
    assert Environment.LOCAL.is_development
    assert Environment.TEST.is_test
    assert Environment.PRODUCTION.requires_strict_security


def test_environment_development_properties():
    assert Environment.DEVELOPMENT.is_development
    assert not Environment.DEVELOPMENT.is_production


def test_environment_staging_is_not_production():
    assert not Environment.STAGING.is_production


def test_environment_requires_strict_security():
    assert Environment.PRODUCTION.requires_strict_security
    assert Environment.STAGING.requires_strict_security
    assert not Environment.LOCAL.requires_strict_security


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------


def test_job_status_values():
    assert JobStatus.PENDING.value == "pending"
    assert JobStatus.QUEUED.value == "queued"
    assert JobStatus.PROCESSING.value == "processing"
    assert JobStatus.COMPLETED.value == "completed"
    assert JobStatus.FAILED.value == "failed"
    assert JobStatus.TIMED_OUT.value == "timed_out"
    assert JobStatus.CANCELLED.value == "cancelled"


def test_job_status_properties():
    assert JobStatus.QUEUED.is_active
    assert JobStatus.PROCESSING.is_active
    assert JobStatus.COMPLETED.is_successful
    assert JobStatus.FAILED.is_retryable
    assert JobStatus.TIMED_OUT.is_retryable
    assert JobStatus.CANCELLED.is_terminal


def test_job_status_pending_is_not_active():
    assert not JobStatus.PENDING.is_active


def test_job_status_completed_is_terminal():
    assert JobStatus.COMPLETED.is_terminal


def test_job_status_failed_is_terminal():
    assert JobStatus.FAILED.is_terminal


def test_job_status_processing_is_not_terminal():
    assert not JobStatus.PROCESSING.is_terminal


def test_job_status_collections():
    assert JobStatus.QUEUED in JobStatus.active_statuses()
    assert JobStatus.TIMED_OUT in JobStatus.terminal_statuses()


def test_job_status_active_collection():
    active_statuses = JobStatus.active_statuses()

    assert JobStatus.QUEUED in active_statuses
    assert JobStatus.PROCESSING in active_statuses
    assert JobStatus.COMPLETED not in active_statuses
    assert JobStatus.FAILED not in active_statuses


def test_job_status_terminal_collection():
    terminal_statuses = JobStatus.terminal_statuses()

    assert JobStatus.COMPLETED in terminal_statuses
    assert JobStatus.FAILED in terminal_statuses
    assert JobStatus.TIMED_OUT in terminal_statuses
    assert JobStatus.CANCELLED in terminal_statuses
    assert JobStatus.PROCESSING not in terminal_statuses


def test_job_status_retryable_collection():
    retryable_statuses = JobStatus.retryable_statuses()

    assert JobStatus.FAILED in retryable_statuses
    assert JobStatus.TIMED_OUT in retryable_statuses
    assert JobStatus.COMPLETED not in retryable_statuses


# ---------------------------------------------------------------------------
# Ingestion job status
# ---------------------------------------------------------------------------


def test_ingestion_job_status_values():
    assert IngestionJobStatus.QUEUED.value == "queued"
    assert IngestionJobStatus.PROCESSING.value == "processing"
    assert IngestionJobStatus.COMPLETED.value == "completed"
    assert IngestionJobStatus.FAILED.value == "failed"
    assert IngestionJobStatus.PARTIALLY_COMPLETED.value == "partially_completed"
    assert IngestionJobStatus.CANCELLED.value == "cancelled"


def test_ingestion_job_status_terminal_states():
    assert IngestionJobStatus.COMPLETED.is_terminal
    assert IngestionJobStatus.FAILED.is_terminal
    assert IngestionJobStatus.CANCELLED.is_terminal
    assert IngestionJobStatus.PARTIALLY_COMPLETED.is_terminal


def test_ingestion_job_status_active_states():
    assert IngestionJobStatus.QUEUED.is_active
    assert IngestionJobStatus.PROCESSING.is_active
    assert not IngestionJobStatus.COMPLETED.is_active


# ---------------------------------------------------------------------------
# Document status
# ---------------------------------------------------------------------------


def test_document_status_values():
    assert DocumentStatus.UPLOADED.value == "uploaded"
    assert DocumentStatus.PROCESSING.value == "processing"
    assert DocumentStatus.PROCESSED.value == "processed"
    assert DocumentStatus.FAILED.value == "failed"
    assert DocumentStatus.ARCHIVED.value == "archived"


def test_document_status_terminal_states():
    assert DocumentStatus.PROCESSED.is_terminal
    assert DocumentStatus.FAILED.is_terminal
    assert DocumentStatus.ARCHIVED.is_terminal


def test_document_status_active_states():
    assert DocumentStatus.UPLOADED.is_active
    assert DocumentStatus.PROCESSING.is_active
    assert not DocumentStatus.PROCESSED.is_active


# ---------------------------------------------------------------------------
# Workflow steps
# ---------------------------------------------------------------------------


def test_workflow_step_values():
    assert WorkflowStep.INITIALIZED.value == "initialized"
    assert WorkflowStep.INTAKE.value == "intake"
    assert WorkflowStep.INGESTION.value == "ingestion"
    assert WorkflowStep.RETRIEVAL.value == "retrieval"
    assert WorkflowStep.REQUIREMENTS.value == "requirements"
    assert WorkflowStep.DESIGN.value == "design"
    assert WorkflowStep.CODE_GENERATION.value == "code_generation"
    assert WorkflowStep.VALIDATION.value == "validation"
    assert WorkflowStep.REVIEW.value == "review"
    assert WorkflowStep.PUBLISHING.value == "publishing"
    assert WorkflowStep.COMPLETED.value == "completed"
    assert WorkflowStep.ERROR.value == "error"
    assert WorkflowStep.CANCELLED.value == "cancelled"
    assert WorkflowStep.PAUSED.value == "paused"


def test_workflow_step_order():
    ordered_steps = WorkflowStep.ordered_steps()

    assert ordered_steps[0] == WorkflowStep.INITIALIZED
    assert WorkflowStep.RETRIEVAL in ordered_steps
    assert ordered_steps[-1] == WorkflowStep.COMPLETED


def test_workflow_step_order_is_ordered():
    ordered_steps = WorkflowStep.ordered_steps()

    assert ordered_steps.index(
        WorkflowStep.INITIALIZED
    ) < ordered_steps.index(
        WorkflowStep.INGESTION
    )

    assert ordered_steps.index(
        WorkflowStep.INGESTION
    ) < ordered_steps.index(
        WorkflowStep.RETRIEVAL
    )

    assert ordered_steps.index(
        WorkflowStep.RETRIEVAL
    ) < ordered_steps.index(
        WorkflowStep.REQUIREMENTS
    )

    assert ordered_steps.index(
        WorkflowStep.REQUIREMENTS
    ) < ordered_steps.index(
        WorkflowStep.DESIGN
    )

    assert ordered_steps.index(
        WorkflowStep.DESIGN
    ) < ordered_steps.index(
        WorkflowStep.CODE_GENERATION
    )

    assert ordered_steps.index(
        WorkflowStep.CODE_GENERATION
    ) < ordered_steps.index(
        WorkflowStep.VALIDATION
    )

    assert ordered_steps.index(
        WorkflowStep.VALIDATION
    ) < ordered_steps.index(
        WorkflowStep.REVIEW
    )

    assert ordered_steps.index(
        WorkflowStep.REVIEW
    ) < ordered_steps.index(
        WorkflowStep.PUBLISHING
    )


def test_workflow_step_properties():
    assert WorkflowStep.COMPLETED.is_terminal
    assert WorkflowStep.ERROR.is_failed
    assert WorkflowStep.CANCELLED.is_cancelled
    assert WorkflowStep.PAUSED.is_paused
    assert WorkflowStep.REVIEW.requires_human_review


def test_workflow_step_terminal_states():
    assert WorkflowStep.COMPLETED.is_terminal
    assert WorkflowStep.ERROR.is_terminal
    assert WorkflowStep.CANCELLED.is_terminal


def test_workflow_step_failed_states():
    assert WorkflowStep.ERROR.is_failed
    assert not WorkflowStep.COMPLETED.is_failed


def test_workflow_step_human_review_states():
    assert WorkflowStep.REVIEW.requires_human_review
    assert not WorkflowStep.VALIDATION.requires_human_review


def test_workflow_retryable_steps():
    retryable_steps = WorkflowStep.retryable_steps()

    assert WorkflowStep.RETRIEVAL in retryable_steps
    assert WorkflowStep.VALIDATION in retryable_steps
    assert WorkflowStep.COMPLETED not in retryable_steps


def test_workflow_step_retryable_collection():
    retryable_steps = WorkflowStep.retryable_steps()

    assert WorkflowStep.INGESTION in retryable_steps
    assert WorkflowStep.REQUIREMENTS in retryable_steps
    assert WorkflowStep.DESIGN in retryable_steps
    assert WorkflowStep.CODE_GENERATION in retryable_steps
    assert WorkflowStep.PUBLISHING not in retryable_steps


# ---------------------------------------------------------------------------
# Interrupts
# ---------------------------------------------------------------------------


def test_interrupt_types():
    assert InterruptType.APPROVAL_REQUIRED.requires_approval
    assert InterruptType.HUMAN_REVIEW.requires_approval is False


def test_interrupt_type_values():
    assert (
        InterruptType.APPROVAL_REQUIRED.value
        == "approval_required"
    )
    assert (
        InterruptType.HUMAN_REVIEW.value
        == "human_review"
    )


def test_interrupt_statuses():
    assert InterruptStatus.PENDING_REVIEW.is_pending
    assert InterruptStatus.RESOLVED.is_terminal


def test_interrupt_status_values():
    assert (
        InterruptStatus.PENDING_REVIEW.value
        == "pending_review"
    )
    assert (
        InterruptStatus.RESOLVED.value
        == "resolved"
    )


def test_interrupt_status_terminal_collection():
    terminal_statuses = InterruptStatus.terminal_statuses()

    assert InterruptStatus.RESOLVED in terminal_statuses
    assert InterruptStatus.PENDING_REVIEW not in terminal_statuses


# ---------------------------------------------------------------------------
# Recovery actions
# ---------------------------------------------------------------------------


def test_recovery_actions():
    assert RecoveryAction.RETRY.value == "retry"
    assert RecoveryAction.HUMAN_REVIEW.value == "human_review"


def test_recovery_action_values():
    assert RecoveryAction.CANCEL.value == "cancel"
    assert RecoveryAction.FAIL.value == "fail"


def test_recovery_action_retry_is_automated():
    assert RecoveryAction.RETRY.is_automated


def test_recovery_action_human_review_requires_human():
    assert RecoveryAction.HUMAN_REVIEW.requires_human


# ---------------------------------------------------------------------------
# Approval statuses
# ---------------------------------------------------------------------------


def test_approval_statuses():
    assert ApprovalStatus.PENDING.is_pending
    assert ApprovalStatus.APPROVED.allows_publication
    assert ApprovalStatus.REJECTED.is_decided


def test_approval_status_values():
    assert ApprovalStatus.PENDING.value == "pending"
    assert ApprovalStatus.APPROVED.value == "approved"
    assert ApprovalStatus.REJECTED.value == "rejected"


def test_approval_status_decided_states():
    assert ApprovalStatus.APPROVED.is_decided
    assert ApprovalStatus.REJECTED.is_decided
    assert not ApprovalStatus.PENDING.is_decided


def test_approval_status_publication_permissions():
    assert ApprovalStatus.APPROVED.allows_publication
    assert not ApprovalStatus.PENDING.allows_publication
    assert not ApprovalStatus.REJECTED.allows_publication


# ---------------------------------------------------------------------------
# Approval decisions
# ---------------------------------------------------------------------------


def test_approval_decisions():
    assert ApprovalDecision.APPROVED.allows_publication
    assert ApprovalDecision.REQUEST_CHANGES.is_recoverable
    assert ApprovalDecision.REJECTED.is_recoverable is False


def test_approval_decision_values():
    assert ApprovalDecision.APPROVED.value == "approve"
    assert ApprovalDecision.REJECTED.value == "reject"
    assert (
        ApprovalDecision.REQUEST_CHANGES.value
        == "request_changes"
    )


def test_approval_decision_publication_permissions():
    assert ApprovalDecision.APPROVED.allows_publication
    assert not ApprovalDecision.REJECTED.allows_publication
    assert not ApprovalDecision.REQUEST_CHANGES.allows_publication


# ---------------------------------------------------------------------------
# Review statuses
# ---------------------------------------------------------------------------


def test_review_statuses():
    assert ReviewStatus.PENDING_APPROVAL.is_pending
    assert ReviewStatus.APPROVED.allows_publication


def test_review_status_values():
    assert (
        ReviewStatus.PENDING_APPROVAL.value
        == "pending_approval"
    )
    assert ReviewStatus.APPROVED.value == "approved"
    assert ReviewStatus.REJECTED.value == "rejected"
    assert (
        ReviewStatus.CHANGES_REQUESTED.value
        == "changes_requested"
    )


def test_review_status_publication_permissions():
    assert ReviewStatus.APPROVED.allows_publication
    assert not ReviewStatus.PENDING_APPROVAL.allows_publication
    assert not ReviewStatus.REJECTED.allows_publication
    assert not ReviewStatus.CHANGES_REQUESTED.allows_publication


def test_review_status_decided_states():
    assert ReviewStatus.APPROVED.is_decided
    assert ReviewStatus.REJECTED.is_decided
    assert not ReviewStatus.PENDING_APPROVAL.is_decided


# ---------------------------------------------------------------------------
# Validation outcomes
# ---------------------------------------------------------------------------


def test_validation_outcomes():
    assert ValidationOutcome.PASSED.allows_review
    assert ValidationOutcome.WARNING.allows_review
    assert ValidationOutcome.FAILED.blocks_publication


def test_validation_outcome_values():
    assert ValidationOutcome.PASSED.value == "passed"
    assert ValidationOutcome.WARNING.value == "warning"
    assert ValidationOutcome.FAILED.value == "failed"


def test_validation_outcome_publication_permissions():
    assert not ValidationOutcome.PASSED.blocks_publication
    assert not ValidationOutcome.WARNING.blocks_publication
    assert ValidationOutcome.FAILED.blocks_publication


def test_validation_outcome_review_permissions():
    assert ValidationOutcome.PASSED.allows_review
    assert ValidationOutcome.WARNING.allows_review
    assert ValidationOutcome.FAILED.allows_review


# ---------------------------------------------------------------------------
# Quality gate statuses
# ---------------------------------------------------------------------------


def test_quality_gate_statuses():
    assert QualityGateStatus.PASSED.blocks_publication is False
    assert QualityGateStatus.FAILED.blocks_publication
    assert QualityGateStatus.NOT_RUN.blocks_publication


def test_quality_gate_status_values():
    assert QualityGateStatus.NOT_RUN.value == "not_run"
    assert QualityGateStatus.PASSED.value == "passed"
    assert QualityGateStatus.WARNING.value == "warning"
    assert QualityGateStatus.FAILED.value == "failed"


def test_quality_gate_warning_does_not_block_publication():
    assert not QualityGateStatus.WARNING.blocks_publication


# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------


def test_error_codes():
    assert ErrorCode.PROVIDER_TIMEOUT.is_retryable
    assert ErrorCode.VALIDATION_ERROR.is_client_error
    assert ErrorCode.INTERNAL_ERROR.is_client_error is False


def test_error_code_values():
    assert ErrorCode.VALIDATION_ERROR.value == "validation_error"
    assert ErrorCode.AUTHENTICATION_ERROR.value == "authentication_error"
    assert ErrorCode.AUTHORIZATION_ERROR.value == "authorization_error"
    assert ErrorCode.NOT_FOUND.value == "not_found"
    assert ErrorCode.RATE_LIMIT_EXCEEDED.value == "rate_limit_exceeded"
    assert ErrorCode.PROVIDER_TIMEOUT.value == "provider_timeout"
    assert ErrorCode.EXTERNAL_PROVIDER_ERROR.value == "external_provider_error"
    assert ErrorCode.WORKFLOW_ERROR.value == "workflow_error"
    assert ErrorCode.INTERNAL_ERROR.value == "internal_error"


def test_retryable_error_codes():
    assert ErrorCode.PROVIDER_TIMEOUT.is_retryable
    assert ErrorCode.EXTERNAL_PROVIDER_ERROR.is_retryable
    assert ErrorCode.RATE_LIMIT_EXCEEDED.is_retryable


def test_non_retryable_error_codes():
    assert not ErrorCode.VALIDATION_ERROR.is_retryable
    assert not ErrorCode.AUTHORIZATION_ERROR.is_retryable
    assert not ErrorCode.INTERNAL_ERROR.is_retryable


def test_client_error_codes():
    assert ErrorCode.VALIDATION_ERROR.is_client_error
    assert ErrorCode.AUTHENTICATION_ERROR.is_client_error
    assert ErrorCode.AUTHORIZATION_ERROR.is_client_error
    assert ErrorCode.NOT_FOUND.is_client_error
    assert ErrorCode.RATE_LIMIT_EXCEEDED.is_client_error


def test_server_error_codes():
    assert not ErrorCode.INTERNAL_ERROR.is_client_error
    assert not ErrorCode.EXTERNAL_PROVIDER_ERROR.is_client_error
    assert not ErrorCode.PROVIDER_TIMEOUT.is_client_error


# ---------------------------------------------------------------------------
# User roles
# ---------------------------------------------------------------------------


def test_user_roles():
    assert UserRole.ADMIN.can_approve_workflows
    assert UserRole.REVIEWER.can_approve_workflows
    assert UserRole.VIEWER.can_approve_workflows is False
    assert UserRole.DEVELOPER.can_manage_documents


def test_user_role_values():
    assert UserRole.ADMIN.value == "admin"
    assert UserRole.DEVELOPER.value == "developer"
    assert UserRole.REVIEWER.value == "reviewer"
    assert UserRole.VIEWER.value == "viewer"


def test_user_role_document_permissions():
    assert UserRole.ADMIN.can_manage_documents
    assert UserRole.DEVELOPER.can_manage_documents
    assert not UserRole.VIEWER.can_manage_documents


def test_user_role_workflow_permissions():
    assert UserRole.ADMIN.can_start_workflows
    assert UserRole.DEVELOPER.can_start_workflows
    assert not UserRole.VIEWER.can_start_workflows


def test_user_role_admin_has_all_permissions():
    assert UserRole.ADMIN.can_start_workflows
    assert UserRole.ADMIN.can_manage_documents
    assert UserRole.ADMIN.can_approve_workflows


# ---------------------------------------------------------------------------
# Telemetry events
# ---------------------------------------------------------------------------


def test_telemetry_events():
    assert TelemetryEvent.WORKFLOW_STARTED.value == "workflow_started"
    assert TelemetryEvent.PROVIDER_USAGE.value == "workflow_provider_usage"
    assert TelemetryEvent.HUMAN_REVIEW_RESOLVED.value == (
        "workflow_human_review_resolved"
    )


def test_telemetry_event_values():
    assert (
        TelemetryEvent.WORKFLOW_COMPLETED.value
        == "workflow_completed"
    )
    assert (
        TelemetryEvent.WORKFLOW_FAILED.value
        == "workflow_failed"
    )
    assert (
        TelemetryEvent.WORKFLOW_CANCELLED.value
        == "workflow_cancelled"
    )
    assert (
        TelemetryEvent.WORKFLOW_RETRY_SCHEDULED.value
        == "workflow_retry_scheduled"
    )
    assert (
        TelemetryEvent.HUMAN_REVIEW_REQUESTED.value
        == "workflow_human_review_requested"
    )


def test_telemetry_event_names_are_snake_case():
    for event in TelemetryEvent:
        assert event.value == event.value.lower()
        assert " " not in event.value


# ---------------------------------------------------------------------------
# Workflow control actions and triggers
# ---------------------------------------------------------------------------


def test_workflow_control_and_triggers():
    assert WorkflowControlAction.APPROVE.value == "approve"
    assert WorkflowTrigger.RETRY.value == "retry"


def test_workflow_control_action_values():
    assert WorkflowControlAction.APPROVE.value == "approve"
    assert WorkflowControlAction.REJECT.value == "reject"
    assert (
        WorkflowControlAction.REQUEST_CHANGES.value
        == "request_changes"
    )
    assert WorkflowControlAction.CANCEL.value == "cancel"


def test_workflow_trigger_values():
    assert WorkflowTrigger.MANUAL.value == "manual"
    assert WorkflowTrigger.API.value == "api"
    assert WorkflowTrigger.MCP.value == "mcp"
    assert WorkflowTrigger.RETRY.value == "retry"
    assert WorkflowTrigger.SCHEDULED.value == "scheduled"


# ---------------------------------------------------------------------------
# Cross-enum consistency
# ---------------------------------------------------------------------------


def test_job_and_workflow_terminal_states_align():
    assert JobStatus.COMPLETED.is_terminal
    assert WorkflowStep.COMPLETED.is_terminal

    assert JobStatus.FAILED.is_terminal
    assert WorkflowStep.ERROR.is_terminal

    assert JobStatus.CANCELLED.is_terminal
    assert WorkflowStep.CANCELLED.is_terminal


def test_approval_and_review_publication_alignment():
    assert (
        ApprovalStatus.APPROVED.allows_publication
        == ReviewStatus.APPROVED.allows_publication
    )

    assert (
        ApprovalDecision.APPROVED.allows_publication
        == ReviewStatus.APPROVED.allows_publication
    )


def test_error_and_recovery_alignment():
    assert ErrorCode.PROVIDER_TIMEOUT.is_retryable
    assert RecoveryAction.RETRY.is_automated

    assert not ErrorCode.INTERNAL_ERROR.is_retryable
    assert RecoveryAction.HUMAN_REVIEW.requires_human
