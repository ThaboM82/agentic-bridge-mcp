"""
Shared API request and response schemas.

This module defines reusable Pydantic models for API errors, approvals,
artifacts, audit events, briefs, retrieval, usage, users, and workflow
executions. It centralizes validation rules and response shapes used across
the API layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


# ---------------------------------------------------------------------------
# Error schemas
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Structured error detail."""

    code: str = Field(
        min_length=1,
        description="Stable machine-readable error code.",
    )
    message: str = Field(
        min_length=1,
        description="Human-readable error message.",
    )
    retryable: bool = Field(
        default=False,
        description="Whether the client may retry the request.",
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error context.",
    )
    request_id: UUID | None = Field(
        default=None,
        description="Request correlation identifier.",
    )


class ErrorResponse(BaseModel):
    """Standard API error response."""

    error: ErrorDetail


class ValidationErrorItem(BaseModel):
    """Individual request-validation failure."""

    field: str = Field(
        description="Dotted path to the invalid field.",
    )
    message: str = Field(
        description="Human-readable validation message.",
    )
    type: str = Field(
        default="value_error",
        description="Validation error type.",
    )


class ValidationErrorResponse(BaseModel):
    """Standard validation error response."""

    error: ErrorDetail
    validation_errors: list[ValidationErrorItem] = Field(
        default_factory=list,
        description="Field-level validation failures.",
    )


# ---------------------------------------------------------------------------
# Approval schemas
# ---------------------------------------------------------------------------


class ApprovalResolveRequest(BaseModel):
    """Request to resolve a pending approval."""

    decision: str = Field(
        description="Approved, rejected, or request_changes.",
    )
    comment: str | None = Field(
        default=None,
        max_length=2_000,
        description="Optional reviewer comment.",
    )

    @field_validator(
        "decision"
    )
    @classmethod
    def validate_decision(
        cls,
        value: str,
    ) -> str:
        """Normalize and validate the approval decision."""
        normalized_decision = (
            value.strip().lower()
        )

        allowed_decisions = {
            "approved",
            "rejected",
            "request_changes",
        }

        if normalized_decision not in allowed_decisions:
            raise ValueError(
                "Decision must be one of: "
                + ", ".join(
                    sorted(
                        allowed_decisions
                    )
                )
            )

        return normalized_decision


class ApprovalResponse(BaseModel):
    """Approval resource response."""

    model_config = ConfigDict(
        from_attributes=True
    )

    id: UUID
    execution_id: UUID
    tenant_id: UUID
    status: str
    decision: str | None = None
    reason: str
    comment: str | None = None
    workflow_step: str
    requested_by: UUID
    reviewer_id: UUID | None = None
    created_at: datetime
    resolved_at: datetime | None = None


# ---------------------------------------------------------------------------
# Artifact schemas
# ---------------------------------------------------------------------------


class ArtifactResponse(BaseModel):
    """Generated artifact metadata response."""

    model_config = ConfigDict(
        from_attributes=True
    )

    id: UUID
    execution_id: UUID
    tenant_id: UUID
    artifact_type: str
    storage_key: str
    file_size_bytes: int = Field(
        ge=0,
    )
    checksum: str | None = None
    mime_type: str | None = None
    status: str
    created_at: datetime
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class ArtifactDownloadResponse(BaseModel):
    """Short-lived artifact download reference."""

    artifact_id: UUID
    download_url: str = Field(
        min_length=1,
    )
    expires_in_seconds: int = Field(
        ge=1,
    )
    expires_at: datetime


# ---------------------------------------------------------------------------
# Audit schemas
# ---------------------------------------------------------------------------


class AuditEventResponse(BaseModel):
    """Audit event response."""

    model_config = ConfigDict(
        from_attributes=True
    )

    id: UUID
    tenant_id: UUID
    event_type: str
    actor_id: UUID | None = None
    execution_id: UUID | None = None
    resource_type: str | None = None
    resource_id: UUID | None = None
    action: str | None = None
    outcome: str | None = None
    occurred_at: datetime
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class AuditEventSummaryResponse(BaseModel):
    """Tenant-level audit summary."""

    tenant_id: UUID
    total_events: int = Field(
        ge=0,
    )
    events_by_type: dict[str, int] = Field(
        default_factory=dict
    )
    activity_by_actor: dict[str, int] = Field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Brief schemas
# ---------------------------------------------------------------------------


class BriefCreateRequest(BaseModel):
    """Request to create a product brief."""

    title: str = Field(
        min_length=1,
        max_length=300,
    )
    raw_brief_text: str = Field(
        min_length=1,
        max_length=100_000,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class BriefUpdateRequest(BaseModel):
    """Request to update a product brief."""

    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=300,
    )
    raw_brief_text: str | None = Field(
        default=None,
        min_length=1,
        max_length=100_000,
    )
    metadata: dict[str, Any] | None = None


class BriefResponse(BaseModel):
    """Brief resource response."""

    model_config = ConfigDict(
        from_attributes=True
    )

    id: UUID
    tenant_id: UUID
    title: str
    raw_brief_text: str
    status: str
    created_by: UUID
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Retrieval schemas
# ---------------------------------------------------------------------------


class RetrievalSearchRequest(BaseModel):
    """Semantic retrieval search request."""

    query: str = Field(
        min_length=1,
        max_length=2_000,
    )
    document_ids: list[UUID] = Field(
        default_factory=list
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=50,
    )
    similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    metadata_filters: dict[str, Any] = Field(
        default_factory=dict
    )


class RetrievalResultResponse(BaseModel):
    """Semantic retrieval result."""

    embedding_id: UUID
    document_id: UUID
    score: float = Field(
        ge=0.0,
        le=1.0,
    )
    source_chunk: str | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class RetrievalSearchResponse(BaseModel):
    """Semantic retrieval response envelope."""

    query: str
    results: list[RetrievalResultResponse]
    total_results: int = Field(
        ge=0,
    )
    elapsed_ms: float = Field(
        ge=0,
    )


# ---------------------------------------------------------------------------
# Usage schemas
# ---------------------------------------------------------------------------


class UsageSummaryResponse(BaseModel):
    """Provider usage summary."""

    tenant_id: UUID
    usage_events: int = Field(
        ge=0,
    )
    prompt_tokens: int = Field(
        ge=0,
    )
    completion_tokens: int = Field(
        ge=0,
    )
    total_tokens: int = Field(
        ge=0,
    )


class UsageByModelResponse(BaseModel):
    """Provider usage grouped by model."""

    provider: str
    model: str
    usage_events: int = Field(
        ge=0,
    )
    prompt_tokens: int = Field(
        ge=0,
    )
    completion_tokens: int = Field(
        ge=0,
    )
    total_tokens: int = Field(
        ge=0,
    )


# ---------------------------------------------------------------------------
# User schemas
# ---------------------------------------------------------------------------


class UserResponse(BaseModel):
    """User profile response."""

    model_config = ConfigDict(
        from_attributes=True
    )

    id: UUID
    tenant_id: UUID
    email: str
    full_name: str | None = None
    role: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class UserRoleUpdateRequest(BaseModel):
    """Request to update a user role."""

    role: str = Field(
        min_length=1,
        max_length=100,
    )

    @field_validator(
        "role"
    )
    @classmethod
    def validate_role(
        cls,
        value: str,
    ) -> str:
        """Normalize and validate the requested role."""
        normalized_role = (
            value.strip().lower()
        )

        allowed_roles = {
            "owner",
            "admin",
            "member",
            "viewer",
        }

        if normalized_role not in allowed_roles:
            raise ValueError(
                "Role must be one of: "
                + ", ".join(
                    sorted(
                        allowed_roles
                    )
                )
            )

        return normalized_role


class UserStatusUpdateRequest(BaseModel):
    """Request to update a user activation status."""

    is_active: bool


# ---------------------------------------------------------------------------
# Workflow schemas
# ---------------------------------------------------------------------------


class WorkflowCreateRequest(BaseModel):
    """Request to create a workflow execution."""

    workflow_type: str = Field(
        min_length=1,
        max_length=100,
    )
    title: str = Field(
        min_length=1,
        max_length=300,
    )
    input_data: dict[str, Any] = Field(
        default_factory=dict
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class WorkflowStatusUpdateRequest(BaseModel):
    """Request to update a workflow execution status."""

    status: str = Field(
        min_length=1,
        max_length=50,
    )

    @field_validator(
        "status"
    )
    @classmethod
    def validate_status(
        cls,
        value: str,
    ) -> str:
        """Normalize and validate the requested workflow status."""
        normalized_status = (
            value.strip().lower()
        )

        allowed_statuses = {
            "queued",
            "processing",
            "waiting_approval",
            "completed",
            "failed",
            "cancelled",
        }

        if normalized_status not in allowed_statuses:
            raise ValueError(
                "Status must be one of: "
                + ", ".join(
                    sorted(
                        allowed_statuses
                    )
                )
            )

        return normalized_status


class WorkflowExecutionResponse(BaseModel):
    """Workflow execution resource response."""

    model_config = ConfigDict(
        from_attributes=True
    )

    execution_id: UUID
    tenant_id: UUID
    workflow_type: str
    title: str
    status: str
    current_step: str | None = None
    input_data: dict[str, Any] = Field(
        default_factory=dict
    )
    result_data: dict[str, Any] = Field(
        default_factory=dict
    )
    error_message: str | None = None
    cancel_requested: bool = False
    created_by: UUID
    created_at: datetime
    updated_at: datetime