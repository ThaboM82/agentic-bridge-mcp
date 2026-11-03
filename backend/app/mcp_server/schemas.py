"""
Strict request and response schemas for MCP tools.

These models validate and bound every MCP argument before it reaches a
workflow service or runner. They never include credentials, raw provider
payloads, generated file contents, prompts, completions, or embeddings.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


MAX_BRIEF_TEXT_CHARS = 20_000
MAX_COMMENT_CHARS = 2_000
MAX_DOCUMENT_IDS = 25
MAX_METADATA_KEYS = 50
MAX_METADATA_VALUE_CHARS = 1_000
MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 20

WorkflowStatusFilter = Literal[
    "queued",
    "processing",
    "completed",
    "partially_completed",
    "failed",
    "cancelled",
    "timed_out",
]

WorkflowStepFilter = Literal[
    "initialized",
    "retrieval",
    "ingestion",
    "requirements",
    "design",
    "code_generation",
    "validation",
    "review",
    "publishing",
    "paused",
    "cancelled",
    "completed",
    "error",
]

ReviewDecisionFilter = Literal[
    "approved",
    "request_changes",
    "rejected",
]


class MCPMetadata(BaseModel):
    """Bounded, non-sensitive MCP metadata."""

    model_config = {
        "extra": "forbid",
    }

    values: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional non-sensitive metadata.",
    )

    @field_validator("values")
    @classmethod
    def validate_values(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        """Bound metadata keys and values."""
        if len(value) > MAX_METADATA_KEYS:
            raise ValueError(
                f"Metadata cannot contain more than "
                f"{MAX_METADATA_KEYS} keys."
            )

        safe_values: dict[str, Any] = {}

        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    "Metadata keys must be non-empty strings."
                )

            if item is None or isinstance(
                item,
                (bool, int, float, str),
            ):
                safe_values[key] = (
                    item[:MAX_METADATA_VALUE_CHARS]
                    if isinstance(item, str)
                    else item
                )
            else:
                raise ValueError(
                    "Metadata values must be strings, numbers, booleans, "
                    "or null."
                )

        return safe_values


class WorkflowStartRequest(BaseModel):
    """Request to start an AgenticBridge generation workflow."""

    brief_text: str = Field(
        min_length=1,
        max_length=MAX_BRIEF_TEXT_CHARS,
    )

    document_ids: list[UUID] = Field(
        default_factory=list,
        max_length=MAX_DOCUMENT_IDS,
    )

    workflow_name: str = Field(
        default="design_to_code",
        min_length=1,
        max_length=100,
    )

    approval_required: bool = True

    metadata: MCPMetadata = Field(
        default_factory=MCPMetadata,
    )

    @field_validator("brief_text")
    @classmethod
    def validate_brief_text(
        cls,
        value: str,
    ) -> str:
        """Require non-whitespace brief text."""
        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError(
                "brief_text is required."
            )

        return normalized_value


class WorkflowStatusRequest(BaseModel):
    """Request to inspect a workflow execution."""

    execution_id: UUID


class WorkflowListRequest(BaseModel):
    """Request to list workflow executions."""

    status: WorkflowStatusFilter | None = None
    current_step: WorkflowStepFilter | None = None
    page: int = Field(
        default=1,
        ge=1,
        le=10_000,
    )
    page_size: int = Field(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
    )


class WorkflowCancelRequest(BaseModel):
    """Request to cancel a workflow execution."""

    execution_id: UUID
    reason: str | None = Field(
        default=None,
        max_length=1_000,
    )


class WorkflowReviewRequest(BaseModel):
    """Request to resolve a workflow human-review interrupt."""

    execution_id: UUID
    decision: ReviewDecisionFilter
    comment: str | None = Field(
        default=None,
        max_length=MAX_COMMENT_CHARS,
    )

    @field_validator("comment")
    @classmethod
    def validate_comment(
        cls,
        value: str | None,
    ) -> str | None:
        """Require a comment when changes or rejection are requested."""
        if value is None:
            return None

        normalized_comment = value.strip()

        if not normalized_comment:
            return None

        return normalized_comment


class RetrievalRequest(BaseModel):
    """Request to retrieve workflow-safe context."""

    query: str = Field(
        min_length=1,
        max_length=2_000,
    )
    document_ids: list[UUID] = Field(
        default_factory=list,
        max_length=MAX_DOCUMENT_IDS,
    )
    limit: int = Field(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
    )

    @field_validator("query")
    @classmethod
    def validate_query(
        cls,
        value: str,
    ) -> str:
        """Require non-whitespace query text."""
        normalized_query = value.strip()

        if not normalized_query:
            raise ValueError(
                "query is required."
            )

        return normalized_query


class DocumentSummaryRequest(BaseModel):
    """Request to inspect document metadata."""

    document_id: UUID


class WorkflowAcknowledgement(BaseModel):
    """Safe acknowledgement returned after a workflow operation."""

    execution_id: UUID
    status: str
    current_step: str | None = None
    approval_required: bool = False
    review_status: str | None = None
    interrupt_pending: bool = False
    error_code: str | None = None
    error_message: str | None = None
    recoverable_error: bool = False
    duration_ms: float | None = None
    requirement_count: int = 0
    artifact_count: int = 0
    citation_count: int = 0