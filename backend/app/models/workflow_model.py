"""
Workflow execution database models.

These models track tenant-scoped autonomous agentic workflow executions,
hierarchical sub-graph tracing, step telemetry, human approvals, generated
artifacts, LLM token utilization, semantic memory embeddings, and audit
events.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)
from pgvector.sqlalchemy import Vector

from app.core.database import Base


EMBEDDING_DIMENSIONS = 1536


def _generate_uuid() -> uuid.UUID:
    """Generate a database-side UUID default."""
    return uuid.uuid4()


class WorkflowExecutionModel(Base):
    """
    Enterprise-grade ORM model for tracking autonomous agentic workflow
    executions, hierarchical sub-graph tracing, step telemetry logs, LLM token
    utilization, and semantic vector embeddings.
    """

    __tablename__ = "workflow_executions"

    # -------------------------------------------------------------------------
    # Primary identifiers and status
    # -------------------------------------------------------------------------

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=_generate_uuid,
        nullable=False,
        index=True,
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    parent_execution_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "workflow_executions.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    root_execution_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    workflow_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    workflow_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
        index=True,
    )

    current_step: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    priority: Mapped[int] = mapped_column(
        Integer,
        default=5,
        nullable=False,
        index=True,
    )

    cancel_requested: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Agent and model telemetry
    # -------------------------------------------------------------------------

    agent_version: Mapped[str | None] = mapped_column(
        String(50),
        default="1.0.0",
        nullable=True,
    )

    agent_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    model_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    model_provider: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Execution payloads, traces, and results
    # -------------------------------------------------------------------------

    input_payload: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    output_result: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    execution_steps: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Token usage and cost auditing
    # -------------------------------------------------------------------------

    prompt_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    completion_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    total_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    estimated_cost: Mapped[float | None] = mapped_column(
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Semantic memory and retrieval
    # -------------------------------------------------------------------------

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
        nullable=True,
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Metadata and tracing
    # -------------------------------------------------------------------------

    extra_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    correlation_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    idempotency_key: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Timestamps
    # -------------------------------------------------------------------------

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    failed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    cancelled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    parent_execution: Mapped["WorkflowExecutionModel | None"] = relationship(
        remote_side=[id],
        back_populates="child_executions",
    )

    child_executions: Mapped[list["WorkflowExecutionModel"]] = relationship(
        back_populates="parent_execution",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    approvals: Mapped[list["WorkflowApprovalModel"]] = relationship(
        back_populates="workflow_execution",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    artifacts: Mapped[list["WorkflowArtifactModel"]] = relationship(
        back_populates="workflow_execution",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    audit_events: Mapped[list["WorkflowAuditEventModel"]] = relationship(
        back_populates="workflow_execution",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_workflow_executions_tenant_idempotency_key",
        ),
        Index(
            "idx_workflow_name_status",
            "workflow_name",
            "status",
        ),
        Index(
            "idx_workflow_created_at",
            "created_at",
        ),
        Index(
            "idx_parent_execution",
            "parent_execution_id",
        ),
        Index(
            "idx_workflow_tenant_status_created",
            "tenant_id",
            "status",
            "created_at",
        ),
        Index(
            "idx_workflow_tenant_type",
            "tenant_id",
            "workflow_type",
        ),
        Index(
            "idx_workflow_root_execution",
            "root_execution_id",
        ),
        Index(
            "idx_workflow_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={
                "embedding": "vector_cosine_ops"
            },
            postgresql_with={
                "lists": 100
            },
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """Return whether the execution has reached a terminal state."""
        return self.status in {
            "completed",
            "failed",
            "cancelled",
        }

    @property
    def is_active(self) -> bool:
        """Return whether the execution is still active."""
        return self.status in {
            "pending",
            "running",
            "paused",
        }

    @property
    def can_retry(self) -> bool:
        """Return whether the execution can be retried."""
        return self.status in {
            "failed",
            "cancelled",
        }

    @property
    def duration_seconds(self) -> float | None:
        """Return execution duration in seconds, when available."""
        if not self.started_at:
            return None

        end_time = (
            self.completed_at
            or self.failed_at
            or self.cancelled_at
        )

        if not end_time:
            return None

        return (
            end_time - self.started_at
        ).total_seconds()

    def __repr__(self) -> str:
        """Return a concise representation of the workflow execution."""
        return (
            f"<WorkflowExecutionModel(id='{self.id}', "
            f"name='{self.workflow_name}', "
            f"status='{self.status}')>"
        )


class WorkflowStepModel(Base):
    """
    ORM model representing a single step within a workflow execution.

    Steps provide durable, queryable telemetry for agent reasoning, tool calls,
    model usage, and failures.
    """

    __tablename__ = "workflow_steps"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=_generate_uuid,
        nullable=False,
    )

    execution_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "workflow_executions.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    step_index: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    step_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    step_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
        index=True,
    )

    agent_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    model_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    input_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    output_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    prompt_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    completion_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    total_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    step_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True,
    )

    workflow_execution: Mapped[WorkflowExecutionModel] = relationship(
        back_populates=None,
    )

    __table_args__ = (
        UniqueConstraint(
            "execution_id",
            "step_index",
            name="uq_workflow_steps_execution_step_index",
        ),
        Index(
            "idx_workflow_steps_execution_status",
            "execution_id",
            "status",
        ),
        Index(
            "idx_workflow_steps_tenant_status",
            "tenant_id",
            "status",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the workflow step."""
        return (
            f"<WorkflowStepModel(id='{self.id}', "
            f"execution_id='{self.execution_id}', "
            f"step='{self.step_name}', "
            f"status='{self.status}')>"
        )


class WorkflowApprovalModel(Base):
    """
    ORM model representing a human approval request for a workflow step.
    """

    __tablename__ = "workflow_approvals"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=_generate_uuid,
        nullable=False,
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    execution_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "workflow_executions.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    workflow_step: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
        index=True,
    )

    decision: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    comment: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    requested_by: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
    )

    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    workflow_execution: Mapped[WorkflowExecutionModel] = relationship(
        back_populates="approvals",
    )

    __table_args__ = (
        Index(
            "idx_workflow_approvals_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "idx_workflow_approvals_execution_status",
            "execution_id",
            "status",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the approval."""
        return (
            f"<WorkflowApprovalModel(id='{self.id}', "
            f"execution_id='{self.execution_id}', "
            f"status='{self.status}')>"
        )


class WorkflowArtifactModel(Base):
    """
    ORM model representing metadata for an artifact generated by a workflow.
    """

    __tablename__ = "workflow_artifacts"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=_generate_uuid,
        nullable=False,
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    execution_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "workflow_executions.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    artifact_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    storage_key: Mapped[str] = mapped_column(
        String(1_000),
        nullable=False,
    )

    file_size_bytes: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    checksum: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
    )

    mime_type: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="created",
        nullable=False,
        index=True,
    )

    artifact_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True,
    )

    workflow_execution: Mapped[WorkflowExecutionModel] = relationship(
        back_populates="artifacts",
    )

    __table_args__ = (
        Index(
            "idx_workflow_artifacts_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "idx_workflow_artifacts_execution_type",
            "execution_id",
            "artifact_type",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the artifact."""
        return (
            f"<WorkflowArtifactModel(id='{self.id}', "
            f"execution_id='{self.execution_id}', "
            f"type='{self.artifact_type}', "
            f"status='{self.status}')>"
        )


class WorkflowAuditEventModel(Base):
    """
    ORM model representing an audit event associated with a workflow.
    """

    __tablename__ = "workflow_audit_events"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=_generate_uuid,
        nullable=False,
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    execution_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "workflow_executions.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    actor_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    action: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    outcome: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    resource_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    resource_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    event_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    workflow_execution: Mapped[WorkflowExecutionModel] = relationship(
        back_populates="audit_events",
    )

    __table_args__ = (
        Index(
            "idx_workflow_audit_events_tenant_type_time",
            "tenant_id",
            "event_type",
            "occurred_at",
        ),
        Index(
            "idx_workflow_audit_events_execution_time",
            "execution_id",
            "occurred_at",
        ),
        Index(
            "idx_workflow_audit_events_actor",
            "tenant_id",
            "actor_id",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the audit event."""
        return (
            f"<WorkflowAuditEventModel(id='{self.id}', "
            f"execution_id='{self.execution_id}', "
            f"event_type='{self.event_type}', "
            f"occurred_at='{self.occurred_at}')>"
        )


__all__ = [
    "WorkflowApprovalModel",
    "WorkflowArtifactModel",
    "WorkflowAuditEventModel",
    "WorkflowExecutionModel",
    "WorkflowStepModel",
    "EMBEDDING_DIMENSIONS",
]