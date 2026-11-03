"""
Approval, policy, delegation, and audit database models.

These models support tenant-scoped human-in-the-loop review: approval requests,
multi-approver decisions, approval policies, temporary delegations, reminders,
escalations, and immutable approval audit events.
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

from app.core.database import Base
from app.models.base import (
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    generate_uuid,
)


class Approval(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Human approval request for a workflow execution or resource.

    An approval may require one or more reviewers. Its lifecycle moves from
    `pending` to `approved`, `rejected`, `changes_requested`, `expired`, or
    `cancelled`.
    """

    __tablename__ = "approvals"

    # -------------------------------------------------------------------------
    # Identity and ownership
    # -------------------------------------------------------------------------

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    execution_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "workflow_executions.id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    resource_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    resource_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    workflow_step: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
        index=True,
    )

    decision: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    comment: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    priority: Mapped[int] = mapped_column(
        Integer,
        default=5,
        nullable=False,
        index=True,
    )

    # -------------------------------------------------------------------------
    # People
    # -------------------------------------------------------------------------

    requested_by: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    assigned_role: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # SLA, reminders, and escalation
    # -------------------------------------------------------------------------

    due_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    reminder_sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    escalated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    escalated_to: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    approval_metadata: Mapped[dict[str, Any]] = mapped_column(
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

    # -------------------------------------------------------------------------
    # Timestamps
    # -------------------------------------------------------------------------

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

    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    workflow_execution = relationship(
        "WorkflowExecutionModel",
        back_populates="approvals",
        foreign_keys=[execution_id],
    )

    decisions = relationship(
        "ApprovalDecision",
        back_populates="approval",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    reminders = relationship(
        "ApprovalReminder",
        back_populates="approval",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    audit_events = relationship(
        "ApprovalAuditEvent",
        back_populates="approval",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        Index(
            "ix_approvals_tenant_status_created",
            "tenant_id",
            "status",
            "created_at",
        ),
        Index(
            "ix_approvals_execution_status",
            "execution_id",
            "status",
        ),
        Index(
            "ix_approvals_tenant_reviewer_status",
            "tenant_id",
            "reviewer_id",
            "status",
        ),
        Index(
            "ix_approvals_tenant_due_status",
            "tenant_id",
            "due_at",
            "status",
        ),
        Index(
            "ix_approvals_resource",
            "tenant_id",
            "resource_type",
            "resource_id",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_pending(self) -> bool:
        """Return whether the approval is still awaiting a decision."""
        return self.status == "pending"

    @property
    def is_terminal(self) -> bool:
        """Return whether the approval has reached a terminal state."""
        return self.status in {
            "approved",
            "rejected",
            "changes_requested",
            "expired",
            "cancelled",
        }

    @property
    def is_overdue(self) -> bool:
        """Return whether a pending approval is past its due time."""
        if not self.is_pending or self.due_at is None:
            return False

        return (
            datetime.now(
                self.due_at.tzinfo
            )
            > self.due_at
        )

    @property
    def is_expired(self) -> bool:
        """Return whether a pending approval has expired."""
        if not self.is_pending or self.expires_at is None:
            return False

        return (
            datetime.now(
                self.expires_at.tzinfo
            )
            > self.expires_at
        )

    def __repr__(self) -> str:
        """Return a concise representation of the approval."""
        return (
            f"<Approval id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"execution_id={self.execution_id} "
            f"status={self.status!r}>"
        )


class ApprovalDecision(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    ORM model representing one reviewer decision on an approval.

    This supports multi-approver workflows while preserving an immutable
    decision history.
    """

    __tablename__ = "approval_decisions"

    approval_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "approvals.id",
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

    reviewer_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    decision: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    comment: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    decision_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    approval = relationship(
        "Approval",
        back_populates="decisions",
    )

    __table_args__ = (
        UniqueConstraint(
            "approval_id",
            "reviewer_id",
            name="uq_approval_decisions_approval_reviewer",
        ),
        Index(
            "ix_approval_decisions_tenant_decided",
            "tenant_id",
            "decided_at",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the approval decision."""
        return (
            f"<ApprovalDecision id={self.id} "
            f"approval_id={self.approval_id} "
            f"reviewer_id={self.reviewer_id} "
            f"decision={self.decision!r}>"
        )


class ApprovalPolicy(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Tenant approval policy for workflow types and steps.

    Policies define who must approve, how many approvals are required, SLA
    behavior, and escalation rules.
    """

    __tablename__ = "approval_policies"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    workflow_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    workflow_step: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        index=True,
    )

    required_role: Mapped[str] = mapped_column(
        String(100),
        default="admin",
        nullable=False,
    )

    required_approvals: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    required_rejections_to_block: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    approval_mode: Mapped[str] = mapped_column(
        String(50),
        default="any",
        nullable=False,
        index=True,
    )

    sla_hours: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    auto_expire_hours: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    escalation_user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    policy_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workflow_type",
            "workflow_step",
            name="uq_approval_policies_tenant_workflow_step",
        ),
        Index(
            "ix_approval_policies_tenant_active",
            "tenant_id",
            "is_active",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the approval policy."""
        return (
            f"<ApprovalPolicy id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"workflow_type={self.workflow_type!r} "
            f"step={self.workflow_step!r}>"
        )


class ApprovalDelegation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Temporary approval delegation between users.

    A delegation allows a delegate to act on behalf of a delegator for a
    specified workflow type and time window.
    """

    __tablename__ = "approval_delegations"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    delegator_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    delegate_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    workflow_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    workflow_step: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    starts_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    ends_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    delegation_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_approval_delegations_tenant_active",
            "tenant_id",
            "is_active",
        ),
        Index(
            "ix_approval_delegations_delegate_active",
            "delegate_id",
            "is_active",
        ),
        Index(
            "ix_approval_delegations_period",
            "starts_at",
            "ends_at",
        ),
    )

    @property
    def is_currently_active(self) -> bool:
        """Return whether the delegation is active for the current time."""
        if not self.is_active:
            return False

        current_time = datetime.now(
            self.starts_at.tzinfo
        )

        return (
            self.starts_at
            <= current_time
            <= self.ends_at
        )

    def __repr__(self) -> str:
        """Return a concise representation of the delegation."""
        return (
            f"<ApprovalDelegation id={self.id} "
            f"delegator={self.delegator_id} "
            f"delegate={self.delegate_id}>"
        )


class ApprovalReminder(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    ORM model representing a reminder sent for a pending approval.
    """

    __tablename__ = "approval_reminders"

    approval_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "approvals.id",
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

    recipient_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    channel: Mapped[str] = mapped_column(
        String(50),
        default="email",
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="queued",
        nullable=False,
        index=True,
    )

    sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    reminder_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    approval = relationship(
        "Approval",
        back_populates="reminders",
    )

    __table_args__ = (
        Index(
            "ix_approval_reminders_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_approval_reminders_approval_sent",
            "approval_id",
            "sent_at",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the reminder."""
        return (
            f"<ApprovalReminder id={self.id} "
            f"approval_id={self.approval_id} "
            f"channel={self.channel!r} "
            f"status={self.status!r}>"
        )


class ApprovalAuditEvent(UUIDPrimaryKeyMixin, Base):
    """
    ORM model representing an immutable approval audit event.
    """

    __tablename__ = "approval_audit_events"

    approval_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "approvals.id",
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

    approval = relationship(
        "Approval",
        back_populates="audit_events",
    )

    __table_args__ = (
        Index(
            "ix_approval_audit_events_tenant_time",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_approval_audit_events_approval_time",
            "approval_id",
            "occurred_at",
        ),
        Index(
            "ix_approval_audit_events_actor",
            "tenant_id",
            "actor_id",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the audit event."""
        return (
            f"<ApprovalAuditEvent id={self.id} "
            f"approval_id={self.approval_id} "
            f"event_type={self.event_type!r}>"
        )


__all__ = [
    "Approval",
    "ApprovalAuditEvent",
    "ApprovalDecision",
    "ApprovalDelegation",
    "ApprovalPolicy",
    "ApprovalReminder",
]