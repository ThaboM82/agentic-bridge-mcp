"""
Creative brief database models.

These models store tenant-scoped creative briefs, immutable brief versions,
reusable templates, workflow links, review feedback, and brief audit events.
They support the full brief lifecycle from drafting through approval and
workflow execution.
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


class Brief(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Tenant-scoped creative brief.

    A brief captures campaign intent, audience, tone, deliverables, and
    constraints. It can be versioned, reviewed, approved, and used to start one
    or more workflow executions.
    """

    __tablename__ = "briefs"

    # -------------------------------------------------------------------------
    # Identity and ownership
    # -------------------------------------------------------------------------

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

    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    title: Mapped[str] = mapped_column(
        String(300),
        nullable=False,
        index=True,
    )

    external_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Brief content
    # -------------------------------------------------------------------------

    raw_brief_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    objective: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    audience: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    tone: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    deliverables: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    constraints: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    key_messages: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    channels: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(50),
        default="draft",
        nullable=False,
        index=True,
    )

    current_version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    priority: Mapped[int] = mapped_column(
        Integer,
        default=5,
        nullable=False,
        index=True,
    )

    due_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    approved_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Workflow linkage
    # -------------------------------------------------------------------------

    execution_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    workflow_status: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    brief_metadata: Mapped[dict[str, Any]] = mapped_column(
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
    # Relationships
    # -------------------------------------------------------------------------

    versions = relationship(
        "BriefVersion",
        back_populates="brief",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    feedback = relationship(
        "BriefFeedback",
        back_populates="brief",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    audit_events = relationship(
        "BriefAuditEvent",
        back_populates="brief",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "external_id",
            name="uq_briefs_tenant_external_id",
        ),
        Index(
            "ix_briefs_tenant_status_created",
            "tenant_id",
            "status",
            "created_at",
        ),
        Index(
            "ix_briefs_tenant_created_by",
            "tenant_id",
            "created_by",
        ),
        Index(
            "ix_briefs_tenant_execution",
            "tenant_id",
            "execution_id",
        ),
        Index(
            "ix_briefs_tenant_due_status",
            "tenant_id",
            "due_date",
            "status",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_editable(self) -> bool:
        """Return whether the brief can still be edited."""
        return self.status in {
            "draft",
            "changes_requested",
        }

    @property
    def is_approved(self) -> bool:
        """Return whether the brief has been approved."""
        return self.status == "approved"

    @property
    def is_archived(self) -> bool:
        """Return whether the brief has been archived."""
        return self.status == "archived"

    @property
    def is_overdue(self) -> bool:
        """Return whether the brief is past its due date."""
        if self.due_date is None:
            return False

        return (
            datetime.now(
                self.due_date.tzinfo
            )
            > self.due_date
        )

    def __repr__(self) -> str:
        """Return a concise representation of the brief."""
        return (
            f"<Brief id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"title={self.title!r} "
            f"status={self.status!r}>"
        )


class BriefVersion(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Immutable version snapshot of a brief.

    Each version preserves the exact brief content at a point in time so that
    changes, reviews, and workflow inputs remain auditable.
    """

    __tablename__ = "brief_versions"

    brief_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "briefs.id",
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

    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(300),
        nullable=False,
    )

    raw_brief_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    objective: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    audience: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    tone: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    deliverables: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    constraints: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    change_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    version_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    brief = relationship(
        "Brief",
        back_populates="versions",
    )

    __table_args__ = (
        UniqueConstraint(
            "brief_id",
            "version_number",
            name="uq_brief_versions_brief_version",
        ),
        Index(
            "ix_brief_versions_tenant_created",
            "tenant_id",
            "created_at",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the brief version."""
        return (
            f"<BriefVersion id={self.id} "
            f"brief_id={self.brief_id} "
            f"version={self.version_number}>"
        )


class BriefTemplate(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Reusable brief template.

    A template may be global when `tenant_id` is null, or tenant-specific when
    a tenant is specified.
    """

    __tablename__ = "brief_templates"

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        index=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    template_body: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    default_deliverables: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    default_constraints: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    template_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "name",
            name="uq_brief_templates_tenant_name",
        ),
        Index(
            "ix_brief_templates_tenant_active",
            "tenant_id",
            "is_active",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the brief template."""
        return (
            f"<BriefTemplate id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"name={self.name!r}>"
        )


class BriefFeedback(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Reviewer feedback on a brief.

    Feedback can be attached to a specific version and supports approval,
    rejection, or change requests.
    """

    __tablename__ = "brief_feedback"

    brief_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "briefs.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    brief_version_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "brief_versions.id",
            ondelete="SET NULL",
        ),
        nullable=True,
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

    feedback_type: Mapped[str] = mapped_column(
        String(50),
        default="comment",
        nullable=False,
        index=True,
    )

    comment: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    rating: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    is_resolved: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    feedback_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    brief = relationship(
        "Brief",
        back_populates="feedback",
    )

    __table_args__ = (
        Index(
            "ix_brief_feedback_tenant_brief",
            "tenant_id",
            "brief_id",
        ),
        Index(
            "ix_brief_feedback_tenant_reviewer",
            "tenant_id",
            "reviewer_id",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of brief feedback."""
        return (
            f"<BriefFeedback id={self.id} "
            f"brief_id={self.brief_id} "
            f"reviewer_id={self.reviewer_id} "
            f"type={self.feedback_type!r}>"
        )


class BriefAuditEvent(UUIDPrimaryKeyMixin, Base):
    """
    Immutable audit event for brief lifecycle and access activity.
    """

    __tablename__ = "brief_audit_events"

    brief_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "briefs.id",
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

    request_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
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

    brief = relationship(
        "Brief",
        back_populates="audit_events",
    )

    __table_args__ = (
        Index(
            "ix_brief_audit_events_tenant_time",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_brief_audit_events_brief_time",
            "brief_id",
            "occurred_at",
        ),
        Index(
            "ix_brief_audit_events_actor",
            "tenant_id",
            "actor_id",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the brief audit event."""
        return (
            f"<BriefAuditEvent id={self.id} "
            f"brief_id={self.brief_id} "
            f"event_type={self.event_type!r}>"
        )


__all__ = [
    "Brief",
    "BriefAuditEvent",
    "BriefFeedback",
    "BriefTemplate",
    "BriefVersion",
]