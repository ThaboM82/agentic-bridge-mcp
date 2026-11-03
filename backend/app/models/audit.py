"""
Audit-event, retention-policy, export, and integrity database models.

These models provide an immutable, tenant-scoped audit trail for security and
compliance. They capture actor, request, resource, outcome, and metadata for
significant actions, along with retention rules, export jobs, and hash-chain
integrity records.
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


class AuditEvent(UUIDPrimaryKeyMixin, Base):
    """
    Immutable tenant audit event.

    Each event records who performed an action, against which resource, from
    which request, with what outcome, and when it occurred. Audit records
    should be appended only; application APIs must not update or delete them.
    """

    __tablename__ = "audit_events"

    # -------------------------------------------------------------------------
    # Identity and context
    # -------------------------------------------------------------------------

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    actor_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    actor_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    actor_ip: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    user_agent: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Request correlation
    # -------------------------------------------------------------------------

    request_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    correlation_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Resource and workflow context
    # -------------------------------------------------------------------------

    execution_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
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

    action: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    outcome: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Structured details
    # -------------------------------------------------------------------------

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Integrity and retention
    # -------------------------------------------------------------------------

    event_hash: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    previous_event_hash: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    retention_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Timestamp
    # -------------------------------------------------------------------------

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    exports = relationship(
        "AuditExportJob",
        back_populates="source_event",
        foreign_keys="AuditExportJob.source_event_id",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        Index(
            "ix_audit_events_tenant_type_time",
            "tenant_id",
            "event_type",
            "occurred_at",
        ),
        Index(
            "ix_audit_events_tenant_actor",
            "tenant_id",
            "actor_id",
        ),
        Index(
            "ix_audit_events_tenant_action",
            "tenant_id",
            "action",
        ),
        Index(
            "ix_audit_events_resource",
            "tenant_id",
            "resource_type",
            "resource_id",
        ),
        Index(
            "ix_audit_events_execution_time",
            "execution_id",
            "occurred_at",
        ),
        Index(
            "ix_audit_events_request_id",
            "request_id",
        ),
        Index(
            "ix_audit_events_retention_expiry",
            "retention_expires_at",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_expired(self) -> bool:
        """Return whether the event has passed its retention deadline."""
        if self.retention_expires_at is None:
            return False

        return (
            datetime.now(
                self.retention_expires_at.tzinfo
            )
            > self.retention_expires_at
        )

    def to_log_context(self) -> dict[str, Any]:
        """Return a structured dictionary for logging."""
        return {
            "audit_event_id": str(self.id),
            "tenant_id": str(self.tenant_id),
            "event_type": self.event_type,
            "actor_id": (
                str(self.actor_id)
                if self.actor_id
                else None
            ),
            "execution_id": (
                str(self.execution_id)
                if self.execution_id
                else None
            ),
            "resource_type": self.resource_type,
            "resource_id": (
                str(self.resource_id)
                if self.resource_id
                else None
            ),
            "action": self.action,
            "outcome": self.outcome,
            "request_id": (
                str(self.request_id)
                if self.request_id
                else None
            ),
            "occurred_at": self.occurred_at.isoformat(),
        }

    def __repr__(self) -> str:
        """Return a concise representation of the audit event."""
        return (
            f"<AuditEvent id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"event_type={self.event_type!r} "
            f"occurred_at={self.occurred_at}>"
        )


class AuditRetentionPolicy(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Retention policy for audit events.

    A policy may be global when `tenant_id` is null, or tenant-specific when a
    tenant is specified. It can target all event types or a single event type.
    """

    __tablename__ = "audit_retention_policies"

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    event_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    retention_days: Mapped[int] = mapped_column(
        Integer,
        default=365,
        nullable=False,
    )

    archive_before_delete: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    archive_storage_key: Mapped[str | None] = mapped_column(
        String(1_000),
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
            "event_type",
            name="uq_audit_retention_policies_tenant_event_type",
        ),
        Index(
            "ix_audit_retention_policies_active",
            "is_active",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the retention policy."""
        return (
            f"<AuditRetentionPolicy id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"event_type={self.event_type!r} "
            f"retention_days={self.retention_days}>"
        )


class AuditExportJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    ORM model representing an asynchronous audit-event export job.

    Export jobs allow compliance teams to request filtered audit archives
    without blocking API requests.
    """

    __tablename__ = "audit_export_jobs"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    requested_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    source_event_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "audit_events.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    export_format: Mapped[str] = mapped_column(
        String(20),
        default="csv",
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="queued",
        nullable=False,
        index=True,
    )

    event_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    start_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    end_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    record_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    storage_key: Mapped[str | None] = mapped_column(
        String(1_000),
        nullable=True,
    )

    checksum: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    export_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    source_event = relationship(
        "AuditEvent",
        back_populates="exports",
        foreign_keys=[source_event_id],
    )

    __table_args__ = (
        Index(
            "ix_audit_export_jobs_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_audit_export_jobs_tenant_requested",
            "tenant_id",
            "requested_by",
        ),
    )

    @property
    def is_terminal(self) -> bool:
        """Return whether the export job has reached a terminal state."""
        return self.status in {
            "completed",
            "failed",
            "expired",
            "cancelled",
        }

    def __repr__(self) -> str:
        """Return a concise representation of the export job."""
        return (
            f"<AuditExportJob id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"status={self.status!r}>"
        )


class AuditIntegrityCheckpoint(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    ORM model representing a periodic audit-trail integrity checkpoint.

    Checkpoints store a rolling hash over a sequence of audit events, enabling
    detection of tampering or gaps in the audit trail.
    """

    __tablename__ = "audit_integrity_checkpoints"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    checkpoint_hash: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
    )

    previous_checkpoint_hash: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    first_event_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    last_event_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    event_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    checkpointed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    checkpoint_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_audit_integrity_checkpoints_tenant_time",
            "tenant_id",
            "checkpointed_at",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the integrity checkpoint."""
        return (
            f"<AuditIntegrityCheckpoint id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"event_count={self.event_count}>"
        )


__all__ = [
    "AuditEvent",
    "AuditExportJob",
    "AuditIntegrityCheckpoint",
    "AuditRetentionPolicy",
]