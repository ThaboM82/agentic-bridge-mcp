"""
Data-retention policy, retention-job, exemption, and audit models.

These models support tenant-scoped and global data-retention governance. They
define retention rules for resource types, track retention sweep execution,
record legal-hold exemptions, and retain immutable audit events for compliance.
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
    Numeric,
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
    utc_now,
)


class RetentionPolicy(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Tenant or global data-retention policy.

    A policy defines how long records of a given resource type should be
    retained before they are eligible for deletion, archival, or anonymization.
    A null `tenant_id` indicates a platform-wide default policy.
    """

    __tablename__ = "retention_policies"

    # -------------------------------------------------------------------------
    # Identity and ownership
    # -------------------------------------------------------------------------

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    resource_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Retention behavior
    # -------------------------------------------------------------------------

    retention_days: Mapped[int] = mapped_column(
        Integer,
        default=365,
        nullable=False,
    )

    retention_action: Mapped[str] = mapped_column(
        String(50),
        default="delete",
        nullable=False,
        index=True,
    )

    archive_before_delete: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    anonymize_before_delete: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    archive_resource_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Scheduling
    # -------------------------------------------------------------------------

    schedule_cron: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    batch_size: Mapped[int] = mapped_column(
        Integer,
        default=1_000,
        nullable=False,
    )

    max_runtime_minutes: Mapped[int] = mapped_column(
        Integer,
        default=60,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    is_dry_run: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    requires_approval: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    approved_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    policy_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    jobs = relationship(
        "RetentionJob",
        back_populates="policy",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    exemptions = relationship(
        "RetentionExemption",
        back_populates="policy",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    audit_events = relationship(
        "RetentionAuditEvent",
        back_populates="policy",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "resource_type",
            "name",
            name="uq_retention_policies_tenant_resource_name",
        ),
        Index(
            "ix_retention_policies_resource_active",
            "resource_type",
            "is_active",
        ),
        Index(
            "ix_retention_policies_tenant_active",
            "tenant_id",
            "is_active",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_global(self) -> bool:
        """Return whether this is a platform-wide policy."""
        return self.tenant_id is None

    @property
    def is_approved(self) -> bool:
        """Return whether the policy has been approved."""
        return (
            not self.requires_approval
            or self.approved_at is not None
        )

    def calculate_expiry(
        self,
        created_at: datetime,
    ) -> datetime:
        """Return the retention expiry date for a record."""
        return created_at + timedelta(
            days=self.retention_days,
        )

    def __repr__(self) -> str:
        """Return a concise representation of the retention policy."""
        return (
            f"<RetentionPolicy id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"resource_type={self.resource_type!r} "
            f"retention_days={self.retention_days}>"
        )


class RetentionJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Execution record for a retention sweep.

    A retention job records the scope, progress, outcome, and diagnostics for
    one execution of a retention policy.
    """

    __tablename__ = "retention_jobs"

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    policy_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "retention_policies.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    resource_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    trigger: Mapped[str] = mapped_column(
        String(50),
        default="scheduled",
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="queued",
        nullable=False,
        index=True,
    )

    is_dry_run: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Execution progress
    # -------------------------------------------------------------------------

    records_scanned: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    records_processed: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    records_deleted: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    records_archived: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    records_anonymized: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    records_exempted: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    records_failed: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    bytes_freed: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Execution context
    # -------------------------------------------------------------------------

    started_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    worker_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    cursor: Mapped[str | None] = mapped_column(
        String(1_000),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    job_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    policy = relationship(
        "RetentionPolicy",
        back_populates="jobs",
    )

    audit_events = relationship(
        "RetentionAuditEvent",
        back_populates="job",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        Index(
            "ix_retention_jobs_status_created",
            "status",
            "created_at",
        ),
        Index(
            "ix_retention_jobs_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_retention_jobs_policy_status",
            "policy_id",
            "status",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """Return whether the retention job has reached a terminal state."""
        return self.status in {
            "completed",
            "failed",
            "cancelled",
        }

    @property
    def success_rate(self) -> float | None:
        """Return the percentage of processed records handled successfully."""
        if self.records_processed == 0:
            return None

        successful = (
            self.records_deleted
            + self.records_archived
            + self.records_anonymized
        )

        return (
            successful
            / self.records_processed
            * 100
        )

    def __repr__(self) -> str:
        """Return a concise representation of the retention job."""
        return (
            f"<RetentionJob id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"resource_type={self.resource_type!r} "
            f"status={self.status!r}>"
        )


class RetentionExemption(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Legal-hold or compliance exemption for a specific resource.

    An exempt resource is excluded from retention deletion until the exemption
    is released or expires.
    """

    __tablename__ = "retention_exemptions"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    policy_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "retention_policies.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    resource_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    resource_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    reason: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    released_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    released_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    exemption_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    policy = relationship(
        "RetentionPolicy",
        back_populates="exemptions",
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "resource_type",
            "resource_id",
            name="uq_retention_exemptions_tenant_resource",
        ),
        Index(
            "ix_retention_exemptions_tenant_active",
            "tenant_id",
            "is_active",
        ),
        Index(
            "ix_retention_exemptions_expiry",
            "expires_at",
        ),
    )

    @property
    def is_expired(self) -> bool:
        """Return whether the exemption has expired."""
        if self.expires_at is None:
            return False

        return (
            datetime.now(
                self.expires_at.tzinfo
            )
            > self.expires_at
        )

    @property
    def is_effective(self) -> bool:
        """Return whether the exemption currently blocks retention actions."""
        return (
            self.is_active
            and not self.is_expired
            and self.released_at is None
        )

    def release(
        self,
        released_by: uuid.UUID | None = None,
    ) -> None:
        """Release the retention exemption."""
        self.is_active = False
        self.released_at = utc_now()
        self.released_by = released_by

    def __repr__(self) -> str:
        """Return a concise representation of the retention exemption."""
        return (
            f"<RetentionExemption id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"resource_type={self.resource_type!r} "
            f"resource_id={self.resource_id}>"
        )


class RetentionAuditEvent(UUIDPrimaryKeyMixin, Base):
    """
    Immutable audit event for retention policy and job activity.
    """

    __tablename__ = "retention_audit_events"

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    policy_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "retention_policies.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    job_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "retention_jobs.id",
            ondelete="SET NULL",
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

    policy = relationship(
        "RetentionPolicy",
        back_populates="audit_events",
    )

    job = relationship(
        "RetentionJob",
        back_populates="audit_events",
    )

    __table_args__ = (
        Index(
            "ix_retention_audit_events_tenant_time",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_retention_audit_events_policy_time",
            "policy_id",
            "occurred_at",
        ),
        Index(
            "ix_retention_audit_events_job_time",
            "job_id",
            "occurred_at",
        ),
        Index(
            "ix_retention_audit_events_resource",
            "resource_type",
            "resource_id",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the retention audit event."""
        return (
            f"<RetentionAuditEvent id={self.id} "
            f"policy_id={self.policy_id} "
            f"job_id={self.job_id} "
            f"event_type={self.event_type!r}>"
        )


__all__ = [
    "RetentionAuditEvent",
    "RetentionExemption",
    "RetentionJob",
    "RetentionPolicy",
]