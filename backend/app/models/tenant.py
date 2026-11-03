"""
Tenant, settings, invitation, domain, membership, and audit database models.

These models define the tenant workspace and isolation boundary for
AgenticBridge. They support tenant lifecycle management, subscription plans,
configuration settings, custom domains, member invitations, usage limits,
audit history, and tenant-scoped relationships.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
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


class Tenant(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Tenant workspace and isolation boundary.

    A tenant represents an organization, customer, or workspace. All
    tenant-scoped resources should reference `tenant_id` to enforce data
    isolation.
    """

    __tablename__ = "tenants"

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    slug: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    display_name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    external_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Contact information
    # -------------------------------------------------------------------------

    contact_email: Mapped[str | None] = mapped_column(
        String(320),
        nullable=True,
    )

    contact_name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    billing_email: Mapped[str | None] = mapped_column(
        String(320),
        nullable=True,
    )

    country_code: Mapped[str | None] = mapped_column(
        String(2),
        nullable=True,
    )

    timezone: Mapped[str | None] = mapped_column(
        String(100),
        default="UTC",
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(50),
        default="active",
        nullable=False,
        index=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    trial_ends_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    suspended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    suspended_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Subscription
    # -------------------------------------------------------------------------

    plan: Mapped[str] = mapped_column(
        String(50),
        default="standard",
        nullable=False,
        index=True,
    )

    billing_status: Mapped[str | None] = mapped_column(
        String(50),
        default="active",
        nullable=True,
        index=True,
    )

    subscription_external_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Usage limits
    # -------------------------------------------------------------------------

    max_users: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    max_documents: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    max_storage_bytes: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    max_monthly_api_calls: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    feature_flags: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    users = relationship(
        "User",
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    documents = relationship(
        "Document",
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    settings_record = relationship(
        "TenantSettings",
        back_populates="tenant",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    invitations = relationship(
        "TenantInvitation",
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    domains = relationship(
        "TenantDomain",
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    usage_records = relationship(
        "TenantUsageRecord",
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    audit_events = relationship(
        "TenantAuditEvent",
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "slug",
            name="uq_tenants_slug",
        ),
        UniqueConstraint(
            "external_id",
            name="uq_tenants_external_id",
        ),
        Index(
            "ix_tenants_status_created",
            "status",
            "created_at",
        ),
        Index(
            "ix_tenants_plan_status",
            "plan",
            "status",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_deleted(self) -> bool:
        """Return whether the tenant has been deleted."""
        return self.deleted_at is not None

    @property
    def is_suspended(self) -> bool:
        """Return whether the tenant is suspended."""
        return self.status == "suspended"

    @property
    def is_trial(self) -> bool:
        """Return whether the tenant is currently in a trial period."""
        if self.trial_ends_at is None:
            return False

        return (
            datetime.now(
                self.trial_ends_at.tzinfo
            )
            <= self.trial_ends_at
        )

    def suspend(
        self,
        suspended_by: uuid.UUID | None = None,
    ) -> None:
        """Suspend the tenant."""
        self.status = "suspended"
        self.is_active = False
        self.suspended_at = utc_now()
        self.suspended_by = suspended_by

    def reactivate(self) -> None:
        """Reactivate a suspended tenant."""
        self.status = "active"
        self.is_active = True
        self.suspended_at = None
        self.suspended_by = None

    def mark_deleted(self) -> None:
        """Mark the tenant as deleted."""
        self.status = "deleted"
        self.is_active = False
        self.deleted_at = utc_now()

    def __repr__(self) -> str:
        """Return a concise representation of the tenant."""
        return (
            f"<Tenant id={self.id} "
            f"name={self.name!r} "
            f"slug={self.slug!r} "
            f"status={self.status!r}>"
        )


class TenantSettings(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Tenant-level configuration settings.

    This model stores structured tenant configuration separately from the
    tenant record. It supports defaults, feature toggles, security settings,
    notification settings, and integration defaults.
    """

    __tablename__ = "tenant_settings"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "tenants.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        unique=True,
    )

    default_language: Mapped[str | None] = mapped_column(
        String(20),
        default="en",
        nullable=True,
    )

    default_timezone: Mapped[str | None] = mapped_column(
        String(100),
        default="UTC",
        nullable=True,
    )

    require_mfa: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    allow_public_signup: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    session_timeout_minutes: Mapped[int] = mapped_column(
        Integer,
        default=480,
        nullable=False,
    )

    password_policy: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    notification_settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    security_settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    ai_settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    tenant = relationship(
        "Tenant",
        back_populates="settings_record",
    )

    def get_setting(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Return a setting value from the general settings map."""
        return self.settings.get(
            key,
            default,
        )

    def set_setting(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Set a value in the general settings map."""
        self.settings[key] = value

    def __repr__(self) -> str:
        """Return a concise representation of tenant settings."""
        return (
            f"<TenantSettings id={self.id} "
            f"tenant_id={self.tenant_id}>"
        )


class TenantInvitation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Invitation for a user to join a tenant.

    Invitation tokens should be cryptographically random and stored only after
    hashing if the application requires additional protection against database
    exposure.
    """

    __tablename__ = "tenant_invitations"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "tenants.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    email: Mapped[str] = mapped_column(
        String(320),
        nullable=False,
        index=True,
    )

    role: Mapped[str] = mapped_column(
        String(100),
        default="member",
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
        index=True,
    )

    invitation_token: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
    )

    invited_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    accepted_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    revoked_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    invitation_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    tenant = relationship(
        "Tenant",
        back_populates="invitations",
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "email",
            name="uq_tenant_invitations_tenant_email",
        ),
        Index(
            "ix_tenant_invitations_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_tenant_invitations_expiry",
            "expires_at",
        ),
    )

    @property
    def is_expired(self) -> bool:
        """Return whether the invitation has expired."""
        if self.expires_at is None:
            return False

        return (
            datetime.now(
                self.expires_at.tzinfo
            )
            > self.expires_at
        )

    @property
    def is_acceptable(self) -> bool:
        """Return whether the invitation can still be accepted."""
        return (
            self.status == "pending"
            and not self.is_expired
            and self.revoked_at is None
        )

    def accept(
        self,
        accepted_by: uuid.UUID | None = None,
    ) -> None:
        """Mark the invitation as accepted."""
        self.status = "accepted"
        self.accepted_at = utc_now()
        self.accepted_by = accepted_by

    def revoke(
        self,
        revoked_by: uuid.UUID | None = None,
    ) -> None:
        """Mark the invitation as revoked."""
        self.status = "revoked"
        self.revoked_at = utc_now()
        self.revoked_by = revoked_by

    def __repr__(self) -> str:
        """Return a concise representation of the tenant invitation."""
        return (
            f"<TenantInvitation id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"email={self.email!r} "
            f"status={self.status!r}>"
        )


class TenantDomain(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Custom domain associated with a tenant.

    Domains can be used for tenant resolution, SSO routing, email sending, and
    branded application access.
    """

    __tablename__ = "tenant_domains"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "tenants.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    domain: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
    )

    is_primary: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    verification_token: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    domain_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    tenant = relationship(
        "Tenant",
        back_populates="domains",
    )

    def mark_verified(self) -> None:
        """Mark the domain as verified."""
        self.is_verified = True
        self.verified_at = utc_now()

    def __repr__(self) -> str:
        """Return a concise representation of the tenant domain."""
        return (
            f"<TenantDomain id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"domain={self.domain!r}>"
        )


class TenantUsageRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Tenant usage record for metering and quota enforcement.
    """

    __tablename__ = "tenant_usage_records"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "tenants.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    usage_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    usage_amount: Mapped[float] = mapped_column(
        Numeric(20, 4),
        default=0,
        nullable=False,
    )

    usage_unit: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    usage_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    tenant = relationship(
        "Tenant",
        back_populates="usage_records",
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "usage_type",
            "period_start",
            "period_end",
            name="uq_tenant_usage_records_tenant_type_period",
        ),
        Index(
            "ix_tenant_usage_records_tenant_period",
            "tenant_id",
            "period_start",
            "period_end",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the usage record."""
        return (
            f"<TenantUsageRecord id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"usage_type={self.usage_type!r} "
            f"usage_amount={self.usage_amount}>"
        )


class TenantAuditEvent(UUIDPrimaryKeyMixin, Base):
    """
    Immutable audit event for tenant lifecycle and administrative activity.
    """

    __tablename__ = "tenant_audit_events"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "tenants.id",
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

    tenant = relationship(
        "Tenant",
        back_populates="audit_events",
    )

    __table_args__ = (
        Index(
            "ix_tenant_audit_events_tenant_time",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_tenant_audit_events_actor",
            "tenant_id",
            "actor_id",
        ),
        Index(
            "ix_tenant_audit_events_type_time",
            "event_type",
            "occurred_at",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the tenant audit event."""
        return (
            f"<TenantAuditEvent id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"event_type={self.event_type!r}>"
        )


__all__ = [
    "Tenant",
    "TenantAuditEvent",
    "TenantDomain",
    "TenantInvitation",
    "TenantSettings",
    "TenantUsageRecord",
]