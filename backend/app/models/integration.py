"""
External integration, credential, synchronization, webhook, and audit models.

These models store tenant-scoped connections to external systems such as CMS
platforms, object-storage providers, identity providers, and third-party APIs.
They support encrypted credential references, synchronization cursors, scheduled
sync jobs, webhook routing, health checks, and integration audit events.
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
    utc_now,
)


class Integration(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Tenant-connected external system.

    An integration represents a configured connection to an external provider,
    such as SharePoint, Google Drive, S3, Salesforce, or a custom CMS.
    """

    __tablename__ = "integrations"

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

    provider: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    provider_account_id: Mapped[str | None] = mapped_column(
        String(255),
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

    # -------------------------------------------------------------------------
    # Connection configuration
    # -------------------------------------------------------------------------

    base_url: Mapped[str | None] = mapped_column(
        String(2_000),
        nullable=True,
    )

    auth_type: Mapped[str | None] = mapped_column(
        String(50),
        default="oauth2",
        nullable=True,
        index=True,
    )

    scopes: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    configuration: Mapped[dict[str, Any]] = mapped_column(
        "configuration",
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(50),
        default="connected",
        nullable=False,
        index=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    is_default: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    connected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    disconnected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_health_check_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    health_status: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    health_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    integration_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    credentials = relationship(
        "IntegrationCredential",
        back_populates="integration",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    sync_state = relationship(
        "IntegrationSyncState",
        back_populates="integration",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    sync_jobs = relationship(
        "IntegrationSyncJob",
        back_populates="integration",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    webhook_routes = relationship(
        "IntegrationWebhookRoute",
        back_populates="integration",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    audit_events = relationship(
        "IntegrationAuditEvent",
        back_populates="integration",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "provider",
            "name",
            name="uq_integrations_tenant_provider_name",
        ),
        Index(
            "ix_integrations_tenant_provider",
            "tenant_id",
            "provider",
        ),
        Index(
            "ix_integrations_tenant_active",
            "tenant_id",
            "is_active",
        ),
        Index(
            "ix_integrations_tenant_status",
            "tenant_id",
            "status",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Return whether the integration is connected and active."""
        return (
            self.is_active
            and self.status == "connected"
        )

    @property
    def is_healthy(self) -> bool:
        """Return whether the last health check succeeded."""
        return self.health_status == "healthy"

    def mark_disconnected(
        self,
        reason: str | None = None,
    ) -> None:
        """Mark the integration as disconnected."""
        self.status = "disconnected"
        self.is_active = False
        self.disconnected_at = utc_now()

        if reason:
            self.health_message = reason

    def __repr__(self) -> str:
        """Return a concise representation of the integration."""
        return (
            f"<Integration id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"provider={self.provider!r} "
            f"status={self.status!r}>"
        )


class IntegrationCredential(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Encrypted credential reference for an integration.

    Plaintext secrets must never be stored in this model. The
    `encrypted_secret` field should contain ciphertext produced by the
    application's secret-encryption service.
    """

    __tablename__ = "integration_credentials"

    integration_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "integrations.id",
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

    credential_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    encrypted_secret: Mapped[str] = mapped_column(
        String(2_000),
        nullable=False,
    )

    key_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    refresh_token_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    rotated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    rotated_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    credential_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    integration = relationship(
        "Integration",
        back_populates="credentials",
    )

    __table_args__ = (
        Index(
            "ix_integration_credentials_tenant_active",
            "tenant_id",
            "is_active",
        ),
        Index(
            "ix_integration_credentials_expiry",
            "expires_at",
        ),
    )

    @property
    def is_expired(self) -> bool:
        """Return whether the credential has expired."""
        if self.expires_at is None:
            return False

        return (
            datetime.now(
                self.expires_at.tzinfo
            )
            > self.expires_at
        )

    @property
    def is_usable(self) -> bool:
        """Return whether the credential can be used for authentication."""
        return (
            self.is_active
            and not self.is_expired
        )

    def __repr__(self) -> str:
        """Return a concise representation of the credential."""
        return (
            f"<IntegrationCredential id={self.id} "
            f"integration_id={self.integration_id} "
            f"type={self.credential_type!r}>"
        )


class IntegrationSyncState(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Synchronization cursor and state for an integration.

    Each integration has at most one sync-state record, which allows workers to
    resume incremental synchronization safely.
    """

    __tablename__ = "integration_sync_states"

    integration_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "integrations.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        unique=True,
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    cursor: Mapped[str | None] = mapped_column(
        String(1_000),
        nullable=True,
    )

    checkpoint: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    last_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    next_sync_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="idle",
        nullable=False,
        index=True,
    )

    records_synced: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    records_failed: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    sync_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    integration = relationship(
        "Integration",
        back_populates="sync_state",
    )

    __table_args__ = (
        Index(
            "ix_integration_sync_states_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_integration_sync_states_next_sync",
            "next_sync_at",
        ),
    )

    @property
    def is_running(self) -> bool:
        """Return whether a synchronization run is currently active."""
        return self.status == "running"

    def __repr__(self) -> str:
        """Return a concise representation of the sync state."""
        return (
            f"<IntegrationSyncState id={self.id} "
            f"integration_id={self.integration_id} "
            f"status={self.status!r}>"
        )


class IntegrationSyncJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    ORM model representing an individual synchronization run.

    Sync jobs record what was synchronized, how many records were processed,
    and whether the run succeeded or failed.
    """

    __tablename__ = "integration_sync_jobs"

    integration_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "integrations.id",
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

    sync_type: Mapped[str] = mapped_column(
        String(50),
        default="incremental",
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="queued",
        nullable=False,
        index=True,
    )

    trigger: Mapped[str | None] = mapped_column(
        String(50),
        default="manual",
        nullable=True,
        index=True,
    )

    records_processed: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    records_created: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    records_updated: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    records_failed: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    cursor_start: Mapped[str | None] = mapped_column(
        String(1_000),
        nullable=True,
    )

    cursor_end: Mapped[str | None] = mapped_column(
        String(1_000),
        nullable=True,
    )

    duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    job_metadata: Mapped[dict[str, Any]] = mapped_column(
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

    integration = relationship(
        "Integration",
        back_populates="sync_jobs",
    )

    __table_args__ = (
        Index(
            "ix_integration_sync_jobs_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_integration_sync_jobs_integration_status",
            "integration_id",
            "status",
        ),
    )

    @property
    def is_terminal(self) -> bool:
        """Return whether the sync job has reached a terminal state."""
        return self.status in {
            "completed",
            "failed",
            "cancelled",
        }

    def __repr__(self) -> str:
        """Return a concise representation of the sync job."""
        return (
            f"<IntegrationSyncJob id={self.id} "
            f"integration_id={self.integration_id} "
            f"status={self.status!r}>"
        )


class IntegrationWebhookRoute(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    ORM model representing an inbound webhook route for an integration.

    This model maps provider event types to an integration and records whether
    events for that route should be processed.
    """

    __tablename__ = "integration_webhook_routes"

    integration_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "integrations.id",
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

    provider: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        index=True,
    )

    target_queue: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    route_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    integration = relationship(
        "Integration",
        back_populates="webhook_routes",
    )

    __table_args__ = (
        UniqueConstraint(
            "integration_id",
            "event_type",
            name="uq_integration_webhook_routes_integration_event",
        ),
        Index(
            "ix_integration_webhook_routes_tenant_active",
            "tenant_id",
            "is_active",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the webhook route."""
        return (
            f"<IntegrationWebhookRoute id={self.id} "
            f"integration_id={self.integration_id} "
            f"event_type={self.event_type!r}>"
        )


class IntegrationAuditEvent(UUIDPrimaryKeyMixin, Base):
    """
    Immutable audit event for integration lifecycle and synchronization.
    """

    __tablename__ = "integration_audit_events"

    integration_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "integrations.id",
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

    integration = relationship(
        "Integration",
        back_populates="audit_events",
    )

    __table_args__ = (
        Index(
            "ix_integration_audit_events_tenant_time",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_integration_audit_events_integration_time",
            "integration_id",
            "occurred_at",
        ),
        Index(
            "ix_integration_audit_events_actor",
            "tenant_id",
            "actor_id",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the integration audit event."""
        return (
            f"<IntegrationAuditEvent id={self.id} "
            f"integration_id={self.integration_id} "
            f"event_type={self.event_type!r}>"
        )


__all__ = [
    "Integration",
    "IntegrationAuditEvent",
    "IntegrationCredential",
    "IntegrationSyncJob",
    "IntegrationSyncState",
    "IntegrationWebhookRoute",
]