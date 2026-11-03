"""
Webhook endpoint, event, delivery, subscription, and audit database models.

These models support tenant-scoped outbound webhook delivery, inbound signed
webhook ingestion, event subscriptions, delivery retries, signature verification,
dead-letter handling, and immutable webhook audit events.
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
    utc_now,
)


class WebhookEndpoint(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Tenant-configured outbound webhook endpoint.

    An endpoint receives selected event types from AgenticBridge. Signing
    secrets should be stored only as secure hashes; the raw secret should be
    displayed only once when the endpoint is created or rotated.
    """

    __tablename__ = "webhook_endpoints"

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

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    url: Mapped[str] = mapped_column(
        String(2_000),
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Security
    # -------------------------------------------------------------------------

    secret_hash: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    signature_algorithm: Mapped[str] = mapped_column(
        String(50),
        default="hmac-sha256",
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Subscription configuration
    # -------------------------------------------------------------------------

    event_types: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    headers: Mapped[dict[str, str]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    timeout_seconds: Mapped[int] = mapped_column(
        Integer,
        default=10,
        nullable=False,
    )

    max_attempts: Mapped[int] = mapped_column(
        Integer,
        default=5,
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

    status: Mapped[str] = mapped_column(
        String(50),
        default="active",
        nullable=False,
        index=True,
    )

    last_delivery_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_success_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_failure_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    consecutive_failures: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    disabled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    disabled_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    endpoint_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    deliveries = relationship(
        "WebhookDelivery",
        back_populates="endpoint",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    subscriptions = relationship(
        "WebhookSubscription",
        back_populates="endpoint",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    audit_events = relationship(
        "WebhookAuditEvent",
        back_populates="endpoint",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "name",
            name="uq_webhook_endpoints_tenant_name",
        ),
        Index(
            "ix_webhook_endpoints_tenant_active",
            "tenant_id",
            "is_active",
        ),
        Index(
            "ix_webhook_endpoints_tenant_status",
            "tenant_id",
            "status",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        """Return whether the endpoint is active and not failing repeatedly."""
        return (
            self.is_active
            and self.status == "active"
            and self.consecutive_failures == 0
        )

    def accepts_event_type(
        self,
        event_type: str,
    ) -> bool:
        """Return whether the endpoint subscribes to an event type."""
        return (
            self.is_active
            and (
                not self.event_types
                or event_type in self.event_types
            )
        )

    def record_success(self) -> None:
        """Record a successful webhook delivery."""
        self.last_delivery_at = utc_now()
        self.last_success_at = utc_now()
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """Record a failed webhook delivery."""
        self.last_delivery_at = utc_now()
        self.last_failure_at = utc_now()
        self.consecutive_failures += 1

    def disable(
        self,
        reason: str | None = None,
    ) -> None:
        """Disable the webhook endpoint."""
        self.is_active = False
        self.status = "disabled"
        self.disabled_at = utc_now()
        self.disabled_reason = reason

    def enable(self) -> None:
        """Enable the webhook endpoint."""
        self.is_active = True
        self.status = "active"
        self.disabled_at = None
        self.disabled_reason = None
        self.consecutive_failures = 0

    def __repr__(self) -> str:
        """Return a concise representation of the webhook endpoint."""
        return (
            f"<WebhookEndpoint id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"name={self.name!r} "
            f"status={self.status!r}>"
        )


class WebhookEvent(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Inbound signed webhook event.

    This model records events received from external providers before they are
    processed or fanned out to tenant-configured outbound endpoints.
    """

    __tablename__ = "webhook_events"

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    source: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    event_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        index=True,
    )

    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    signature: Mapped[str | None] = mapped_column(
        String(2_000),
        nullable=True,
    )

    signature_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="received",
        nullable=False,
        index=True,
    )

    attempt_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    failed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    event_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    deliveries = relationship(
        "WebhookDelivery",
        back_populates="webhook_event",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    audit_events = relationship(
        "WebhookAuditEvent",
        back_populates="webhook_event",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint(
            "source",
            "event_id",
            name="uq_webhook_events_source_event_id",
        ),
        Index(
            "ix_webhook_events_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_webhook_events_source_type",
            "source",
            "event_type",
        ),
        Index(
            "ix_webhook_events_received_at",
            "received_at",
        ),
    )

    @property
    def is_terminal(self) -> bool:
        """Return whether the webhook event has reached a terminal state."""
        return self.status in {
            "processed",
            "failed",
            "dead_lettered",
            "ignored",
        }

    def mark_processing(self) -> None:
        """Mark the webhook event as processing."""
        self.status = "processing"
        self.attempt_count += 1

    def mark_processed(self) -> None:
        """Mark the webhook event as successfully processed."""
        self.status = "processed"
        self.processed_at = utc_now()

    def mark_failed(
        self,
        error_message: str | None = None,
    ) -> None:
        """Mark the webhook event as failed."""
        self.status = "failed"
        self.failed_at = utc_now()
        self.error_message = error_message

    def mark_dead_lettered(
        self,
        error_message: str | None = None,
    ) -> None:
        """Move the webhook event to the dead-letter state."""
        self.status = "dead_lettered"
        self.failed_at = utc_now()
        self.error_message = error_message

    def __repr__(self) -> str:
        """Return a concise representation of the webhook event."""
        return (
            f"<WebhookEvent id={self.id} "
            f"source={self.source!r} "
            f"event_type={self.event_type!r} "
            f"status={self.status!r}>"
        )


class WebhookDelivery(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Outbound webhook delivery attempt.

    Each delivery represents one endpoint-specific attempt to deliver a webhook
    event, including retry state, response diagnostics, and final outcome.
    """

    __tablename__ = "webhook_deliveries"

    webhook_event_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "webhook_events.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    endpoint_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "webhook_endpoints.id",
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

    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
        index=True,
    )

    attempt_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    max_attempts: Mapped[int] = mapped_column(
        Integer,
        default=5,
        nullable=False,
    )

    request_headers: Mapped[dict[str, str]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    request_body: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    response_status_code: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    response_body: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    response_headers: Mapped[dict[str, str]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    next_retry_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    delivered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    failed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    delivery_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    webhook_event = relationship(
        "WebhookEvent",
        back_populates="deliveries",
    )

    endpoint = relationship(
        "WebhookEndpoint",
        back_populates="deliveries",
    )

    __table_args__ = (
        Index(
            "ix_webhook_deliveries_status_retry",
            "status",
            "next_retry_at",
        ),
        Index(
            "ix_webhook_deliveries_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_webhook_deliveries_endpoint_status",
            "endpoint_id",
            "status",
        ),
        Index(
            "ix_webhook_deliveries_event_endpoint",
            "webhook_event_id",
            "endpoint_id",
        ),
    )

    @property
    def is_terminal(self) -> bool:
        """Return whether the delivery has reached a terminal state."""
        return self.status in {
            "delivered",
            "failed",
            "cancelled",
            "dead_lettered",
        }

    @property
    def can_retry(self) -> bool:
        """Return whether another delivery attempt is permitted."""
        return (
            self.status == "failed"
            and self.attempt_count < self.max_attempts
        )

    def mark_delivered(self) -> None:
        """Mark the delivery as successfully delivered."""
        self.status = "delivered"
        self.delivered_at = utc_now()

    def mark_failed(
        self,
        error_message: str | None = None,
    ) -> None:
        """Mark the delivery attempt as failed."""
        self.status = "failed"
        self.failed_at = utc_now()
        self.error_message = error_message

    def mark_dead_lettered(
        self,
        error_message: str | None = None,
    ) -> None:
        """Move the delivery to the dead-letter state."""
        self.status = "dead_lettered"
        self.failed_at = utc_now()
        self.error_message = error_message

    def __repr__(self) -> str:
        """Return a concise representation of the webhook delivery."""
        return (
            f"<WebhookDelivery id={self.id} "
            f"event_id={self.webhook_event_id} "
            f"endpoint_id={self.endpoint_id} "
            f"status={self.status!r} "
            f"attempt={self.attempt_count}>"
        )


class WebhookSubscription(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Explicit event subscription for a webhook endpoint.

    Subscriptions allow fine-grained routing beyond the endpoint-level
    `event_types` list.
    """

    __tablename__ = "webhook_subscriptions"

    endpoint_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "webhook_endpoints.id",
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

    event_type: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
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

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    subscription_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    endpoint = relationship(
        "WebhookEndpoint",
        back_populates="subscriptions",
    )

    __table_args__ = (
        UniqueConstraint(
            "endpoint_id",
            "event_type",
            "resource_type",
            "resource_id",
            name="uq_webhook_subscriptions_endpoint_event_resource",
        ),
        Index(
            "ix_webhook_subscriptions_tenant_active",
            "tenant_id",
            "is_active",
        ),
        Index(
            "ix_webhook_subscriptions_event",
            "event_type",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the webhook subscription."""
        return (
            f"<WebhookSubscription id={self.id} "
            f"endpoint_id={self.endpoint_id} "
            f"event_type={self.event_type!r}>"
        )


class WebhookAuditEvent(UUIDPrimaryKeyMixin, Base):
    """
    Immutable audit event for webhook ingestion, delivery, and endpoint activity.
    """

    __tablename__ = "webhook_audit_events"

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    endpoint_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "webhook_endpoints.id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    webhook_event_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "webhook_events.id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    delivery_id: Mapped[uuid.UUID | None] = mapped_column(
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

    endpoint = relationship(
        "WebhookEndpoint",
        back_populates="audit_events",
    )

    webhook_event = relationship(
        "WebhookEvent",
        back_populates="audit_events",
    )

    __table_args__ = (
        Index(
            "ix_webhook_audit_events_tenant_time",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_webhook_audit_events_endpoint_time",
            "endpoint_id",
            "occurred_at",
        ),
        Index(
            "ix_webhook_audit_events_event_time",
            "webhook_event_id",
            "occurred_at",
        ),
        Index(
            "ix_webhook_audit_events_type_time",
            "event_type",
            "occurred_at",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the webhook audit event."""
        return (
            f"<WebhookAuditEvent id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"event_type={self.event_type!r}>"
        )


__all__ = [
    "WebhookAuditEvent",
    "WebhookDelivery",
    "WebhookEndpoint",
    "WebhookEvent",
    "WebhookSubscription",
]