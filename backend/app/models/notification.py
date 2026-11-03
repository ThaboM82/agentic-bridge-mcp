"""
Notification, preference, delivery, template, and audit database models.

These models support tenant-scoped in-app, email, webhook, and push
notifications. They include user delivery preferences, notification templates,
delivery retries, read state, scheduling, and notification audit events.
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


class Notification(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Tenant-scoped notification.

    A notification represents an event that should be surfaced to a user or
    tenant. It can be delivered through one or more channels and tracks its own
    read state independently from delivery attempts.
    """

    __tablename__ = "notifications"

    # -------------------------------------------------------------------------
    # Identity and ownership
    # -------------------------------------------------------------------------

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    notification_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    category: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(300),
        nullable=False,
    )

    body: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(50),
        default="unread",
        nullable=False,
        index=True,
    )

    priority: Mapped[int] = mapped_column(
        Integer,
        default=5,
        nullable=False,
        index=True,
    )

    is_action_required: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    action_url: Mapped[str | None] = mapped_column(
        String(2_000),
        nullable=True,
    )

    scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    read_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Context and metadata
    # -------------------------------------------------------------------------

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

    execution_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    correlation_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    notification_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    deliveries = relationship(
        "NotificationDelivery",
        back_populates="notification",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    audit_events = relationship(
        "NotificationAuditEvent",
        back_populates="notification",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        Index(
            "ix_notifications_tenant_user_status",
            "tenant_id",
            "user_id",
            "status",
        ),
        Index(
            "ix_notifications_tenant_type_created",
            "tenant_id",
            "notification_type",
            "created_at",
        ),
        Index(
            "ix_notifications_tenant_resource",
            "tenant_id",
            "resource_type",
            "resource_id",
        ),
        Index(
            "ix_notifications_scheduled",
            "scheduled_at",
        ),
        Index(
            "ix_notifications_expiry",
            "expires_at",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_read(self) -> bool:
        """Return whether the notification has been read."""
        return self.read_at is not None

    @property
    def is_archived(self) -> bool:
        """Return whether the notification has been archived."""
        return self.archived_at is not None

    @property
    def is_expired(self) -> bool:
        """Return whether the notification has expired."""
        if self.expires_at is None:
            return False

        return (
            datetime.now(
                self.expires_at.tzinfo
            )
            > self.expires_at
        )

    def mark_read(self) -> None:
        """Mark the notification as read."""
        if not self.is_read:
            self.status = "read"
            self.read_at = utc_now()

    def mark_archived(self) -> None:
        """Mark the notification as archived."""
        self.status = "archived"
        self.archived_at = utc_now()

    def __repr__(self) -> str:
        """Return a concise representation of the notification."""
        return (
            f"<Notification id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"user_id={self.user_id} "
            f"type={self.notification_type!r} "
            f"status={self.status!r}>"
        )


class NotificationPreference(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    User notification preferences.

    Preferences are unique per tenant, user, and notification type. A missing
    preference record can be treated as application defaults.
    """

    __tablename__ = "notification_preferences"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    notification_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    email_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    in_app_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    webhook_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    push_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    sms_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    quiet_hours_start: Mapped[str | None] = mapped_column(
        String(5),
        nullable=True,
    )

    quiet_hours_end: Mapped[str | None] = mapped_column(
        String(5),
        nullable=True,
    )

    timezone: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    preference_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "user_id",
            "notification_type",
            name="uq_notification_preferences_user_type",
        ),
        Index(
            "ix_notification_preferences_tenant_user",
            "tenant_id",
            "user_id",
        ),
    )

    def is_channel_enabled(
        self,
        channel: str,
    ) -> bool:
        """Return whether a delivery channel is enabled."""
        channel_flags = {
            "email": self.email_enabled,
            "in_app": self.in_app_enabled,
            "webhook": self.webhook_enabled,
            "push": self.push_enabled,
            "sms": self.sms_enabled,
        }

        return channel_flags.get(
            channel,
            False,
        )

    def __repr__(self) -> str:
        """Return a concise representation of notification preferences."""
        return (
            f"<NotificationPreference id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"user_id={self.user_id} "
            f"type={self.notification_type!r}>"
        )


class NotificationDelivery(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Delivery attempt for a notification.

    Each delivery record tracks one channel-specific attempt, including retry
    state, provider response, and failure diagnostics.
    """

    __tablename__ = "notification_deliveries"

    notification_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "notifications.id",
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

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    channel: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    recipient: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
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
        default=3,
        nullable=False,
    )

    next_retry_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    provider: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    provider_message_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    response_status_code: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    response_body: Mapped[str | None] = mapped_column(
        Text,
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

    notification = relationship(
        "Notification",
        back_populates="deliveries",
    )

    __table_args__ = (
        Index(
            "ix_notification_deliveries_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_notification_deliveries_notification_channel",
            "notification_id",
            "channel",
        ),
        Index(
            "ix_notification_deliveries_retry",
            "status",
            "next_retry_at",
        ),
    )

    @property
    def is_terminal(self) -> bool:
        """Return whether the delivery has reached a terminal state."""
        return self.status in {
            "delivered",
            "failed",
            "cancelled",
        }

    @property
    def can_retry(self) -> bool:
        """Return whether another delivery attempt is permitted."""
        return (
            self.status == "failed"
            and self.attempt_count
            < self.max_attempts
        )

    def __repr__(self) -> str:
        """Return a concise representation of the delivery."""
        return (
            f"<NotificationDelivery id={self.id} "
            f"notification_id={self.notification_id} "
            f"channel={self.channel!r} "
            f"status={self.status!r}>"
        )


class NotificationTemplate(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Reusable notification template.

    Templates may be global when `tenant_id` is null, or tenant-specific when a
    tenant is specified.
    """

    __tablename__ = "notification_templates"

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

    notification_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    channel: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    subject_template: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    body_template: Mapped[str] = mapped_column(
        Text,
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
            "notification_type",
            "channel",
            "name",
            name="uq_notification_templates_tenant_type_channel_name",
        ),
        Index(
            "ix_notification_templates_tenant_active",
            "tenant_id",
            "is_active",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the notification template."""
        return (
            f"<NotificationTemplate id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"name={self.name!r} "
            f"channel={self.channel!r}>"
        )


class NotificationAuditEvent(UUIDPrimaryKeyMixin, Base):
    """
    Immutable audit event for notification lifecycle and delivery activity.
    """

    __tablename__ = "notification_audit_events"

    notification_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "notifications.id",
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

    channel: Mapped[str | None] = mapped_column(
        String(50),
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

    notification = relationship(
        "Notification",
        back_populates="audit_events",
    )

    __table_args__ = (
        Index(
            "ix_notification_audit_events_tenant_time",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_notification_audit_events_notification_time",
            "notification_id",
            "occurred_at",
        ),
        Index(
            "ix_notification_audit_events_actor",
            "tenant_id",
            "actor_id",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the notification audit event."""
        return (
            f"<NotificationAuditEvent id={self.id} "
            f"notification_id={self.notification_id} "
            f"event_type={self.event_type!r}>"
        )


__all__ = [
    "Notification",
    "NotificationAuditEvent",
    "NotificationDelivery",
    "NotificationPreference",
    "NotificationTemplate",
]