"""
User, authentication-session, password-reset, and audit models.

These models support tenant-scoped user accounts, authentication sessions,
refresh-token rotation, multi-factor authentication, password resets, account
lockout, user preferences, and immutable user audit events.

API-key models are owned by `app.models.api_key` to ensure the `api_keys`
table is registered only once.
"""

from __future__ import annotations

import uuid
from datetime import (
    datetime,
    timedelta,
)
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import (
    UUID as PGUUID,
)
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


class User(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    Tenant-scoped user account.

    A user belongs to exactly one tenant. Email addresses are unique within a
    tenant, allowing the same person to have accounts in multiple tenants.
    """

    __tablename__ = "users"

    # -------------------------------------------------------------------------
    # Identity and ownership
    # -------------------------------------------------------------------------

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

    full_name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    display_name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    avatar_url: Mapped[str | None] = mapped_column(
        String(2_000),
        nullable=True,
    )

    external_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------

    hashed_password: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    password_changed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    must_change_password: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Authorization
    # -------------------------------------------------------------------------

    role: Mapped[str] = mapped_column(
        String(100),
        default="member",
        nullable=False,
        index=True,
    )

    permissions: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Security
    # -------------------------------------------------------------------------

    mfa_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    mfa_secret_encrypted: Mapped[str | None] = mapped_column(
        String(1_000),
        nullable=True,
    )

    mfa_backup_codes_hashed: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    failed_login_attempts: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_login_ip: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    last_login_user_agent: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Profile and preferences
    # -------------------------------------------------------------------------

    timezone: Mapped[str | None] = mapped_column(
        String(100),
        default="UTC",
        nullable=True,
    )

    locale: Mapped[str | None] = mapped_column(
        String(20),
        default="en",
        nullable=True,
    )

    phone_number: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    job_title: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    preferences: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    user_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    deactivated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    deactivated_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    tenant = relationship(
        "Tenant",
        back_populates="users",
    )

    sessions = relationship(
        "UserSession",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    api_keys = relationship(
        "ApiKey",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    password_resets = relationship(
        "PasswordResetToken",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    audit_events = relationship(
        "UserAuditEvent",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "email",
            name="uq_users_tenant_email",
        ),
        UniqueConstraint(
            "tenant_id",
            "external_id",
            name="uq_users_tenant_external_id",
        ),
        Index(
            "ix_users_tenant_role_active",
            "tenant_id",
            "role",
            "is_active",
        ),
        Index(
            "ix_users_tenant_status",
            "tenant_id",
            "is_active",
        ),
        Index(
            "ix_users_locked_until",
            "locked_until",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_locked(self) -> bool:
        """Return whether the account is temporarily locked."""
        if self.locked_until is None:
            return False

        return (
            datetime.now(
                self.locked_until.tzinfo,
            )
            < self.locked_until
        )

    @property
    def is_deleted(self) -> bool:
        """Return whether the user has been deleted."""
        return self.deleted_at is not None

    def record_successful_login(
        self,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        """Record a successful authentication attempt."""
        self.last_login_at = utc_now()
        self.last_login_ip = ip_address
        self.last_login_user_agent = user_agent
        self.failed_login_attempts = 0
        self.locked_until = None

    def record_failed_login(
        self,
        max_attempts: int = 5,
        lockout_minutes: int = 15,
    ) -> None:
        """Record a failed authentication attempt and apply lockout."""
        self.failed_login_attempts += 1

        if (
            self.failed_login_attempts
            >= max_attempts
        ):
            self.locked_until = utc_now() + timedelta(
                minutes=lockout_minutes,
            )

    def deactivate(
        self,
        deactivated_by: uuid.UUID | None = None,
    ) -> None:
        """Deactivate the user account."""
        self.is_active = False
        self.deactivated_at = utc_now()
        self.deactivated_by = deactivated_by

    def reactivate(self) -> None:
        """Reactivate the user account."""
        self.is_active = True
        self.deactivated_at = None
        self.deactivated_by = None

    def mark_deleted(self) -> None:
        """Mark the user as deleted."""
        self.is_active = False
        self.deleted_at = utc_now()

    def __repr__(self) -> str:
        """Return a concise representation of the user."""
        return (
            f"<User id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"email={self.email!r} "
            f"role={self.role!r}>"
        )


class UserSession(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    Authenticated user session or refresh-token record.

    Session and refresh tokens must be stored only as secure hashes. Raw token
    values should never be persisted.
    """

    __tablename__ = "user_sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "users.id",
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

    session_token_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
    )

    refresh_token_hash: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    refresh_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    revoked_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    revoked_reason: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    user_agent: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    client_ip: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    device_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    session_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    user = relationship(
        "User",
        back_populates="sessions",
    )

    __table_args__ = (
        Index(
            "ix_user_sessions_tenant_active_expiry",
            "tenant_id",
            "is_active",
            "expires_at",
        ),
        Index(
            "ix_user_sessions_user_active_expiry",
            "user_id",
            "is_active",
            "expires_at",
        ),
        Index(
            "ix_user_sessions_refresh_expiry",
            "refresh_expires_at",
        ),
    )

    @property
    def is_expired(self) -> bool:
        """Return whether the access session has expired."""
        return (
            datetime.now(
                self.expires_at.tzinfo,
            )
            > self.expires_at
        )

    @property
    def is_refresh_expired(self) -> bool:
        """Return whether the refresh token has expired."""
        if self.refresh_expires_at is None:
            return True

        return (
            datetime.now(
                self.refresh_expires_at.tzinfo,
            )
            > self.refresh_expires_at
        )

    @property
    def is_valid(self) -> bool:
        """Return whether the session can be used for authentication."""
        return (
            self.is_active
            and not self.is_expired
            and self.revoked_at is None
        )

    def revoke(
        self,
        revoked_by: uuid.UUID | None = None,
        reason: str | None = None,
    ) -> None:
        """Revoke the session."""
        self.is_active = False
        self.revoked_at = utc_now()
        self.revoked_by = revoked_by
        self.revoked_reason = reason

    def __repr__(self) -> str:
        """Return a concise representation of the user session."""
        return (
            f"<UserSession id={self.id} "
            f"user_id={self.user_id} "
            f"is_active={self.is_active} "
            f"expires_at={self.expires_at}>"
        )


class PasswordResetToken(
    UUIDPrimaryKeyMixin,
    Base,
):
    """
    Single-use password reset token.

    Reset tokens must be stored only as secure hashes and should expire quickly.
    """

    __tablename__ = "password_reset_tokens"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "users.id",
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

    token_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
    )

    is_used: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    requested_ip: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    requested_user_agent: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    user = relationship(
        "User",
        back_populates="password_resets",
    )

    __table_args__ = (
        Index(
            "ix_password_reset_tokens_user_expiry",
            "user_id",
            "expires_at",
        ),
    )

    @property
    def is_expired(self) -> bool:
        """Return whether the reset token has expired."""
        return (
            datetime.now(
                self.expires_at.tzinfo,
            )
            > self.expires_at
        )

    @property
    def is_valid(self) -> bool:
        """Return whether the reset token can be used."""
        return (
            not self.is_used
            and not self.is_expired
        )

    def mark_used(self) -> None:
        """Mark the reset token as used."""
        self.is_used = True
        self.used_at = utc_now()

    def __repr__(self) -> str:
        """Return a concise representation of the password reset token."""
        return (
            f"<PasswordResetToken id={self.id} "
            f"user_id={self.user_id} "
            f"is_used={self.is_used}>"
        )


class UserAuditEvent(
    UUIDPrimaryKeyMixin,
    Base,
):
    """
    Immutable audit event for user lifecycle and authentication activity.
    """

    __tablename__ = "user_audit_events"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "users.id",
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

    client_ip: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    user_agent: Mapped[str | None] = mapped_column(
        String(500),
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

    user = relationship(
        "User",
        back_populates="audit_events",
    )

    __table_args__ = (
        Index(
            "ix_user_audit_events_tenant_time",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_user_audit_events_user_time",
            "user_id",
            "occurred_at",
        ),
        Index(
            "ix_user_audit_events_actor",
            "tenant_id",
            "actor_id",
        ),
        Index(
            "ix_user_audit_events_type_time",
            "event_type",
            "occurred_at",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the user audit event."""
        return (
            f"<UserAuditEvent id={self.id} "
            f"user_id={self.user_id} "
            f"event_type={self.event_type!r}>"
        )


__all__ = [
    "PasswordResetToken",
    "User",
    "UserAuditEvent",
    "UserSession",
]