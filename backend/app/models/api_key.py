"""
API-key database models for machine-to-machine authentication.

These models store tenant-scoped API keys, hashed key material, authorization
scopes, usage telemetry, rotation history, and revocation state. Raw API keys
are never persisted; only a non-sensitive prefix and a secure hash are stored.
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
)


class ApiKey(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    Tenant-scoped API key with explicit authorization scopes.

    An API key can be associated with a user for attribution, or used as a
    standalone machine identity. The plaintext key is returned only once at
    creation; the database retains only its prefix and hash.
    """

    __tablename__ = "api_keys"

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

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "users.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        String(1_000),
        nullable=True,
    )

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Key material
    # -------------------------------------------------------------------------

    key_prefix: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        index=True,
    )

    key_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
    )

    key_fingerprint: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Authorization
    # -------------------------------------------------------------------------

    scopes: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    role: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    allowed_ip_addresses: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    allowed_referrers: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
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

    expires_at: Mapped[datetime | None] = mapped_column(
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

    revocation_reason: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_used_ip: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Usage and rate limiting
    # -------------------------------------------------------------------------

    usage_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    rate_limit_per_minute: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    rate_limit_per_hour: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    rate_limit_per_day: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    api_key_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    tenant = relationship(
        "Tenant",
        backref="api_keys",
    )

    user = relationship(
        "User",
        back_populates="api_keys",
    )

    rotations = relationship(
        "ApiKeyRotation",
        back_populates="api_key",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    usage_events = relationship(
        "ApiKeyUsageEvent",
        back_populates="api_key",
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
            name="uq_api_keys_tenant_name",
        ),
        Index(
            "ix_api_keys_tenant_active_expiry",
            "tenant_id",
            "is_active",
            "expires_at",
        ),
        Index(
            "ix_api_keys_tenant_status_created",
            "tenant_id",
            "status",
            "created_at",
        ),
        Index(
            "ix_api_keys_tenant_user",
            "tenant_id",
            "user_id",
        ),
        Index(
            "ix_api_keys_key_prefix",
            "key_prefix",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_expired(self) -> bool:
        """Return whether the key has passed its expiration time."""
        if self.expires_at is None:
            return False

        return (
            datetime.now(
                self.expires_at.tzinfo,
            )
            >= self.expires_at
        )

    @property
    def is_revoked(self) -> bool:
        """Return whether the key has been revoked."""
        return self.revoked_at is not None

    @property
    def is_usable(self) -> bool:
        """Return whether the key can authenticate requests."""
        return (
            self.is_active
            and not self.is_revoked
            and not self.is_expired
        )

    def has_scope(
        self,
        required_scope: str,
    ) -> bool:
        """Return whether the key has a required scope."""
        return required_scope in self.scopes

    def has_any_scope(
        self,
        required_scopes: list[str]
        | tuple[str, ...],
    ) -> bool:
        """Return whether the key has at least one required scope."""
        return any(
            required_scope in self.scopes
            for required_scope in required_scopes
        )

    def has_all_scopes(
        self,
        required_scopes: list[str]
        | tuple[str, ...],
    ) -> bool:
        """Return whether the key has every required scope."""
        return all(
            required_scope in self.scopes
            for required_scope in required_scopes
        )

    def __repr__(self) -> str:
        """Return a concise representation of the API key."""
        return (
            f"<ApiKey id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"name={self.name!r} "
            f"prefix={self.key_prefix!r} "
            f"status={self.status!r}>"
        )


class ApiKeyRotation(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    ORM model representing an API-key rotation event.

    Rotations preserve an audit trail when a key is replaced, while ensuring
    that only hashed key material is stored.
    """

    __tablename__ = "api_key_rotations"

    api_key_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "api_keys.id",
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

    previous_key_prefix: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )

    new_key_prefix: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )

    previous_key_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    new_key_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    rotated_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    rotation_reason: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    rotated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    api_key = relationship(
        "ApiKey",
        back_populates="rotations",
    )

    __table_args__ = (
        Index(
            "ix_api_key_rotations_tenant_created",
            "tenant_id",
            "created_at",
        ),
        Index(
            "ix_api_key_rotations_api_key",
            "api_key_id",
            "rotated_at",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the key rotation."""
        return (
            f"<ApiKeyRotation id={self.id} "
            f"api_key_id={self.api_key_id} "
            f"rotated_at={self.rotated_at}>"
        )


class ApiKeyUsageEvent(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    ORM model representing an authenticated API-key request.

    This model supports security auditing, anomaly detection, usage analytics,
    and per-key rate-limit diagnostics.
    """

    __tablename__ = "api_key_usage_events"

    api_key_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "api_keys.id",
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

    request_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    method: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
    )

    path: Mapped[str | None] = mapped_column(
        String(1_000),
        nullable=True,
        index=True,
    )

    status_code: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        index=True,
    )

    client_ip: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    user_agent: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    outcome: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    usage_metadata: Mapped[dict[str, Any]] = mapped_column(
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

    api_key = relationship(
        "ApiKey",
        back_populates="usage_events",
    )

    __table_args__ = (
        Index(
            "ix_api_key_usage_events_tenant_time",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_api_key_usage_events_key_time",
            "api_key_id",
            "occurred_at",
        ),
        Index(
            "ix_api_key_usage_events_tenant_status",
            "tenant_id",
            "status_code",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the usage event."""
        return (
            f"<ApiKeyUsageEvent id={self.id} "
            f"api_key_id={self.api_key_id} "
            f"path={self.path!r} "
            f"status_code={self.status_code}>"
        )


__all__ = [
    "ApiKey",
    "ApiKeyRotation",
    "ApiKeyUsageEvent",
]