"""
Shared SQLAlchemy model mixins, column helpers, and utilities.

This module provides reusable database building blocks for AgenticBridge ORM
models: UUID and integer primary keys, timestamps, soft deletion, tenant
isolation, audit fields, JSON metadata, and common domain helper methods.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def generate_uuid() -> uuid.UUID:
    """Generate a UUID suitable for a database default."""
    return uuid.uuid4()


def generate_uuid_string() -> str:
    """Generate a UUID string suitable for string-based primary keys."""
    return str(
        uuid.uuid4(),
    )


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(
        timezone.utc,
    )


def empty_dict() -> dict[str, Any]:
    """Return a new empty dictionary for JSON column defaults."""
    return {}


def empty_list() -> list[Any]:
    """Return a new empty list for JSON array column defaults."""
    return []


# ---------------------------------------------------------------------------
# Primary-key mixins
# ---------------------------------------------------------------------------


class UUIDPrimaryKeyMixin:
    """Adds a UUID primary key to an ORM model."""

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid,
        nullable=False,
    )


class StringUUIDPrimaryKeyMixin:
    """Adds a string-based UUID primary key to an ORM model."""

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid_string,
        nullable=False,
    )


class IntegerPrimaryKeyMixin:
    """Adds an auto-incrementing integer primary key to an ORM model."""

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        nullable=False,
    )


# ---------------------------------------------------------------------------
# Timestamp mixins
# ---------------------------------------------------------------------------


class TimestampMixin:
    """Adds created and updated timestamps to an ORM model."""

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


class LifecycleTimestampMixin:
    """
    Adds full lifecycle timestamps to an ORM model.

    This is useful for jobs, workflows, exports, and other long-running
    records that require started, completed, failed, and cancelled times.
    """

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    failed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    cancelled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------


class TenantMixin:
    """Adds a tenant identifier and tenant-scoped query helpers."""

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    @classmethod
    def tenant_filter(
        cls,
        tenant_id: uuid.UUID,
    ) -> dict[str, uuid.UUID]:
        """Return a filter dictionary scoped to one tenant."""
        return {
            "tenant_id": tenant_id,
        }


class TenantForeignKeyMixin:
    """
    Adds a tenant foreign key referencing the tenants table.

    Use this only for models where a database-level tenant relationship is
    required.
    """

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "tenants.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )


# ---------------------------------------------------------------------------
# Status and soft deletion
# ---------------------------------------------------------------------------


class StatusMixin:
    """Adds a status field and terminal-state helper."""

    status: Mapped[str] = mapped_column(
        String(50),
        default="active",
        nullable=False,
        index=True,
    )

    @property
    def is_active_status(self) -> bool:
        """Return whether the record has an active status."""
        return self.status == "active"


class SoftDeleteMixin:
    """Adds soft-deletion fields and helpers."""

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    deleted_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    @property
    def is_soft_deleted(self) -> bool:
        """Return whether the record has been soft deleted."""
        return self.is_deleted

    def mark_deleted(
        self,
        deleted_by: uuid.UUID | None = None,
    ) -> None:
        """Mark the record as soft deleted."""
        self.is_deleted = True
        self.deleted_at = utc_now()
        self.deleted_by = deleted_by


class ActivationMixin:
    """Adds an activation flag and activation timestamp."""

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    deactivated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def activate(self) -> None:
        """Mark the record as active."""
        self.is_active = True
        self.activated_at = utc_now()
        self.deactivated_at = None

    def deactivate(self) -> None:
        """Mark the record as inactive."""
        self.is_active = False
        self.deactivated_at = utc_now()


# ---------------------------------------------------------------------------
# Metadata and correlation
# ---------------------------------------------------------------------------


class JsonMetadataMixin:
    """Adds a JSONB metadata dictionary to an ORM model."""

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=empty_dict,
        nullable=False,
    )

    def get_metadata_value(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Return a value from the metadata dictionary."""
        return (
            self.metadata_json or {}
        ).get(
            key,
            default,
        )

    def set_metadata_value(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Set a value in the metadata dictionary."""
        metadata = dict(
            self.metadata_json or {},
        )

        metadata[key] = value

        self.metadata_json = metadata


class CorrelationMixin:
    """Adds request and correlation identifiers for distributed tracing."""

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


# ---------------------------------------------------------------------------
# Audit and integrity
# ---------------------------------------------------------------------------


class AuditMixin:
    """Adds creator and updater attribution fields."""

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )


class IntegrityMixin:
    """Adds checksum and integrity-verification fields."""

    checksum: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        index=True,
    )

    checksum_algorithm: Mapped[str | None] = mapped_column(
        String(50),
        default="sha256",
        nullable=True,
    )

    integrity_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


# ---------------------------------------------------------------------------
# Composite mixins
# ---------------------------------------------------------------------------


class AuditedTenantModelMixin(
    TenantMixin,
    AuditMixin,
    TimestampMixin,
):
    """
    Combines tenant isolation, attribution, and timestamps.

    This is a convenient base for most business records.
    """


class TrackedTenantModelMixin(
    TenantMixin,
    AuditMixin,
    TimestampMixin,
    SoftDeleteMixin,
    JsonMetadataMixin,
):
    """
    Combines tenant isolation, attribution, timestamps, soft deletion, and
    metadata.
    """


# ---------------------------------------------------------------------------
# Table argument helpers
# ---------------------------------------------------------------------------


def tenant_status_created_index(
    table_name: str,
) -> Index:
    """Create a standard tenant/status/created_at composite index."""
    return Index(
        f"ix_{table_name}_tenant_status_created",
        "tenant_id",
        "status",
        "created_at",
    )


def tenant_created_index(
    table_name: str,
) -> Index:
    """Create a standard tenant/created_at composite index."""
    return Index(
        f"ix_{table_name}_tenant_created",
        "tenant_id",
        "created_at",
    )


def resource_index(
    table_name: str,
) -> Index:
    """Create a standard tenant/resource_type/resource_id composite index."""
    return Index(
        f"ix_{table_name}_resource",
        "tenant_id",
        "resource_type",
        "resource_id",
    )


# ---------------------------------------------------------------------------
# Representation helper
# ---------------------------------------------------------------------------


def model_repr(
    model: Any,
    *fields: str,
) -> str:
    """
    Build a concise representation for an ORM model.

    Example:
        model_repr(obj, "id", "tenant_id", "status")
    """
    values = []

    for field_name in fields:
        value = getattr(
            model,
            field_name,
            None,
        )

        values.append(
            f"{field_name}={value!r}",
        )

    class_name = type(
        model,
    ).__name__

    return (
        f"<{class_name} "
        + " ".join(values)
        + ">"
    )


__all__ = [
    "ActivationMixin",
    "AuditMixin",
    "AuditedTenantModelMixin",
    "CorrelationMixin",
    "IntegerPrimaryKeyMixin",
    "IntegrityMixin",
    "JsonMetadataMixin",
    "LifecycleTimestampMixin",
    "SoftDeleteMixin",
    "StatusMixin",
    "StringUUIDPrimaryKeyMixin",
    "TenantForeignKeyMixin",
    "TenantMixin",
    "TimestampMixin",
    "TrackedTenantModelMixin",
    "UUIDPrimaryKeyMixin",
    "empty_dict",
    "empty_list",
    "generate_uuid",
    "generate_uuid_string",
    "model_repr",
    "resource_index",
    "tenant_created_index",
    "tenant_status_created_index",
    "utc_now",
]