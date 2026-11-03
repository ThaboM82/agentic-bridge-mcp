"""
Generated artifact, version, download-token, and audit database models.

These models store tenant-scoped metadata for workflow-generated artifacts,
immutable artifact versions, short-lived secure download references, retention
state, and artifact audit events. Artifact contents remain in object storage;
the database stores only metadata and secure references.
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


class Artifact(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Tenant-scoped generated artifact metadata.

    An artifact represents a file or object produced by a workflow, ingestion
    job, or external integration. Its binary content is stored externally; this
    record tracks its storage location, integrity, lifecycle, and access.
    """

    __tablename__ = "artifacts"

    # -------------------------------------------------------------------------
    # Identity and ownership
    # -------------------------------------------------------------------------

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    execution_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "workflow_executions.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    artifact_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(300),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Storage and integrity
    # -------------------------------------------------------------------------

    storage_provider: Mapped[str] = mapped_column(
        String(100),
        default="s3",
        nullable=False,
        index=True,
    )

    storage_bucket: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    storage_key: Mapped[str] = mapped_column(
        String(1_000),
        nullable=False,
    )

    file_size_bytes: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

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

    mime_type: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(50),
        default="created",
        nullable=False,
        index=True,
    )

    is_current: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    is_public: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    retention_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
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

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    artifact_metadata: Mapped[dict[str, Any]] = mapped_column(
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

    workflow_execution = relationship(
        "WorkflowExecutionModel",
        back_populates="artifacts",
        foreign_keys=[execution_id],
    )

    versions = relationship(
        "ArtifactVersion",
        back_populates="artifact",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    download_tokens = relationship(
        "ArtifactDownloadToken",
        back_populates="artifact",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    audit_events = relationship(
        "ArtifactAuditEvent",
        back_populates="artifact",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        Index(
            "ix_artifacts_tenant_status_created",
            "tenant_id",
            "status",
            "created_at",
        ),
        Index(
            "ix_artifacts_execution_type",
            "execution_id",
            "artifact_type",
        ),
        Index(
            "ix_artifacts_tenant_type_status",
            "tenant_id",
            "artifact_type",
            "status",
        ),
        Index(
            "ix_artifacts_tenant_current",
            "tenant_id",
            "is_current",
        ),
        Index(
            "ix_artifacts_retention_expiry",
            "retention_expires_at",
        ),
        Index(
            "ix_artifacts_storage_key",
            "storage_key",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_deleted(self) -> bool:
        """Return whether the artifact has been deleted."""
        return self.deleted_at is not None

    @property
    def is_expired(self) -> bool:
        """Return whether the artifact has passed its retention deadline."""
        if self.retention_expires_at is None:
            return False

        return (
            datetime.now(
                self.retention_expires_at.tzinfo
            )
            > self.retention_expires_at
        )

    @property
    def is_downloadable(self) -> bool:
        """Return whether the artifact can be downloaded."""
        return (
            not self.is_deleted
            and not self.is_expired
            and self.status
            in {
                "created",
                "uploaded",
                "ready",
            }
        )

    def __repr__(self) -> str:
        """Return a concise representation of the artifact."""
        return (
            f"<Artifact id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"execution_id={self.execution_id} "
            f"type={self.artifact_type!r} "
            f"status={self.status!r}>"
        )


class ArtifactVersion(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Immutable artifact version.

    Each version records a distinct stored revision of an artifact, including
    its storage location, integrity metadata, and creator.
    """

    __tablename__ = "artifact_versions"

    artifact_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "artifacts.id",
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

    storage_provider: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    storage_bucket: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    storage_key: Mapped[str] = mapped_column(
        String(1_000),
        nullable=False,
    )

    file_size_bytes: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    checksum: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
    )

    checksum_algorithm: Mapped[str | None] = mapped_column(
        String(50),
        default="sha256",
        nullable=True,
    )

    mime_type: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    version_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    artifact = relationship(
        "Artifact",
        back_populates="versions",
    )

    __table_args__ = (
        UniqueConstraint(
            "artifact_id",
            "version_number",
            name="uq_artifact_versions_artifact_version",
        ),
        Index(
            "ix_artifact_versions_tenant_created",
            "tenant_id",
            "created_at",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the artifact version."""
        return (
            f"<ArtifactVersion id={self.id} "
            f"artifact_id={self.artifact_id} "
            f"version={self.version_number}>"
        )


class ArtifactDownloadToken(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Short-lived secure download reference for an artifact.

    Only the token hash is persisted. The plaintext token is returned to the
    caller once when issued and can be revoked or expire independently of the
    artifact.
    """

    __tablename__ = "artifact_download_tokens"

    artifact_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "artifacts.id",
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

    issued_to: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    issued_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
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

    used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    used_by_ip: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    use_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    max_use_count: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    token_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    artifact = relationship(
        "Artifact",
        back_populates="download_tokens",
    )

    __table_args__ = (
        Index(
            "ix_artifact_download_tokens_tenant_expiry",
            "tenant_id",
            "expires_at",
        ),
        Index(
            "ix_artifact_download_tokens_artifact_active",
            "artifact_id",
            "is_active",
        ),
    )

    @property
    def is_expired(self) -> bool:
        """Return whether the download token has expired."""
        return (
            datetime.now(
                self.expires_at.tzinfo
            )
            > self.expires_at
        )

    @property
    def is_revoked(self) -> bool:
        """Return whether the download token has been revoked."""
        return self.revoked_at is not None

    @property
    def is_usable(self) -> bool:
        """Return whether the token can still be used."""
        return (
            self.is_active
            and not self.is_expired
            and not self.is_revoked
            and self.use_count
            < self.max_use_count
        )

    def __repr__(self) -> str:
        """Return a concise representation of the download token."""
        return (
            f"<ArtifactDownloadToken id={self.id} "
            f"artifact_id={self.artifact_id} "
            f"expires_at={self.expires_at}>"
        )


class ArtifactAuditEvent(UUIDPrimaryKeyMixin, Base):
    """
    Immutable audit event for artifact lifecycle and access activity.
    """

    __tablename__ = "artifact_audit_events"

    artifact_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "artifacts.id",
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

    client_ip: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
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

    artifact = relationship(
        "Artifact",
        back_populates="audit_events",
    )

    __table_args__ = (
        Index(
            "ix_artifact_audit_events_tenant_time",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_artifact_audit_events_artifact_time",
            "artifact_id",
            "occurred_at",
        ),
        Index(
            "ix_artifact_audit_events_actor",
            "tenant_id",
            "actor_id",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the artifact audit event."""
        return (
            f"<ArtifactAuditEvent id={self.id} "
            f"artifact_id={self.artifact_id} "
            f"event_type={self.event_type!r}>"
        )


__all__ = [
    "Artifact",
    "ArtifactAuditEvent",
    "ArtifactDownloadToken",
    "ArtifactVersion",
]