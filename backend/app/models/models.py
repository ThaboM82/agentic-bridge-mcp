"""
Core ORM models for AgenticBridge.

This module defines the shared document, chunk, embedding, tenant, user, and
workflow-related database models used by the API, retrieval engine, and
background workers.
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
    JSON,
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
from pgvector.sqlalchemy import Vector

from app.core.database import Base


EMBEDDING_DIMENSIONS = 1536


def _generate_uuid() -> uuid.UUID:
    """Generate a database-side UUID default."""
    return uuid.uuid4()


# ---------------------------------------------------------------------------
# Tenant model
# ---------------------------------------------------------------------------


class Tenant(Base):
    """
    ORM model representing a tenant workspace.

    Tenants provide the primary isolation boundary for users, documents,
    workflows, usage, and audit events.
    """

    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=_generate_uuid,
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    slug: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

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

    settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

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

    users: Mapped[list["User"]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    documents: Mapped[list["Document"]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint(
            "slug",
            name="uq_tenants_slug",
        ),
        Index(
            "ix_tenants_status_created",
            "status",
            "created_at",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the tenant."""
        return (
            f"<Tenant id={self.id} "
            f"name={self.name!r} "
            f"status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------


class User(Base):
    """
    ORM model representing a tenant-scoped user.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=_generate_uuid,
        nullable=False,
    )

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

    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        String(100),
        default="member",
        nullable=False,
        index=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

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

    tenant: Mapped[Tenant] = relationship(
        back_populates="users",
    )

    documents: Mapped[list["Document"]] = relationship(
        back_populates="created_by_user",
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "email",
            name="uq_users_tenant_email",
        ),
        Index(
            "ix_users_tenant_role_active",
            "tenant_id",
            "role",
            "is_active",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the user."""
        return (
            f"<User id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"email={self.email!r} "
            f"role={self.role!r}>"
        )


# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------


class Document(Base):
    """
    ORM model representing a tenant-scoped source document.

    A document is the logical parent of embedded chunks used for semantic
    retrieval.
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=_generate_uuid,
        nullable=False,
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "tenants.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "users.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    title: Mapped[str] = mapped_column(
        String(300),
        nullable=False,
    )

    document_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    source: Mapped[str | None] = mapped_column(
        String(300),
        nullable=True,
        index=True,
    )

    external_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    mime_type: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    file_size_bytes: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    checksum: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="processing",
        nullable=False,
        index=True,
    )

    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
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

    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True,
    )

    tenant: Mapped[Tenant] = relationship(
        back_populates="documents",
    )

    created_by_user: Mapped[User | None] = relationship(
        back_populates="documents",
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "external_id",
            name="uq_documents_tenant_external_id",
        ),
        Index(
            "ix_documents_tenant_status_created",
            "tenant_id",
            "status",
            "created_at",
        ),
        Index(
            "ix_documents_tenant_type",
            "tenant_id",
            "document_type",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the document."""
        return (
            f"<Document id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"title={self.title!r} "
            f"status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# Document chunk model
# ---------------------------------------------------------------------------


class DocumentChunk(Base):
    """
    ORM model representing a processed and embedded document chunk.

    Each chunk contains a portion of a source document and an optional vector
    embedding used for pgvector-based semantic retrieval.
    """

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        index=True,
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "documents.id",
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

    title: Mapped[str] = mapped_column(
        String(300),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    content_hash: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        index=True,
    )

    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
        nullable=True,
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    embedding_status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
        index=True,
    )

    meta_data: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

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

    document: Mapped[Document] = relationship(
        back_populates="chunks",
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_chunk_index",
        ),
        Index(
            "ix_document_chunks_tenant_active",
            "tenant_id",
            "is_active",
        ),
        Index(
            "ix_document_chunks_document_tenant",
            "document_id",
            "tenant_id",
        ),
        Index(
            "ix_document_chunks_embedding_status",
            "embedding_status",
        ),
        Index(
            "ix_document_chunks_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={
                "embedding": "vector_cosine_ops"
            },
            postgresql_with={
                "lists": 100
            },
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the document chunk."""
        return (
            f"<DocumentChunk id={self.id} "
            f"document_id={self.document_id} "
            f"chunk_index={self.chunk_index} "
            f"embedding_status={self.embedding_status!r}>"
        )


# ---------------------------------------------------------------------------
# Workflow execution model
# ---------------------------------------------------------------------------


class WorkflowExecution(Base):
    """
    ORM model representing a tenant-scoped workflow execution.
    """

    __tablename__ = "workflow_executions"

    execution_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=_generate_uuid,
        nullable=False,
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "tenants.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "users.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    workflow_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(300),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="queued",
        nullable=False,
        index=True,
    )

    current_step: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    input_data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    result_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    cancel_requested: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

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

    tenant: Mapped[Tenant] = relationship()

    created_by_user: Mapped[User | None] = relationship()

    approvals: Mapped[list["Approval"]] = relationship(
        back_populates="workflow_execution",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    artifacts: Mapped[list["Artifact"]] = relationship(
        back_populates="workflow_execution",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index(
            "ix_workflow_executions_tenant_status_created",
            "tenant_id",
            "status",
            "created_at",
        ),
        Index(
            "ix_workflow_executions_tenant_type",
            "tenant_id",
            "workflow_type",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the workflow execution."""
        return (
            f"<WorkflowExecution execution_id={self.execution_id} "
            f"tenant_id={self.tenant_id} "
            f"workflow_type={self.workflow_type!r} "
            f"status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# Approval model
# ---------------------------------------------------------------------------


class Approval(Base):
    """
    ORM model representing a human approval request for a workflow step.
    """

    __tablename__ = "approvals"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=_generate_uuid,
        nullable=False,
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    execution_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "workflow_executions.execution_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    workflow_step: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
        index=True,
    )

    decision: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    comment: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    requested_by: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
    )

    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    workflow_execution: Mapped[WorkflowExecution] = relationship(
        back_populates="approvals",
    )

    __table_args__ = (
        Index(
            "ix_approvals_tenant_status_created",
            "tenant_id",
            "status",
            "created_at",
        ),
        Index(
            "ix_approvals_execution_status",
            "execution_id",
            "status",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the approval."""
        return (
            f"<Approval id={self.id} "
            f"execution_id={self.execution_id} "
            f"status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# Artifact model
# ---------------------------------------------------------------------------


class Artifact(Base):
    """
    ORM model representing metadata for a generated workflow artifact.
    """

    __tablename__ = "artifacts"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=_generate_uuid,
        nullable=False,
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    execution_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "workflow_executions.execution_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    artifact_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
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
    )

    mime_type: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="created",
        nullable=False,
        index=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
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

    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True,
    )

    workflow_execution: Mapped[WorkflowExecution] = relationship(
        back_populates="artifacts",
    )

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
    )

    def __repr__(self) -> str:
        """Return a concise representation of the artifact."""
        return (
            f"<Artifact id={self.id} "
            f"execution_id={self.execution_id} "
            f"artifact_type={self.artifact_type!r} "
            f"status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# Audit event model
# ---------------------------------------------------------------------------


class AuditEvent(Base):
    """
    ORM model representing an immutable tenant audit event.
    """

    __tablename__ = "audit_events"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=_generate_uuid,
        nullable=False,
    )

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
    )

    outcome: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
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

    __table_args__ = (
        Index(
            "ix_audit_events_tenant_type_occurred",
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
            "ix_audit_events_resource",
            "tenant_id",
            "resource_type",
            "resource_id",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the audit event."""
        return (
            f"<AuditEvent id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"event_type={self.event_type!r} "
            f"occurred_at={self.occurred_at}>"
        )


# ---------------------------------------------------------------------------
# Usage event model
# ---------------------------------------------------------------------------


class UsageEvent(Base):
    """
    ORM model representing AI provider token usage and estimated cost.
    """

    __tablename__ = "usage_events"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=_generate_uuid,
        nullable=False,
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

    execution_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    provider: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    model: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        index=True,
    )

    operation: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    prompt_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    completion_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    total_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    estimated_cost: Mapped[float | None] = mapped_column(
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
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

    __table_args__ = (
        Index(
            "ix_usage_events_tenant_occurred",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_usage_events_tenant_provider_model",
            "tenant_id",
            "provider",
            "model",
        ),
        Index(
            "ix_usage_events_tenant_user",
            "tenant_id",
            "user_id",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the usage event."""
        return (
            f"<UsageEvent id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"provider={self.provider!r} "
            f"model={self.model!r} "
            f"total_tokens={self.total_tokens}>"
        )


__all__ = [
    "Approval",
    "Artifact",
    "AuditEvent",
    "Document",
    "DocumentChunk",
    "Tenant",
    "UsageEvent",
    "User",
    "WorkflowExecution",
]