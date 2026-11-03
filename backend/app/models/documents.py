"""
Document and embedding database models.

These models store tenant-scoped documents, document chunks, vector embeddings,
document versions, retrieval metadata, tags, feedback, and audit events for the
AgenticBridge document intelligence and audit engine.
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
from pgvector.sqlalchemy import Vector

from app.core.database import Base
from app.models.base import (
    generate_uuid,
    utc_now,
)


EMBEDDING_DIMENSIONS = 1536


class Document(Base):
    """
    ORM model representing a tenant-scoped source document.

    A document is the logical parent of one or more embedded chunks used for
    semantic retrieval. It also supports versioning, tagging, retention,
    workflow linkage, and audit history.
    """

    __tablename__ = "documents"

    # -------------------------------------------------------------------------
    # Identity and ownership
    # -------------------------------------------------------------------------

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid,
        nullable=False,
    )

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

    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    title: Mapped[str] = mapped_column(
        String(300),
        nullable=False,
        index=True,
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
    # Source and storage metadata
    # -------------------------------------------------------------------------

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

    source_url: Mapped[str | None] = mapped_column(
        String(2_000),
        nullable=True,
    )

    storage_provider: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    storage_bucket: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    storage_key: Mapped[str | None] = mapped_column(
        String(1_000),
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
        index=True,
    )

    checksum_algorithm: Mapped[str | None] = mapped_column(
        String(50),
        default="sha256",
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

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

    embedding_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    current_version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    language: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
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
    # Workflow linkage
    # -------------------------------------------------------------------------

    execution_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    ingestion_job_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
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

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    embeddings: Mapped[list["DocumentEmbedding"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    tags = relationship(
        "DocumentTag",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    feedback = relationship(
        "DocumentFeedback",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    audit_events = relationship(
        "DocumentAuditEvent",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

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
        Index(
            "ix_documents_tenant_created_by",
            "tenant_id",
            "created_by",
        ),
        Index(
            "ix_documents_tenant_execution",
            "tenant_id",
            "execution_id",
        ),
        Index(
            "ix_documents_tenant_ingestion_job",
            "tenant_id",
            "ingestion_job_id",
        ),
        Index(
            "ix_documents_retention_expiry",
            "retention_expires_at",
        ),
        Index(
            "ix_documents_storage_key",
            "storage_key",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_deleted(self) -> bool:
        """Return whether the document has been deleted."""
        return self.deleted_at is not None

    @property
    def is_expired(self) -> bool:
        """Return whether the document has passed its retention deadline."""
        if self.retention_expires_at is None:
            return False

        return (
            datetime.now(
                self.retention_expires_at.tzinfo
            )
            > self.retention_expires_at
        )

    @property
    def is_searchable(self) -> bool:
        """Return whether the document has searchable embeddings."""
        return (
            not self.is_deleted
            and not self.is_expired
            and self.embedding_count > 0
        )

    def mark_deleted(
        self,
        deleted_by: uuid.UUID | None = None,
    ) -> None:
        """Mark the document as deleted."""
        self.deleted_at = utc_now()
        self.deleted_by = deleted_by
        self.status = "deleted"

    def __repr__(self) -> str:
        """Return a concise representation of the document."""
        return (
            f"<Document id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"title={self.title!r} "
            f"status={self.status!r}>"
        )


class DocumentChunk(Base):
    """
    ORM model representing a processed and embedded document chunk.

    Each chunk contains a portion of a source document and an optional vector
    embedding used for pgvector-based semantic retrieval.
    """

    __tablename__ = "document_chunks"

    # -------------------------------------------------------------------------
    # Identity and ownership
    # -------------------------------------------------------------------------

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

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Content
    # -------------------------------------------------------------------------

    title: Mapped[str] = mapped_column(
        String(300),
        nullable=False,
        index=True,
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

    character_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    language: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
        nullable=True,
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    embedding_dimensions: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    embedding_status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
        index=True,
    )

    embedding_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Metadata and lifecycle
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    document: Mapped[Document] = relationship(
        back_populates="chunks",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

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
            "ix_document_chunks_embedding_model",
            "embedding_model",
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

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_searchable(self) -> bool:
        """Return whether this chunk can be used for semantic retrieval."""
        return (
            self.is_active
            and self.embedding_status == "ready"
            and self.embedding is not None
        )

    def __repr__(self) -> str:
        """Return a concise representation of the document chunk."""
        return (
            f"<DocumentChunk id={self.id} "
            f"document_id={self.document_id} "
            f"chunk_index={self.chunk_index} "
            f"embedding_status={self.embedding_status!r}>"
        )


class DocumentVersion(Base):
    """
    ORM model representing an immutable version snapshot of a document.
    """

    __tablename__ = "document_versions"

    # -------------------------------------------------------------------------
    # Identity and ownership
    # -------------------------------------------------------------------------

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid,
        nullable=False,
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

    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Content
    # -------------------------------------------------------------------------

    title: Mapped[str] = mapped_column(
        String(300),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
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

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    change_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
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

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    document: Mapped[Document] = relationship(
        back_populates="versions",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "version_number",
            name="uq_document_versions_document_version",
        ),
        Index(
            "ix_document_versions_tenant_created",
            "tenant_id",
            "created_at",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the document version."""
        return (
            f"<DocumentVersion id={self.id} "
            f"document_id={self.document_id} "
            f"version={self.version_number}>"
        )


class DocumentEmbedding(Base):
    """
    ORM model representing a standalone vector embedding for RAG retrieval.

    This model supports embeddings that may originate from documents, external
    systems, or workflow outputs, while retaining tenant isolation and an
    optional document relationship.
    """

    __tablename__ = "document_embeddings"

    # -------------------------------------------------------------------------
    # Identity and ownership
    # -------------------------------------------------------------------------

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid,
        nullable=False,
        index=True,
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    document_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    external_document_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Content
    # -------------------------------------------------------------------------

    title: Mapped[str | None] = mapped_column(
        String(300),
        nullable=True,
        index=True,
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

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
        nullable=False,
    )

    embedding_model: Mapped[str] = mapped_column(
        String(150),
        default="text-embedding-3-small",
        nullable=False,
        index=True,
    )

    embedding_dimensions: Mapped[int] = mapped_column(
        Integer,
        default=EMBEDDING_DIMENSIONS,
        nullable=False,
    )

    embedding_status: Mapped[str] = mapped_column(
        String(50),
        default="ready",
        nullable=False,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Source metadata
    # -------------------------------------------------------------------------

    source: Mapped[str | None] = mapped_column(
        String(300),
        nullable=True,
        index=True,
    )

    document_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
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

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    document: Mapped[Document | None] = relationship(
        back_populates="embeddings",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "document_id",
            "chunk_index",
            name="uq_document_embeddings_tenant_document_chunk",
        ),
        Index(
            "ix_document_embeddings_tenant_active",
            "tenant_id",
            "is_active",
        ),
        Index(
            "ix_document_embeddings_tenant_document",
            "tenant_id",
            "document_id",
        ),
        Index(
            "ix_document_embeddings_embedding_model",
            "embedding_model",
        ),
        Index(
            "ix_document_embeddings_vector_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={
                "m": 16,
                "ef_construction": 64,
            },
            postgresql_ops={
                "embedding": "vector_cosine_ops"
            },
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_searchable(self) -> bool:
        """Return whether the embedding can be used in retrieval."""
        return (
            self.is_active
            and self.embedding_status == "ready"
            and self.embedding is not None
        )

    def to_retrieval_metadata(self) -> dict[str, Any]:
        """Return compact metadata for retrieval responses."""
        return {
            "embedding_id": str(self.id),
            "document_id": (
                str(self.document_id)
                if self.document_id
                else None
            ),
            "external_document_id": self.external_document_id,
            "chunk_index": self.chunk_index,
            "title": self.title,
            "source": self.source,
            "document_type": self.document_type,
            "embedding_model": self.embedding_model,
            "metadata": self.metadata_json,
        }

    def __repr__(self) -> str:
        """Return a concise representation of the embedding."""
        return (
            f"<DocumentEmbedding id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"document_id={self.document_id} "
            f"chunk_index={self.chunk_index} "
            f"model={self.embedding_model!r}>"
        )


class DocumentTag(Base):
    """
    ORM model representing a tag applied to a document.
    """

    __tablename__ = "document_tags"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid,
        nullable=False,
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

    tag: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    document: Mapped[Document] = relationship(
        back_populates="tags",
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "tag",
            name="uq_document_tags_document_tag",
        ),
        Index(
            "ix_document_tags_tenant_tag",
            "tenant_id",
            "tag",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the document tag."""
        return (
            f"<DocumentTag id={self.id} "
            f"document_id={self.document_id} "
            f"tag={self.tag!r}>"
        )


class DocumentFeedback(Base):
    """
    ORM model representing reviewer feedback on a document or retrieval result.
    """

    __tablename__ = "document_feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid,
        nullable=False,
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

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    embedding_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    rating: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    relevance_score: Mapped[float | None] = mapped_column(
        Numeric(8, 6),
        nullable=True,
    )

    comment: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    feedback_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    document: Mapped[Document] = relationship(
        back_populates="feedback",
    )

    __table_args__ = (
        Index(
            "ix_document_feedback_tenant_document",
            "tenant_id",
            "document_id",
        ),
        Index(
            "ix_document_feedback_tenant_user",
            "tenant_id",
            "user_id",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of document feedback."""
        return (
            f"<DocumentFeedback id={self.id} "
            f"document_id={self.document_id} "
            f"rating={self.rating}>"
        )


class DocumentAuditEvent(Base):
    """
    ORM model representing an immutable audit event for document activity.
    """

    __tablename__ = "document_audit_events"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid,
        nullable=False,
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

    document: Mapped[Document] = relationship(
        back_populates="audit_events",
    )

    __table_args__ = (
        Index(
            "ix_document_audit_events_tenant_time",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_document_audit_events_document_time",
            "document_id",
            "occurred_at",
        ),
        Index(
            "ix_document_audit_events_actor",
            "tenant_id",
            "actor_id",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the document audit event."""
        return (
            f"<DocumentAuditEvent id={self.id} "
            f"document_id={self.document_id} "
            f"event_type={self.event_type!r}>"
        )


__all__ = [
    "Document",
    "DocumentAuditEvent",
    "DocumentChunk",
    "DocumentEmbedding",
    "DocumentFeedback",
    "DocumentTag",
    "DocumentVersion",
    "EMBEDDING_DIMENSIONS",
]