"""
Vector embedding, retrieval, feedback, batch, and audit database models.

These models store tenant-scoped document chunks and high-dimensional vector
embeddings for retrieval-augmented generation (RAG), semantic search, hybrid
retrieval, document intelligence workflows, embedding lifecycle management,
retrieval observability, and relevance feedback.
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


EMBEDDING_DIMENSIONS = 1536


def _generate_uuid() -> uuid.UUID:
    """Generate a database-side UUID default."""
    return uuid.uuid4()


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(
        timezone.utc,
    )


class DocumentEmbedding(Base):
    """
    Enterprise ORM model for storing unstructured document chunks and their
    high-dimensional vector embeddings for RAG retrieval.

    Each embedding belongs to a tenant and optionally to a parent document.
    The vector column uses pgvector and is indexed for cosine-similarity search.
    """

    __tablename__ = "document_embeddings"

    # -------------------------------------------------------------------------
    # Identity and ownership
    # -------------------------------------------------------------------------

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=_generate_uuid,
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

    character_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    language: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Vector embedding
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

    embedding_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    embedding_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
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

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
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

    is_archived: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
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

    document = relationship(
        "Document",
        foreign_keys=[document_id],
        lazy="selectin",
    )

    feedback = relationship(
        "EmbeddingFeedback",
        back_populates="embedding",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    audit_events = relationship(
        "EmbeddingAuditEvent",
        back_populates="embedding",
        cascade="all, delete-orphan",
        lazy="selectin",
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
            "ix_document_embeddings_tenant_type",
            "tenant_id",
            "document_type",
        ),
        Index(
            "ix_document_embeddings_embedding_model",
            "embedding_model",
        ),
        Index(
            "ix_document_embeddings_content_hash",
            "content_hash",
        ),
        Index(
            "ix_document_embeddings_resource",
            "resource_type",
            "resource_id",
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
            and not self.is_archived
            and self.embedding_status == "ready"
            and self.embedding is not None
        )

    def archive(self) -> None:
        """Archive the embedding and remove it from active retrieval."""
        self.is_active = False
        self.is_archived = True
        self.archived_at = _utc_now()

    def reactivate(self) -> None:
        """Reactivate the embedding for retrieval."""
        self.is_active = True
        self.is_archived = False
        self.archived_at = None

    def to_retrieval_metadata(self) -> dict[str, Any]:
        """Return a compact metadata dictionary for retrieval responses."""
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


class EmbeddingQueryLog(Base):
    """
    ORM model representing a semantic-search query log.

    This model supports retrieval observability, query auditing, latency
    analysis, and relevance tuning without storing raw embedding vectors.
    """

    __tablename__ = "embedding_query_logs"

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

    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    embedding_model: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        index=True,
    )

    retrieval_mode: Mapped[str] = mapped_column(
        String(50),
        default="semantic",
        nullable=False,
        index=True,
    )

    top_k: Mapped[int] = mapped_column(
        Integer,
        default=5,
        nullable=False,
    )

    score_threshold: Mapped[float | None] = mapped_column(
        Numeric(8, 6),
        nullable=True,
    )

    result_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    elapsed_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    result_embedding_ids: Mapped[list[uuid.UUID]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )

    top_score: Mapped[float | None] = mapped_column(
        Numeric(10, 6),
        nullable=True,
    )

    average_score: Mapped[float | None] = mapped_column(
        Numeric(10, 6),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="completed",
        nullable=False,
        index=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    query_metadata: Mapped[dict[str, Any]] = mapped_column(
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

    __table_args__ = (
        Index(
            "ix_embedding_query_logs_tenant_created",
            "tenant_id",
            "created_at",
        ),
        Index(
            "ix_embedding_query_logs_tenant_user",
            "tenant_id",
            "user_id",
        ),
        Index(
            "ix_embedding_query_logs_retrieval_mode",
            "retrieval_mode",
        ),
        Index(
            "ix_embedding_query_logs_model_mode",
            "embedding_model",
            "retrieval_mode",
        ),
    )

    @property
    def is_successful(self) -> bool:
        """Return whether the retrieval query completed successfully."""
        return self.status == "completed"

    def __repr__(self) -> str:
        """Return a concise representation of the query log."""
        return (
            f"<EmbeddingQueryLog id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"retrieval_mode={self.retrieval_mode!r} "
            f"result_count={self.result_count}>"
        )


class EmbeddingFeedback(Base):
    """
    User relevance feedback for an embedding retrieval result.

    Feedback is used to evaluate retrieval quality, improve ranking strategies,
    and measure embedding-model effectiveness.
    """

    __tablename__ = "embedding_feedback"

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

    embedding_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "document_embeddings.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    query_log_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "embedding_query_logs.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    rating: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    relevance_label: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
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

    embedding = relationship(
        "DocumentEmbedding",
        back_populates="feedback",
    )

    __table_args__ = (
        Index(
            "ix_embedding_feedback_tenant_created",
            "tenant_id",
            "created_at",
        ),
        Index(
            "ix_embedding_feedback_embedding",
            "embedding_id",
        ),
        Index(
            "ix_embedding_feedback_query",
            "query_log_id",
        ),
    )

    @property
    def is_positive(self) -> bool:
        """Return whether the feedback rating is positive."""
        return self.rating >= 4

    @property
    def is_negative(self) -> bool:
        """Return whether the feedback rating is negative."""
        return self.rating <= 2

    def __repr__(self) -> str:
        """Return a concise representation of embedding feedback."""
        return (
            f"<EmbeddingFeedback id={self.id} "
            f"embedding_id={self.embedding_id} "
            f"rating={self.rating}>"
        )


class EmbeddingBatchJob(Base):
    """
    ORM model representing an embedding generation or reindexing batch job.

    Batch jobs track bulk embedding generation for documents, chunks, or
    external resources, including progress and failure diagnostics.
    """

    __tablename__ = "embedding_batch_jobs"

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

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    job_type: Mapped[str] = mapped_column(
        String(50),
        default="embedding",
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="queued",
        nullable=False,
        index=True,
    )

    embedding_model: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        index=True,
    )

    total_items: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    processed_items: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    failed_items: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    batch_size: Mapped[int] = mapped_column(
        Integer,
        default=100,
        nullable=False,
    )

    cursor: Mapped[str | None] = mapped_column(
        String(1_000),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    job_metadata: Mapped[dict[str, Any]] = mapped_column(
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

    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True,
    )

    __table_args__ = (
        Index(
            "ix_embedding_batch_jobs_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_embedding_batch_jobs_status_created",
            "status",
            "created_at",
        ),
    )

    @property
    def is_terminal(self) -> bool:
        """Return whether the batch job has reached a terminal state."""
        return self.status in {
            "completed",
            "failed",
            "cancelled",
        }

    @property
    def progress_percentage(self) -> float:
        """Return batch progress as a percentage."""
        if self.total_items <= 0:
            return 0.0

        return (
            self.processed_items
            / self.total_items
            * 100
        )

    def __repr__(self) -> str:
        """Return a concise representation of the embedding batch job."""
        return (
            f"<EmbeddingBatchJob id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"status={self.status!r} "
            f"progress={self.progress_percentage:.2f}%>"
        )


class EmbeddingAuditEvent(Base):
    """
    Immutable audit event for embedding lifecycle and retrieval activity.
    """

    __tablename__ = "embedding_audit_events"

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

    embedding_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "document_embeddings.id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    query_log_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    batch_job_id: Mapped[uuid.UUID | None] = mapped_column(
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

    embedding = relationship(
        "DocumentEmbedding",
        back_populates="audit_events",
    )

    __table_args__ = (
        Index(
            "ix_embedding_audit_events_tenant_time",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_embedding_audit_events_embedding_time",
            "embedding_id",
            "occurred_at",
        ),
        Index(
            "ix_embedding_audit_events_actor",
            "tenant_id",
            "actor_id",
        ),
        Index(
            "ix_embedding_audit_events_type_time",
            "event_type",
            "occurred_at",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the embedding audit event."""
        return (
            f"<EmbeddingAuditEvent id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"event_type={self.event_type!r}>"
        )


__all__ = [
    "EMBEDDING_DIMENSIONS",
    "DocumentEmbedding",
    "EmbeddingAuditEvent",
    "EmbeddingBatchJob",
    "EmbeddingFeedback",
    "EmbeddingQueryLog",
]