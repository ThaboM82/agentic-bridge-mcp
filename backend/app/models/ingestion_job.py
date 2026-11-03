"""
Asynchronous ingestion-job database models.

These models track tenant-scoped document ingestion batches, per-item ingestion
results, retry attempts, worker progress, cancellation, idempotency, and job
lifecycle state.
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
    generate_uuid,
    utc_now,
)


class IngestionJob(Base):
    """
    ORM model representing an asynchronous document-ingestion job.

    An ingestion job groups one or more source documents into a trackable
    batch, records processing progress, supports retries and cancellation, and
    retains the final result or error details.
    """

    __tablename__ = "ingestion_jobs"

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

    job_type: Mapped[str] = mapped_column(
        String(100),
        default="document_ingestion",
        nullable=False,
        index=True,
    )

    title: Mapped[str | None] = mapped_column(
        String(300),
        nullable=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Queue and lifecycle
    # -------------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(32),
        default="queued",
        nullable=False,
        index=True,
    )

    priority: Mapped[int] = mapped_column(
        Integer,
        default=5,
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

    idempotency_key: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    correlation_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    is_cancel_requested: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    cancelled_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
    )

    cancellation_reason: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Batch progress
    # -------------------------------------------------------------------------

    total_documents: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    processed_documents: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    failed_documents: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    skipped_documents: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    current_batch: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    total_batches: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    progress_percent: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Processing telemetry
    # -------------------------------------------------------------------------

    total_chunks: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    total_embeddings: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    processing_duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    worker_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    worker_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Errors and results
    # -------------------------------------------------------------------------

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    result: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    job_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Timestamps
    # -------------------------------------------------------------------------

    scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

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

    next_retry_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    items: Mapped[list["IngestionJobItem"]] = relationship(
        back_populates="ingestion_job",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    audit_events: Mapped[list["IngestionJobAuditEvent"]] = relationship(
        back_populates="ingestion_job",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_ingestion_jobs_tenant_idempotency_key",
        ),
        Index(
            "ix_ingestion_jobs_tenant_status_created",
            "tenant_id",
            "status",
            "created_at",
        ),
        Index(
            "ix_ingestion_jobs_status_retry",
            "status",
            "next_retry_at",
        ),
        Index(
            "ix_ingestion_jobs_tenant_job_type",
            "tenant_id",
            "job_type",
        ),
        Index(
            "ix_ingestion_jobs_tenant_priority_status",
            "tenant_id",
            "priority",
            "status",
        ),
        Index(
            "ix_ingestion_jobs_worker",
            "worker_id",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """Return whether the job has reached a terminal state."""
        return self.status in {
            "completed",
            "failed",
            "cancelled",
        }

    @property
    def is_active(self) -> bool:
        """Return whether the job is queued or currently processing."""
        return self.status in {
            "queued",
            "processing",
        }

    @property
    def pending_documents(self) -> int:
        """Return the number of documents not yet resolved."""
        return max(
            self.total_documents
            - self.processed_documents
            - self.failed_documents
            - self.skipped_documents,
            0,
        )

    @property
    def can_retry(self) -> bool:
        """Return whether another retry attempt is permitted."""
        return (
            self.status == "failed"
            and self.attempt_count
            < self.max_attempts
        )

    @property
    def success_rate(self) -> float:
        """Return the successful-document percentage."""
        if self.total_documents <= 0:
            return 0.0

        return round(
            (
                self.processed_documents
                / self.total_documents
                * 100
            ),
            2,
        )

    def calculate_progress(self) -> int:
        """Calculate and return job progress as a whole percentage."""
        if self.total_documents <= 0:
            return 0

        resolved_count = (
            self.processed_documents
            + self.failed_documents
            + self.skipped_documents
        )

        return min(
            int(
                resolved_count
                / self.total_documents
                * 100
            ),
            100,
        )

    def __repr__(self) -> str:
        """Return a concise representation of the ingestion job."""
        return (
            f"<IngestionJob id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"status={self.status!r} "
            f"progress={self.progress_percent}%>"
        )


class IngestionJobItem(Base):
    """
    ORM model representing one document within an ingestion job.

    Each item tracks the individual ingestion outcome, including chunk count,
    embedding status, processing duration, retries, and failure details.
    """

    __tablename__ = "ingestion_job_items"

    # -------------------------------------------------------------------------
    # Identity and ownership
    # -------------------------------------------------------------------------

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid,
        nullable=False,
    )

    ingestion_job_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "ingestion_jobs.id",
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

    document_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    source_document_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    title: Mapped[str | None] = mapped_column(
        String(300),
        nullable=True,
    )

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

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(32),
        default="queued",
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

    # -------------------------------------------------------------------------
    # Processing telemetry
    # -------------------------------------------------------------------------

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

    embedding_model: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    file_size_bytes: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    processing_duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    worker_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Errors and metadata
    # -------------------------------------------------------------------------

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    item_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Timestamps
    # -------------------------------------------------------------------------

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

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    ingestion_job: Mapped[IngestionJob] = relationship(
        back_populates="items",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "ingestion_job_id",
            "source_document_id",
            name="uq_ingestion_job_items_job_source_document",
        ),
        Index(
            "ix_ingestion_job_items_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_ingestion_job_items_job_status",
            "ingestion_job_id",
            "status",
        ),
        Index(
            "ix_ingestion_job_items_tenant_document",
            "tenant_id",
            "document_id",
        ),
        Index(
            "ix_ingestion_job_items_retry",
            "status",
            "next_retry_at",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_successful(self) -> bool:
        """Return whether the ingestion item completed successfully."""
        return self.status == "completed"

    @property
    def is_failed(self) -> bool:
        """Return whether the ingestion item failed."""
        return self.status == "failed"

    @property
    def is_terminal(self) -> bool:
        """Return whether the ingestion item has reached a terminal state."""
        return self.status in {
            "completed",
            "failed",
            "skipped",
            "cancelled",
        }

    @property
    def can_retry(self) -> bool:
        """Return whether the ingestion item can be retried."""
        return (
            self.status == "failed"
            and self.attempt_count
            < self.max_attempts
        )

    def __repr__(self) -> str:
        """Return a concise representation of the ingestion item."""
        return (
            f"<IngestionJobItem id={self.id} "
            f"job_id={self.ingestion_job_id} "
            f"status={self.status!r} "
            f"chunks={self.chunk_count}>"
        )


class IngestionJobAuditEvent(Base):
    """
    ORM model representing an immutable audit event for an ingestion job.
    """

    __tablename__ = "ingestion_job_audit_events"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid,
        nullable=False,
    )

    ingestion_job_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "ingestion_jobs.id",
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

    ingestion_job: Mapped[IngestionJob] = relationship(
        back_populates="audit_events",
    )

    __table_args__ = (
        Index(
            "ix_ingestion_job_audit_events_tenant_time",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_ingestion_job_audit_events_job_time",
            "ingestion_job_id",
            "occurred_at",
        ),
        Index(
            "ix_ingestion_job_audit_events_actor",
            "tenant_id",
            "actor_id",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the audit event."""
        return (
            f"<IngestionJobAuditEvent id={self.id} "
            f"job_id={self.ingestion_job_id} "
            f"event_type={self.event_type!r}>"
        )


__all__ = [
    "IngestionJob",
    "IngestionJobAuditEvent",
    "IngestionJobItem",
]