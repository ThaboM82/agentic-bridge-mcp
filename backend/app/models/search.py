"""
Saved-search, feedback, retrieval-experiment, and search-log database models.

These models support tenant-scoped semantic and hybrid retrieval. They store
saved searches, query logs, relevance feedback, retrieval experiments, experiment
variants, and immutable search audit events.
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

from app.core.database import Base
from app.models.base import (
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    generate_uuid,
    utc_now,
)


class SavedSearch(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Tenant-scoped saved retrieval search.

    A saved search stores a reusable query, retrieval filters, and optional
    default retrieval settings. It may be private to a user or shared across a
    tenant.
    """

    __tablename__ = "saved_searches"

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

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        index=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Search definition
    # -------------------------------------------------------------------------

    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    filters: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    retrieval_mode: Mapped[str] = mapped_column(
        String(50),
        default="semantic",
        nullable=False,
        index=True,
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    top_k: Mapped[int] = mapped_column(
        Integer,
        default=10,
        nullable=False,
    )

    score_threshold: Mapped[float | None] = mapped_column(
        Numeric(8, 6),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Sharing and lifecycle
    # -------------------------------------------------------------------------

    is_shared: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    is_pinned: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    last_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    run_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    search_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    query_logs = relationship(
        "SearchQueryLog",
        back_populates="saved_search",
        lazy="selectin",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "user_id",
            "name",
            name="uq_saved_searches_tenant_user_name",
        ),
        Index(
            "ix_saved_searches_tenant_shared",
            "tenant_id",
            "is_shared",
        ),
        Index(
            "ix_saved_searches_tenant_user_active",
            "tenant_id",
            "user_id",
            "is_active",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_personal(self) -> bool:
        """Return whether this search belongs to a specific user."""
        return self.user_id is not None

    def record_run(self) -> None:
        """Record that the saved search was executed."""
        self.run_count += 1
        self.last_run_at = utc_now()

    def __repr__(self) -> str:
        """Return a concise representation of the saved search."""
        return (
            f"<SavedSearch id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"user_id={self.user_id} "
            f"name={self.name!r}>"
        )


class SearchQueryLog(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Search execution log.

    Each log records one retrieval request, including the search mode,
    configuration, latency, result count, and selected result identifiers.
    """

    __tablename__ = "search_query_logs"

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

    saved_search_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "saved_searches.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    experiment_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "retrieval_experiments.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    retrieval_mode: Mapped[str] = mapped_column(
        String(50),
        default="semantic",
        nullable=False,
        index=True,
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    filters: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    result_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    result_ids: Mapped[list[uuid.UUID]] = mapped_column(
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

    latency_ms: Mapped[int | None] = mapped_column(
        Integer,
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

    log_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    saved_search = relationship(
        "SavedSearch",
        back_populates="query_logs",
    )

    experiment = relationship(
        "RetrievalExperiment",
        back_populates="query_logs",
    )

    feedback = relationship(
        "SearchFeedback",
        back_populates="query_log",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index(
            "ix_search_query_logs_tenant_created",
            "tenant_id",
            "created_at",
        ),
        Index(
            "ix_search_query_logs_tenant_user",
            "tenant_id",
            "user_id",
        ),
        Index(
            "ix_search_query_logs_experiment",
            "experiment_id",
        ),
        Index(
            "ix_search_query_logs_mode_model",
            "retrieval_mode",
            "embedding_model",
        ),
    )

    @property
    def is_successful(self) -> bool:
        """Return whether the search completed successfully."""
        return self.status == "completed"

    def __repr__(self) -> str:
        """Return a concise representation of the search query log."""
        return (
            f"<SearchQueryLog id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"retrieval_mode={self.retrieval_mode!r} "
            f"result_count={self.result_count}>"
        )


class SearchFeedback(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    User relevance feedback for a retrieval result.

    Feedback can be attached to a query log, a specific embedding result, or
    both. It is used to improve retrieval quality and evaluate experiments.
    """

    __tablename__ = "search_feedback"

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

    query_log_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "search_query_logs.id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    embedding_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    document_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
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

    query_log = relationship(
        "SearchQueryLog",
        back_populates="feedback",
    )

    __table_args__ = (
        Index(
            "ix_search_feedback_tenant_created",
            "tenant_id",
            "created_at",
        ),
        Index(
            "ix_search_feedback_tenant_user",
            "tenant_id",
            "user_id",
        ),
        Index(
            "ix_search_feedback_query_embedding",
            "query_log_id",
            "embedding_id",
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
        """Return a concise representation of search feedback."""
        return (
            f"<SearchFeedback id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"query_log_id={self.query_log_id} "
            f"rating={self.rating}>"
        )


class RetrievalExperiment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Retrieval configuration experiment.

    An experiment defines a retrieval strategy, such as semantic, keyword, or
    hybrid retrieval, and records evaluation metrics for comparison.
    """

    __tablename__ = "retrieval_experiments"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        index=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
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

    configuration: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Evaluation metrics
    # -------------------------------------------------------------------------

    average_precision: Mapped[float | None] = mapped_column(
        Numeric(8, 6),
        nullable=True,
    )

    precision_at_k: Mapped[float | None] = mapped_column(
        Numeric(8, 6),
        nullable=True,
    )

    recall_at_k: Mapped[float | None] = mapped_column(
        Numeric(8, 6),
        nullable=True,
    )

    mean_reciprocal_rank: Mapped[float | None] = mapped_column(
        Numeric(8, 6),
        nullable=True,
    )

    average_latency_ms: Mapped[float | None] = mapped_column(
        Numeric(10, 2),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(50),
        default="draft",
        nullable=False,
        index=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    is_control: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    traffic_percentage: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    experiment_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    variants = relationship(
        "RetrievalExperimentVariant",
        back_populates="experiment",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    query_logs = relationship(
        "SearchQueryLog",
        back_populates="experiment",
        lazy="selectin",
    )

    audit_events = relationship(
        "SearchAuditEvent",
        back_populates="experiment",
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
            name="uq_retrieval_experiments_tenant_name",
        ),
        Index(
            "ix_retrieval_experiments_tenant_active",
            "tenant_id",
            "is_active",
        ),
        Index(
            "ix_retrieval_experiments_tenant_status",
            "tenant_id",
            "status",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Return whether the experiment is currently running."""
        return self.status == "running"

    def __repr__(self) -> str:
        """Return a concise representation of the retrieval experiment."""
        return (
            f"<RetrievalExperiment id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"name={self.name!r} "
            f"status={self.status!r}>"
        )


class RetrievalExperimentVariant(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Variant configuration for a retrieval experiment.

    Variants allow A/B testing of retrieval parameters such as top-k, score
    thresholds, reranking, and hybrid-search weighting.
    """

    __tablename__ = "retrieval_experiment_variants"

    experiment_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "retrieval_experiments.id",
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

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    is_control: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    traffic_percentage: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    configuration: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    average_precision: Mapped[float | None] = mapped_column(
        Numeric(8, 6),
        nullable=True,
    )

    average_latency_ms: Mapped[float | None] = mapped_column(
        Numeric(10, 2),
        nullable=True,
    )

    variant_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    experiment = relationship(
        "RetrievalExperiment",
        back_populates="variants",
    )

    __table_args__ = (
        UniqueConstraint(
            "experiment_id",
            "name",
            name="uq_retrieval_experiment_variants_experiment_name",
        ),
        Index(
            "ix_retrieval_experiment_variants_tenant",
            "tenant_id",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the experiment variant."""
        return (
            f"<RetrievalExperimentVariant id={self.id} "
            f"experiment_id={self.experiment_id} "
            f"name={self.name!r}>"
        )


class SearchAuditEvent(UUIDPrimaryKeyMixin, Base):
    """
    Immutable audit event for search and retrieval activity.
    """

    __tablename__ = "search_audit_events"

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

    query_log_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    experiment_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "retrieval_experiments.id",
            ondelete="CASCADE",
        ),
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

    experiment = relationship(
        "RetrievalExperiment",
        back_populates="audit_events",
    )

    __table_args__ = (
        Index(
            "ix_search_audit_events_tenant_time",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_search_audit_events_experiment_time",
            "experiment_id",
            "occurred_at",
        ),
        Index(
            "ix_search_audit_events_user",
            "tenant_id",
            "user_id",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the search audit event."""
        return (
            f"<SearchAuditEvent id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"event_type={self.event_type!r}>"
        )


__all__ = [
    "RetrievalExperiment",
    "RetrievalExperimentVariant",
    "SavedSearch",
    "SearchAuditEvent",
    "SearchFeedback",
    "SearchQueryLog",
]