"""
Usage-metering, quota, rate-card, allocation, and audit database models.

These models support tenant-scoped AI usage metering, cost estimation, quota
enforcement, subscription limits, usage rollups, provider rate cards, and
immutable usage audit events.
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


class UsageEvent(UUIDPrimaryKeyMixin, Base):
    """
    AI provider usage event.

    Each event records one billable or meterable AI operation, including token
    consumption, request context, estimated cost, and provider diagnostics.
    """

    __tablename__ = "usage_events"

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

    execution_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    workflow_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    agent_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    integration_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Provider context
    # -------------------------------------------------------------------------

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
        index=True,
    )

    request_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    provider_request_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # -------------------------------------------------------------------------
    # Token usage
    # -------------------------------------------------------------------------

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

    reasoning_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    cached_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    total_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Cost and performance
    # -------------------------------------------------------------------------

    currency: Mapped[str] = mapped_column(
        String(3),
        default="USD",
        nullable=False,
    )

    input_cost: Mapped[float | None] = mapped_column(
        Numeric(14, 6),
        nullable=True,
    )

    output_cost: Mapped[float | None] = mapped_column(
        Numeric(14, 6),
        nullable=True,
    )

    estimated_cost: Mapped[float | None] = mapped_column(
        Numeric(14, 6),
        nullable=True,
    )

    actual_cost: Mapped[float | None] = mapped_column(
        Numeric(14, 6),
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

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

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
        Index(
            "ix_usage_events_execution",
            "execution_id",
        ),
        Index(
            "ix_usage_events_workflow",
            "workflow_id",
        ),
        Index(
            "ix_usage_events_status",
            "status",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_successful(self) -> bool:
        """Return whether the usage event completed successfully."""
        return self.status == "completed"

    def calculate_estimated_cost(
        self,
        input_cost_per_million_tokens: float,
        output_cost_per_million_tokens: float,
    ) -> float | None:
        """Calculate and store estimated cost from a rate card."""
        if input_cost_per_million_tokens is None:
            return None

        if output_cost_per_million_tokens is None:
            return None

        input_cost = (
            self.prompt_tokens
            / 1_000_000
            * input_cost_per_million_tokens
        )

        output_cost = (
            self.completion_tokens
            / 1_000_000
            * output_cost_per_million_tokens
        )

        self.input_cost = input_cost
        self.output_cost = output_cost
        self.estimated_cost = input_cost + output_cost

        return self.estimated_cost

    def __repr__(self) -> str:
        """Return a concise representation of the usage event."""
        return (
            f"<UsageEvent id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"provider={self.provider!r} "
            f"model={self.model!r} "
            f"total_tokens={self.total_tokens}>"
        )


class UsageQuota(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Tenant usage quota for a billing period.

    A quota can apply to tokens, API calls, documents, storage, or a custom
    usage type. A null `usage_type` can represent a tenant-wide token quota.
    """

    __tablename__ = "usage_quotas"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    usage_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    token_quota: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    tokens_used: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    request_quota: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    requests_used: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    cost_quota: Mapped[float | None] = mapped_column(
        Numeric(14, 6),
        nullable=True,
    )

    cost_used: Mapped[float] = mapped_column(
        Numeric(14, 6),
        default=0,
        nullable=False,
    )

    currency: Mapped[str] = mapped_column(
        String(3),
        default="USD",
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

    is_hard_limit: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    exceeded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    reset_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    quota_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Relationships
    # -------------------------------------------------------------------------

    allocations = relationship(
        "UsageAllocation",
        back_populates="quota",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    audit_events = relationship(
        "UsageAuditEvent",
        back_populates="quota",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "usage_type",
            "period_start",
            "period_end",
            name="uq_usage_quotas_tenant_type_period",
        ),
        Index(
            "ix_usage_quotas_tenant_period",
            "tenant_id",
            "period_start",
            "period_end",
        ),
        Index(
            "ix_usage_quotas_active_period",
            "is_active",
            "period_start",
            "period_end",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def tokens_remaining(self) -> int:
        """Return the number of tokens remaining in the quota."""
        return max(
            self.token_quota - self.tokens_used,
            0,
        )

    @property
    def token_usage_percentage(self) -> float:
        """Return token usage as a percentage of the quota."""
        if self.token_quota <= 0:
            return 0.0

        return (
            self.tokens_used
            / self.token_quota
            * 100
        )

    @property
    def is_token_quota_exceeded(self) -> bool:
        """Return whether the token quota has been exceeded."""
        return (
            self.token_quota > 0
            and self.tokens_used >= self.token_quota
        )

    @property
    def is_request_quota_exceeded(self) -> bool:
        """Return whether the request quota has been exceeded."""
        return (
            self.request_quota is not None
            and self.request_quota > 0
            and self.requests_used >= self.request_quota
        )

    @property
    def is_cost_quota_exceeded(self) -> bool:
        """Return whether the cost quota has been exceeded."""
        return (
            self.cost_quota is not None
            and self.cost_quota > 0
            and self.cost_used >= self.cost_quota
        )

    @property
    def is_exceeded(self) -> bool:
        """Return whether any configured quota has been exceeded."""
        return (
            self.is_token_quota_exceeded
            or self.is_request_quota_exceeded
            or self.is_cost_quota_exceeded
        )

    def consume_tokens(
        self,
        tokens: int,
    ) -> None:
        """Add consumed tokens to the quota."""
        self.tokens_used += max(tokens, 0)

        if self.is_token_quota_exceeded and self.exceeded_at is None:
            self.exceeded_at = utc_now()

    def consume_request(self) -> None:
        """Add one consumed request to the quota."""
        self.requests_used += 1

        if self.is_request_quota_exceeded and self.exceeded_at is None:
            self.exceeded_at = utc_now()

    def consume_cost(
        self,
        cost: float,
    ) -> None:
        """Add consumed cost to the quota."""
        self.cost_used += max(cost, 0)

        if self.is_cost_quota_exceeded and self.exceeded_at is None:
            self.exceeded_at = utc_now()

    def __repr__(self) -> str:
        """Return a concise representation of the usage quota."""
        return (
            f"<UsageQuota id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"usage_type={self.usage_type!r} "
            f"tokens_used={self.tokens_used}>"
        )


class UsageRateCard(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Provider model rate card used for cost estimation.

    Rate cards are versioned by effective date so historical usage can be
    recalculated using the pricing that was active when the request occurred.
    """

    __tablename__ = "usage_rate_cards"

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

    currency: Mapped[str] = mapped_column(
        String(3),
        default="USD",
        nullable=False,
    )

    input_cost_per_million_tokens: Mapped[float] = mapped_column(
        Numeric(14, 6),
        default=0,
        nullable=False,
    )

    output_cost_per_million_tokens: Mapped[float] = mapped_column(
        Numeric(14, 6),
        default=0,
        nullable=False,
    )

    cached_input_cost_per_million_tokens: Mapped[float | None] = mapped_column(
        Numeric(14, 6),
        nullable=True,
    )

    reasoning_cost_per_million_tokens: Mapped[float | None] = mapped_column(
        Numeric(14, 6),
        nullable=True,
    )

    embedding_cost_per_million_tokens: Mapped[float | None] = mapped_column(
        Numeric(14, 6),
        nullable=True,
    )

    request_cost: Mapped[float | None] = mapped_column(
        Numeric(14, 6),
        nullable=True,
    )

    # -------------------------------------------------------------------------
    # Effective period
    # -------------------------------------------------------------------------

    effective_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    effective_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    rate_card_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # Indexes
    # -------------------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "provider",
            "model",
            "effective_from",
            name="uq_usage_rate_cards_provider_model_effective_from",
        ),
        Index(
            "ix_usage_rate_cards_provider_model_active",
            "provider",
            "model",
            "is_active",
        ),
        Index(
            "ix_usage_rate_cards_effective_period",
            "effective_from",
            "effective_to",
        ),
    )

    # -------------------------------------------------------------------------
    # Domain helpers
    # -------------------------------------------------------------------------

    @property
    def is_current(self) -> bool:
        """Return whether the rate card is currently effective."""
        now = utc_now()

        if not self.is_active:
            return False

        if now < self.effective_from:
            return False

        if self.effective_to is not None and now > self.effective_to:
            return False

        return True

    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Return the estimated cost for the supplied token counts."""
        input_cost = (
            prompt_tokens
            / 1_000_000
            * float(
                self.input_cost_per_million_tokens
            )
        )

        output_cost = (
            completion_tokens
            / 1_000_000
            * float(
                self.output_cost_per_million_tokens
            )
        )

        return input_cost + output_cost

    def __repr__(self) -> str:
        """Return a concise representation of the rate card."""
        return (
            f"<UsageRateCard id={self.id} "
            f"provider={self.provider!r} "
            f"model={self.model!r} "
            f"effective_from={self.effective_from}>"
        )


class UsageAllocation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Usage allocation for a tenant, user, team, workflow, or agent.

    Allocations allow quota consumption to be attributed to a specific scope
    within a tenant.
    """

    __tablename__ = "usage_allocations"

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    quota_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "usage_quotas.id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    team_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    workflow_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    agent_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    usage_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    token_limit: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    tokens_used: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    request_limit: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    requests_used: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    cost_limit: Mapped[float | None] = mapped_column(
        Numeric(14, 6),
        nullable=True,
    )

    cost_used: Mapped[float] = mapped_column(
        Numeric(14, 6),
        default=0,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    allocation_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    quota = relationship(
        "UsageQuota",
        back_populates="allocations",
    )

    __table_args__ = (
        Index(
            "ix_usage_allocations_tenant_type",
            "tenant_id",
            "usage_type",
        ),
        Index(
            "ix_usage_allocations_tenant_user",
            "tenant_id",
            "user_id",
        ),
        Index(
            "ix_usage_allocations_tenant_workflow",
            "tenant_id",
            "workflow_id",
        ),
        Index(
            "ix_usage_allocations_tenant_agent",
            "tenant_id",
            "agent_id",
        ),
    )

    @property
    def is_token_limit_exceeded(self) -> bool:
        """Return whether the allocation token limit has been exceeded."""
        return (
            self.token_limit is not None
            and self.token_limit > 0
            and self.tokens_used >= self.token_limit
        )

    def consume_tokens(
        self,
        tokens: int,
    ) -> None:
        """Add consumed tokens to the allocation."""
        self.tokens_used += max(tokens, 0)

    def __repr__(self) -> str:
        """Return a concise representation of the usage allocation."""
        return (
            f"<UsageAllocation id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"usage_type={self.usage_type!r} "
            f"tokens_used={self.tokens_used}>"
        )


class UsageRollup(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Aggregated usage rollup for reporting and billing.

    Rollups are typically generated by a scheduled job and should not be the
    sole source of truth for individual usage events.
    """

    __tablename__ = "usage_rollups"

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

    provider: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    model: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    usage_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    event_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
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

    estimated_cost: Mapped[float] = mapped_column(
        Numeric(14, 6),
        default=0,
        nullable=False,
    )

    actual_cost: Mapped[float] = mapped_column(
        Numeric(14, 6),
        default=0,
        nullable=False,
    )

    currency: Mapped[str] = mapped_column(
        String(3),
        default="USD",
        nullable=False,
    )

    rollup_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "user_id",
            "provider",
            "model",
            "usage_type",
            "period_start",
            "period_end",
            name="uq_usage_rollups_tenant_scope_period",
        ),
        Index(
            "ix_usage_rollups_tenant_period",
            "tenant_id",
            "period_start",
            "period_end",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the usage rollup."""
        return (
            f"<UsageRollup id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"provider={self.provider!r} "
            f"model={self.model!r} "
            f"total_tokens={self.total_tokens}>"
        )


class UsageAuditEvent(UUIDPrimaryKeyMixin, Base):
    """
    Immutable audit event for usage, quota, and rate-card activity.
    """

    __tablename__ = "usage_audit_events"

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

    usage_event_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    quota_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(
            "usage_quotas.id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    rate_card_id: Mapped[uuid.UUID | None] = mapped_column(
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

    quota = relationship(
        "UsageQuota",
        back_populates="audit_events",
    )

    __table_args__ = (
        Index(
            "ix_usage_audit_events_tenant_time",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_usage_audit_events_quota_time",
            "quota_id",
            "occurred_at",
        ),
        Index(
            "ix_usage_audit_events_actor",
            "tenant_id",
            "actor_id",
        ),
    )

    def __repr__(self) -> str:
        """Return a concise representation of the usage audit event."""
        return (
            f"<UsageAuditEvent id={self.id} "
            f"tenant_id={self.tenant_id} "
            f"event_type={self.event_type!r}>"
        )


__all__ = [
    "UsageAllocation",
    "UsageAuditEvent",
    "UsageEvent",
    "UsageQuota",
    "UsageRateCard",
    "UsageRollup",
]