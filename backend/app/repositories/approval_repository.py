"""
Approval request and decision repository.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import (
    ApprovalDecision,
    ApprovalStatus,
)
from app.core.exceptions import ValidationError
from app.models.approval_model import ApprovalModel
from app.repositories.base_repository import BaseRepository


logger = logging.getLogger(
    "agentic_bridge.repositories.approval"
)


class ApprovalRepository(
    BaseRepository[ApprovalModel]
):
    """Tenant-scoped persistence for human approvals."""

    model = ApprovalModel

    @staticmethod
    def _normalize_status(
        status: str | ApprovalStatus | None,
    ) -> str | None:
        if status is None:
            return None

        if isinstance(
            status, ApprovalStatus
        ):
            return status.value

        if not isinstance(status, str):
            raise ValidationError(
                "status must be a string or ApprovalStatus."
            )

        normalized_status = (
            status.strip().lower()
        )

        try:
            return ApprovalStatus(
                normalized_status
            ).value
        except ValueError as exc:
            raise ValidationError(
                f"Unsupported approval status: {status!r}."
            ) from exc

    @staticmethod
    def _normalize_decision(
        decision: str
        | ApprovalDecision,
    ) -> str:
        if isinstance(
            decision, ApprovalDecision
        ):
            return decision.value

        if not isinstance(decision, str):
            raise ValidationError(
                "decision must be a string or ApprovalDecision."
            )

        normalized_decision = (
            decision.strip().lower()
        )

        try:
            return ApprovalDecision(
                normalized_decision
            ).value
        except ValueError as exc:
            raise ValidationError(
                f"Unsupported approval decision: {decision!r}."
            ) from exc

    @classmethod
    async def create_approval(
        cls,
        db: AsyncSession,
        execution_id: str | UUID,
        tenant_id: str | UUID,
        requested_by: str | UUID,
        reason: str,
        workflow_step: str,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> ApprovalModel:
        """Create a pending approval request."""
        normalized_execution_id = (
            cls.normalize_uuid(
                execution_id,
                "execution_id",
            )
        )
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )
        normalized_requested_by = (
            cls.normalize_uuid(
                requested_by,
                "requested_by",
            )
        )

        reason = cls.normalize_text(
            reason,
            "reason",
            required=True,
            max_length=2_000,
        )

        if not isinstance(
            workflow_step, str
        ) or not workflow_step.strip():
            raise ValidationError(
                "workflow_step must be a non-empty string."
            )

        if metadata is not None and not isinstance(
            metadata, dict
        ):
            raise ValidationError(
                "metadata must be a dictionary."
            )

        db_record = ApprovalModel(
            execution_id=str(
                normalized_execution_id
            ),
            tenant_id=str(normalized_tenant_id),
            requested_by=str(
                normalized_requested_by
            ),
            reason=reason,
            workflow_step=workflow_step.strip(),
            status=ApprovalStatus.PENDING.value,
            metadata=metadata or {},
        )

        db.add(db_record)
        await db.flush()
        await db.refresh(db_record)

        logger.info(
            "Created approval request.",
            extra={
                "approval_id": str(
                    db_record.id
                ),
                "execution_id": str(
                    normalized_execution_id
                ),
                "tenant_id": str(
                    normalized_tenant_id
                ),
            },
        )

        return db_record

    @classmethod
    async def get_approval(
        cls,
        db: AsyncSession,
        approval_id: str | UUID,
        tenant_id: str | UUID,
    ) -> Optional[ApprovalModel]:
        """Retrieve a tenant-scoped approval by ID."""
        return await cls.get_by_id(
            db,
            record_id=approval_id,
            tenant_id=tenant_id,
        )

    @classmethod
    async def list_approvals_for_execution(
        cls,
        db: AsyncSession,
        execution_id: str | UUID,
        tenant_id: str | UUID,
    ) -> List[ApprovalModel]:
        """List approvals for an execution, newest first."""
        normalized_execution_id = (
            cls.normalize_uuid(
                execution_id,
                "execution_id",
            )
        )
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )

        stmt = (
            select(cls.model)
            .where(
                cls.model.execution_id
                == str(
                    normalized_execution_id
                )
            )
            .where(
                cls.model.tenant_id
                == str(normalized_tenant_id)
            )
            .order_by(
                cls.model.created_at.desc()
            )
        )

        result = await db.execute(stmt)

        return list(
            result.scalars().all()
        )

    @classmethod
    async def list_pending_approvals(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> tuple[
        list[ApprovalModel], int
    ]:
        """List pending approvals for a tenant."""
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )
        normalized_skip, normalized_limit = (
            cls.normalize_pagination(
                skip, limit
            )
        )

        filters = [
            cls.model.tenant_id
            == str(normalized_tenant_id),
            cls.model.status
            == ApprovalStatus.PENDING.value,
        ]

        count_stmt = (
            select(func.count())
            .select_from(cls.model)
            .where(*filters)
        )

        total_result = await db.execute(
            count_stmt
        )
        total_count = (
            total_result.scalar_one()
        )

        list_stmt = (
            select(cls.model)
            .where(*filters)
            .order_by(
                cls.model.created_at.asc()
            )
            .offset(normalized_skip)
            .limit(normalized_limit)
        )

        result = await db.execute(
            list_stmt
        )

        return (
            list(result.scalars().all()),
            total_count,
        )

    @classmethod
    async def resolve_approval(
        cls,
        db: AsyncSession,
        approval_id: str | UUID,
        tenant_id: str | UUID,
        reviewer_id: str | UUID,
        decision: str
        | ApprovalDecision,
        comment: Optional[str] = None,
    ) -> Optional[ApprovalModel]:
        """Resolve a pending approval with reviewer identity and decision."""
        normalized_approval_id = (
            cls.normalize_uuid(
                approval_id,
                "approval_id",
            )
        )
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )
        normalized_reviewer_id = (
            cls.normalize_uuid(
                reviewer_id,
                "reviewer_id",
            )
        )
        normalized_decision = (
            cls._normalize_decision(
                decision
            )
        )
        normalized_comment = (
            cls.normalize_text(
                comment,
                "comment",
                max_length=2_000,
            )
        )

        record = await cls.get_by_id(
            db,
            record_id=normalized_approval_id,
            tenant_id=normalized_tenant_id,
        )

        if not record:
            return None

        if (
            record.status
            != ApprovalStatus.PENDING.value
        ):
            raise ValidationError(
                "Only pending approvals can be resolved."
            )

        record.reviewer_id = str(
            normalized_reviewer_id
        )
        record.decision = (
            normalized_decision
        )
        record.comment = normalized_comment
        record.status = (
            ApprovalStatus.APPROVED.value
            if normalized_decision
            == ApprovalDecision.APPROVED.value
            else ApprovalStatus.REJECTED.value
            if normalized_decision
            == ApprovalDecision.REJECTED.value
            else ApprovalStatus.CHANGES_REQUESTED.value
        )
        record.resolved_at = datetime.now(
            timezone.utc
        )

        await db.flush()
        await db.refresh(record)

        logger.info(
            "Resolved approval request.",
            extra={
                "approval_id": str(
                    normalized_approval_id
                ),
                "tenant_id": str(
                    normalized_tenant_id
                ),
                "decision": normalized_decision,
            },
        )

        return record

    @classmethod
    async def count_approvals(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        status: str
        | ApprovalStatus
        | None = None,
    ) -> int:
        """Count tenant-scoped approvals, optionally by status."""
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )
        normalized_status = (
            cls._normalize_status(status)
        )

        stmt = (
            select(func.count())
            .select_from(cls.model)
            .where(
                cls.model.tenant_id
                == str(normalized_tenant_id)
            )
        )

        if normalized_status:
            stmt = stmt.where(
                cls.model.status
                == normalized_status
            )

        result = await db.execute(stmt)

        return result.scalar_one()