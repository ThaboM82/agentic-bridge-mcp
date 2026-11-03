"""
Workflow checkpoint repository for durable graph-state recovery.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ValidationError
from app.models.workflow_checkpoint_model import WorkflowCheckpointModel
from app.repositories.base_repository import BaseRepository


logger = logging.getLogger(
    "agentic_bridge.repositories.workflow_checkpoint"
)


class WorkflowCheckpointRepository(
    BaseRepository[
        WorkflowCheckpointModel
    ]
):
    """Tenant-scoped persistence for workflow graph checkpoints."""

    model = WorkflowCheckpointModel

    @classmethod
    async def save_checkpoint(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        execution_id: str | UUID,
        checkpoint_id: str,
        checkpoint_data: Dict[str, Any],
        workflow_step: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> WorkflowCheckpointModel:
        """Persist a workflow checkpoint snapshot."""
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )
        normalized_execution_id = (
            cls.normalize_uuid(
                execution_id,
                "execution_id",
            )
        )

        checkpoint_id = cls.normalize_text(
            checkpoint_id,
            "checkpoint_id",
            required=True,
            max_length=200,
        )

        if not isinstance(
            checkpoint_data, dict
        ):
            raise ValidationError(
                "checkpoint_data must be a dictionary."
            )

        workflow_step = cls.normalize_text(
            workflow_step,
            "workflow_step",
            max_length=100,
        )

        if metadata is not None and not isinstance(
            metadata, dict
        ):
            raise ValidationError(
                "metadata must be a dictionary."
            )

        db_record = WorkflowCheckpointModel(
            tenant_id=str(
                normalized_tenant_id
            ),
            execution_id=str(
                normalized_execution_id
            ),
            checkpoint_id=checkpoint_id,
            checkpoint_data=checkpoint_data,
            workflow_step=workflow_step,
            metadata=metadata or {},
        )

        db.add(db_record)
        await db.flush()
        await db.refresh(db_record)

        logger.info(
            "Saved workflow checkpoint.",
            extra={
                "checkpoint_id": checkpoint_id,
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
    async def get_latest_checkpoint(
        cls,
        db: AsyncSession,
        execution_id: str | UUID,
        tenant_id: str | UUID,
    ) -> Optional[
        WorkflowCheckpointModel
    ]:
        """Retrieve the newest checkpoint for an execution."""
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
            .limit(1)
        )

        result = await db.execute(stmt)

        return result.scalar_one_or_none()

    @classmethod
    async def list_checkpoints(
        cls,
        db: AsyncSession,
        execution_id: str | UUID,
        tenant_id: str | UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> tuple[
        list[
            WorkflowCheckpointModel
        ],
        int,
    ]:
        """List checkpoints for a tenant-scoped execution."""
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
        normalized_skip, normalized_limit = (
            cls.normalize_pagination(
                skip, limit
            )
        )

        filters = [
            cls.model.tenant_id
            == str(normalized_tenant_id),
            cls.model.execution_id
            == str(
                normalized_execution_id
            ),
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
                cls.model.created_at.desc()
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
    async def delete_checkpoints_for_execution(
        cls,
        db: AsyncSession,
        execution_id: str | UUID,
        tenant_id: str | UUID,
    ) -> int:
        """Delete all checkpoints for a tenant execution."""
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
            delete(cls.model)
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
        )

        result = await db.execute(stmt)
        await db.flush()

        return result.rowcount or 0