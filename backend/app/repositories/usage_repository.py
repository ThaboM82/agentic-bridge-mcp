"""
Provider usage and token-accounting repository.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ValidationError
from app.models.usage_model import UsageModel
from app.repositories.base_repository import BaseRepository


logger = logging.getLogger(
    "agentic_bridge.repositories.usage"
)


class UsageRepository(
    BaseRepository[UsageModel]
):
    """Tenant-scoped persistence for provider token usage."""

    model = UsageModel

    @classmethod
    async def record_usage(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        execution_id: str | UUID,
        provider_name: str,
        provider_model: str,
        prompt_tokens: int,
        completion_tokens: int,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> UsageModel:
        """Record a provider usage event."""
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

        provider_name = cls.normalize_text(
            provider_name,
            "provider_name",
            required=True,
            max_length=100,
        )
        provider_model = (
            cls.normalize_text(
                provider_model,
                "provider_model",
                required=True,
                max_length=150,
            )
        )

        for field_name, value in (
            (
                "prompt_tokens",
                prompt_tokens,
            ),
            (
                "completion_tokens",
                completion_tokens,
            ),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValidationError(
                    f"{field_name} must be a non-negative integer."
                )

        if metadata is not None and not isinstance(
            metadata, dict
        ):
            raise ValidationError(
                "metadata must be a dictionary."
            )

        db_record = UsageModel(
            tenant_id=str(
                normalized_tenant_id
            ),
            execution_id=str(
                normalized_execution_id
            ),
            provider_name=provider_name,
            provider_model=provider_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens
            + completion_tokens,
            metadata=metadata or {},
        )

        db.add(db_record)
        await db.flush()
        await db.refresh(db_record)

        logger.info(
            "Recorded provider usage.",
            extra={
                "usage_id": str(
                    db_record.id
                ),
                "tenant_id": str(
                    normalized_tenant_id
                ),
                "execution_id": str(
                    normalized_execution_id
                ),
                "provider": provider_name,
                "model": provider_model,
            },
        )

        return db_record

    @classmethod
    async def get_usage_for_execution(
        cls,
        db: AsyncSession,
        execution_id: str | UUID,
        tenant_id: str | UUID,
    ) -> List[UsageModel]:
        """List usage records for a tenant execution."""
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
                cls.model.created_at.asc()
            )
        )

        result = await db.execute(stmt)

        return list(
            result.scalars().all()
        )

    @classmethod
    async def summarize_usage_by_tenant(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
    ) -> Dict[str, Any]:
        """Summarize total token usage for a tenant."""
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )

        stmt = (
            select(
                func.count(
                    cls.model.id
                ),
                func.coalesce(
                    func.sum(
                        cls.model.prompt_tokens
                    ),
                    0,
                ),
                func.coalesce(
                    func.sum(
                        cls.model.completion_tokens
                    ),
                    0,
                ),
                func.coalesce(
                    func.sum(
                        cls.model.total_tokens
                    ),
                    0,
                ),
            ).where(
                cls.model.tenant_id
                == str(normalized_tenant_id)
            )
        )

        result = await db.execute(stmt)
        row = result.one()

        return {
            "tenant_id": str(
                normalized_tenant_id
            ),
            "usage_events": row[0],
            "prompt_tokens": row[1],
            "completion_tokens": row[2],
            "total_tokens": row[3],
        }

    @classmethod
    async def summarize_usage_by_model(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
    ) -> List[Dict[str, Any]]:
        """Summarize token usage grouped by provider and model."""
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )

        stmt = (
            select(
                cls.model.provider_name,
                cls.model.provider_model,
                func.count(
                    cls.model.id
                ),
                func.coalesce(
                    func.sum(
                        cls.model.prompt_tokens
                    ),
                    0,
                ),
                func.coalesce(
                    func.sum(
                        cls.model.completion_tokens
                    ),
                    0,
                ),
                func.coalesce(
                    func.sum(
                        cls.model.total_tokens
                    ),
                    0,
                ),
            )
            .where(
                cls.model.tenant_id
                == str(normalized_tenant_id)
            )
            .group_by(
                cls.model.provider_name,
                cls.model.provider_model,
            )
            .order_by(
                cls.model.provider_name,
                cls.model.provider_model,
            )
        )

        result = await db.execute(stmt)

        return [
            {
                "provider_name": row[0],
                "provider_model": row[1],
                "usage_events": row[2],
                "prompt_tokens": row[3],
                "completion_tokens": row[4],
                "total_tokens": row[5],
            }
            for row in result.all()
        ]