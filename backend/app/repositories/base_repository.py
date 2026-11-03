"""
Base repository contract for tenant-scoped persistence operations.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Generic, Sequence, TypeVar
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ValidationError


logger = logging.getLogger(
    "agentic_bridge.repositories"
)


ModelType = TypeVar("ModelType")


class BaseRepository(Generic[ModelType]):
    """
    Base repository providing common tenant-scoped query, pagination,
    identifier, and embedding-validation behavior.
    """

    model: type[ModelType]

    @staticmethod
    def normalize_uuid(
        value: str | UUID,
        field_name: str,
    ) -> UUID:
        """Validate and normalize a UUID identifier."""
        if isinstance(value, UUID):
            return value

        if not isinstance(value, str):
            raise ValidationError(
                f"{field_name} must be a UUID string."
            )

        normalized_value = value.strip()

        if not normalized_value:
            raise ValidationError(
                f"{field_name} must not be empty."
            )

        try:
            return UUID(normalized_value)
        except ValueError as exc:
            raise ValidationError(
                f"{field_name} must be a valid UUID."
            ) from exc

    @staticmethod
    def normalize_optional_uuid(
        value: str | UUID | None,
        field_name: str,
    ) -> UUID | None:
        """Validate an optional UUID identifier."""
        if value is None:
            return None

        return BaseRepository.normalize_uuid(
            value,
            field_name,
        )

    @staticmethod
    def normalize_pagination(
        skip: int,
        limit: int,
        max_limit: int = 100,
    ) -> tuple[int, int]:
        """Validate and clamp pagination inputs."""
        if not isinstance(skip, int) or isinstance(
            skip, bool
        ):
            raise ValidationError(
                "skip must be an integer."
            )

        if not isinstance(limit, int) or isinstance(
            limit, bool
        ):
            raise ValidationError(
                "limit must be an integer."
            )

        if skip < 0:
            raise ValidationError(
                "skip must be greater than or equal to zero."
            )

        if limit <= 0:
            raise ValidationError(
                "limit must be greater than zero."
            )

        return skip, min(limit, max_limit)

    @staticmethod
    def normalize_embedding(
        embedding: Sequence[float] | None,
        *,
        max_dimensions: int = 10_000,
    ) -> list[float] | None:
        """Validate an optional numeric embedding vector."""
        if embedding is None:
            return None

        if not isinstance(
            embedding, (list, tuple)
        ):
            raise ValidationError(
                "embedding must be a list or tuple of numeric values."
            )

        vector = list(embedding)

        if not vector:
            raise ValidationError(
                "embedding must not be empty."
            )

        if len(vector) > max_dimensions:
            raise ValidationError(
                "embedding exceeds the maximum supported dimensions."
            )

        normalized_vector: list[float] = []

        for value in vector:
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise ValidationError(
                    "embedding values must be numeric."
                )

            float_value = float(value)

            if not math.isfinite(float_value):
                raise ValidationError(
                    "embedding values must be finite."
                )

            normalized_vector.append(
                float_value
            )

        return normalized_vector

    @staticmethod
    def normalize_text(
        value: str | None,
        field_name: str,
        *,
        required: bool = False,
        max_length: int | None = None,
    ) -> str | None:
        """Normalize optional or required text input."""
        if value is None:
            if required:
                raise ValidationError(
                    f"{field_name} is required."
                )

            return None

        if not isinstance(value, str):
            raise ValidationError(
                f"{field_name} must be a string."
            )

        normalized_value = value.strip()

        if not normalized_value:
            if required:
                raise ValidationError(
                    f"{field_name} must not be empty."
                )

            return None

        if (
            max_length is not None
            and len(normalized_value)
            > max_length
        ):
            raise ValidationError(
                f"{field_name} exceeds the maximum length of {max_length}."
            )

        return normalized_value

    @staticmethod
    def normalize_datetime(
        value: datetime | None,
        field_name: str,
    ) -> datetime | None:
        """Validate an optional datetime filter."""
        if value is None:
            return None

        if not isinstance(value, datetime):
            raise ValidationError(
                f"{field_name} must be a datetime."
            )

        return value

    @classmethod
    async def get_by_id(
        cls,
        db: AsyncSession,
        record_id: str | UUID,
        tenant_id: str | UUID,
    ) -> ModelType | None:
        """Retrieve one tenant-scoped record by primary key."""
        normalized_id = cls.normalize_uuid(
            record_id,
            "record_id",
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
                cls.model.id
                == normalized_id
            )
            .where(
                cls.model.tenant_id
                == normalized_tenant_id
            )
        )

        result = await db.execute(stmt)

        return result.scalar_one_or_none()

    @classmethod
    async def exists(
        cls,
        db: AsyncSession,
        record_id: str | UUID,
        tenant_id: str | UUID,
    ) -> bool:
        """Return whether a tenant-scoped record exists."""
        record = await cls.get_by_id(
            db,
            record_id=record_id,
            tenant_id=tenant_id,
        )

        return record is not None

    @classmethod
    async def list_for_tenant(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> tuple[
        list[ModelType], int
    ]:
        """List tenant-scoped records with pagination and total count."""
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
            == normalized_tenant_id
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
    async def delete_by_id(
        cls,
        db: AsyncSession,
        record_id: str | UUID,
        tenant_id: str | UUID,
    ) -> bool:
        """Delete one tenant-scoped record by primary key."""
        normalized_id = cls.normalize_uuid(
            record_id,
            "record_id",
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
                cls.model.id
                == normalized_id
            )
            .where(
                cls.model.tenant_id
                == normalized_tenant_id
            )
        )

        result = await db.execute(stmt)
        await db.flush()

        return bool(result.rowcount)