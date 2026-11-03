"""
User repository.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import UserRole
from app.core.exceptions import ValidationError
from app.models.user_model import UserModel
from app.repositories.base_repository import BaseRepository


logger = logging.getLogger(
    "agentic_bridge.repositories.user"
)


class UserRepository(
    BaseRepository[UserModel]
):
    """Tenant-scoped persistence for user profiles."""

    model = UserModel

    @staticmethod
    def normalize_email(
        email: str,
    ) -> str:
        """Normalize an email address."""
        if not isinstance(email, str):
            raise ValidationError(
                "email must be a string."
            )

        normalized_email = email.strip().lower()

        if (
            not normalized_email
            or "@" not in normalized_email
        ):
            raise ValidationError(
                "email must be a valid email address."
            )

        return normalized_email

    @staticmethod
    def normalize_role(
        role: str | UserRole | None,
    ) -> str:
        """Normalize a user role."""
        if role is None:
            return UserRole.VIEWER.value

        if isinstance(role, UserRole):
            return role.value

        if not isinstance(role, str):
            raise ValidationError(
                "role must be a string or UserRole."
            )

        normalized_role = role.strip().lower()

        try:
            return UserRole(
                normalized_role
            ).value
        except ValueError as exc:
            raise ValidationError(
                f"Unsupported user role: {role!r}."
            ) from exc

    @classmethod
    async def create_user(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        email: str,
        password_hash: str,
        full_name: Optional[str] = None,
        role: str | UserRole | None = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> UserModel:
        """Create a tenant-scoped user record."""
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )
        normalized_email = (
            cls.normalize_email(email)
        )
        normalized_role = (
            cls.normalize_role(role)
        )

        if not isinstance(
            password_hash, str
        ) or not password_hash.strip():
            raise ValidationError(
                "password_hash is required."
            )

        full_name = cls.normalize_text(
            full_name,
            "full_name",
            max_length=200,
        )

        if metadata is not None and not isinstance(
            metadata, dict
        ):
            raise ValidationError(
                "metadata must be a dictionary."
            )

        existing = (
            await cls.get_user_by_email(
                db,
                tenant_id=normalized_tenant_id,
                email=normalized_email,
            )
        )

        if existing:
            raise ValidationError(
                "A user with this email already exists."
            )

        db_record = UserModel(
            tenant_id=str(
                normalized_tenant_id
            ),
            email=normalized_email,
            password_hash=password_hash,
            full_name=full_name,
            role=normalized_role,
            metadata=metadata or {},
            is_active=True,
        )

        db.add(db_record)
        await db.flush()
        await db.refresh(db_record)

        logger.info(
            "Created user record.",
            extra={
                "user_id": str(
                    db_record.id
                ),
                "tenant_id": str(
                    normalized_tenant_id
                ),
            },
        )

        return db_record

    @classmethod
    async def get_user(
        cls,
        db: AsyncSession,
        user_id: str | UUID,
        tenant_id: str | UUID,
    ) -> Optional[UserModel]:
        """Retrieve a tenant-scoped user."""
        return await cls.get_by_id(
            db,
            record_id=user_id,
            tenant_id=tenant_id,
        )

    @classmethod
    async def get_user_by_email(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        email: str,
    ) -> Optional[UserModel]:
        """Retrieve a tenant-scoped user by email."""
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )
        normalized_email = (
            cls.normalize_email(email)
        )

        stmt = (
            select(cls.model)
            .where(
                cls.model.tenant_id
                == str(normalized_tenant_id)
            )
            .where(
                cls.model.email
                == normalized_email
            )
        )

        result = await db.execute(stmt)

        return result.scalar_one_or_none()

    @classmethod
    async def list_users_by_tenant(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        skip: int = 0,
        limit: int = 20,
        role_filter: Optional[
            str | UserRole
        ] = None,
    ) -> tuple[
        list[UserModel], int
    ]:
        """List tenant-scoped users with optional role filtering."""
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
            == str(normalized_tenant_id)
        ]

        if role_filter is not None:
            filters.append(
                cls.model.role
                == cls.normalize_role(
                    role_filter
                )
            )

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
    async def update_role(
        cls,
        db: AsyncSession,
        user_id: str | UUID,
        tenant_id: str | UUID,
        role: str | UserRole,
    ) -> Optional[UserModel]:
        """Update a tenant-scoped user role."""
        record = await cls.get_by_id(
            db,
            record_id=user_id,
            tenant_id=tenant_id,
        )

        if not record:
            return None

        record.role = cls.normalize_role(
            role
        )
        record.updated_at = datetime.now(
            timezone.utc
        )

        await db.flush()
        await db.refresh(record)

        return record

    @classmethod
    async def deactivate_user(
        cls,
        db: AsyncSession,
        user_id: str | UUID,
        tenant_id: str | UUID,
    ) -> Optional[UserModel]:
        """Deactivate a tenant-scoped user."""
        record = await cls.get_by_id(
            db,
            record_id=user_id,
            tenant_id=tenant_id,
        )

        if not record:
            return None

        record.is_active = False
        record.updated_at = datetime.now(
            timezone.utc
        )

        await db.flush()
        await db.refresh(record)

        return record