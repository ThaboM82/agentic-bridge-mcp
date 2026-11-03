"""
CRUD operations for users.

This module owns direct persistence operations for user accounts. API routes
and services should call these functions rather than constructing SQLAlchemy
queries directly.

Responsibilities:
- Create, retrieve, update, activate, deactivate, and delete users.
- Look up users by email or username.
- List and paginate users with filters.
- Track login activity and failed authentication attempts.
- Map database failures to stable application exceptions.

Security rules:
- Never store plaintext passwords in this module.
- Services must pass only a secure password hash.
- Never log passwords, password hashes, tokens, or full email addresses.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import (
    and_,
    func,
    or_,
    select,
)
from sqlalchemy.exc import (
    IntegrityError,
    SQLAlchemyError,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import UserRole
from app.core.exceptions import (
    ConflictError,
    DatabaseError,
    NotFoundError,
    ValidationError,
)
from app.core.pagination import (
    Page,
    PaginationParams,
)
from app.core.utils import mask_email
from app.models.user import User as UserModel


logger = logging.getLogger(
    "agentic_bridge.crud.user",
)


MAX_EMAIL_LENGTH = 254
MAX_USERNAME_LENGTH = 64
MAX_FULL_NAME_LENGTH = 128
MAX_FAILED_LOGIN_ATTEMPTS = 100
MAX_METADATA_JSON_CHARS = 16_000


SENSITIVE_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "password_hash",
        "secret",
        "secret_key",
        "api_key",
        "apikey",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "authorization",
        "cookie",
        "database_url",
        "connection_string",
    }
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _normalize_user_id(
    user_id: str | UUID,
) -> UUID:
    """Validate and normalize a user ID."""
    if isinstance(user_id, UUID):
        return user_id

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValidationError(
            "user_id must be a non-empty string.",
        )

    try:
        return UUID(user_id.strip())
    except ValueError as exc:
        raise ValidationError(
            "user_id must be a valid UUID.",
        ) from exc


def _normalize_email(
    email: str,
) -> str:
    """
    Validate and normalize an email address.

    Email addresses are stored in lowercase for deterministic lookup.
    """
    if not isinstance(email, str) or not email.strip():
        raise ValidationError(
            "email is required.",
        )

    normalized_email = email.strip().lower()

    if len(normalized_email) > MAX_EMAIL_LENGTH:
        raise ValidationError(
            f"Email may not exceed "
            f"{MAX_EMAIL_LENGTH} characters.",
        )

    if (
        "@" not in normalized_email
        or normalized_email.startswith("@")
        or normalized_email.endswith("@")
        or " " in normalized_email
    ):
        raise ValidationError(
            "Email address is invalid.",
        )

    return normalized_email


def _normalize_username(
    username: str,
) -> str:
    """Validate and normalize a username."""
    if not isinstance(username, str) or not username.strip():
        raise ValidationError(
            "username is required.",
        )

    normalized_username = username.strip().lower()

    if len(normalized_username) > MAX_USERNAME_LENGTH:
        raise ValidationError(
            f"Username may not exceed "
            f"{MAX_USERNAME_LENGTH} characters.",
        )

    if not normalized_username.replace(
        "-",
        "",
    ).replace(
        "_",
        "",
    ).isalnum():
        raise ValidationError(
            "Username may contain only letters, "
            "numbers, hyphens, and underscores.",
        )

    return normalized_username


def _normalize_full_name(
    full_name: str | None,
) -> str | None:
    """Validate and normalize an optional full name."""
    if full_name is None:
        return None

    if not isinstance(full_name, str):
        raise ValidationError(
            "full_name must be a string.",
        )

    normalized_name = full_name.strip()

    if not normalized_name:
        return None

    if len(normalized_name) > MAX_FULL_NAME_LENGTH:
        raise ValidationError(
            f"Full name may not exceed "
            f"{MAX_FULL_NAME_LENGTH} characters.",
        )

    return normalized_name


def _normalize_role(
    role: str | UserRole | None,
) -> UserRole:
    """Validate and normalize a user role."""
    if role is None:
        return UserRole.VIEWER

    try:
        return UserRole(role)
    except ValueError as exc:
        raise ValidationError(
            f"Unsupported user role: {role}.",
        ) from exc


def _validate_password_hash(
    password_hash: str,
) -> str:
    """
    Validate that a password hash was supplied.

    This function does not validate the hash algorithm; password hashing and
    verification belong in the authentication service.
    """
    if not isinstance(password_hash, str) or not password_hash.strip():
        raise ValidationError(
            "password_hash is required.",
        )

    normalized_hash = password_hash.strip()

    if len(normalized_hash) < 20:
        raise ValidationError(
            "password_hash appears to be invalid "
            "or too short.",
        )

    return normalized_hash


def _validate_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate user metadata and reject sensitive keys."""
    if metadata is None:
        return {}

    if not isinstance(metadata, dict):
        raise ValidationError(
            "User metadata must be an object.",
        )

    for key in metadata:
        if str(key).lower() in SENSITIVE_METADATA_KEYS:
            raise ValidationError(
                f"Sensitive field '{key}' cannot be "
                "stored in user metadata.",
            )

    if len(str(metadata)) > MAX_METADATA_JSON_CHARS:
        raise ValidationError(
            "User metadata exceeds the maximum "
            "allowed size.",
        )

    return metadata


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(
        timezone.utc,
    )


def _apply_tenant_filter(
    query,
    tenant_id: str | None,
):
    """Apply tenant isolation to a user query."""
    if tenant_id:
        return query.where(
            UserModel.tenant_id == tenant_id,
        )

    return query


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


async def create_user(
    db: AsyncSession,
    *,
    email: str,
    username: str,
    password_hash: str,
    full_name: str | None = None,
    role: str | UserRole = UserRole.VIEWER,
    tenant_id: str | None = None,
    is_active: bool = True,
    is_verified: bool = False,
    metadata: dict[str, Any] | None = None,
) -> UserModel:
    """
    Create and persist a new user.
    """
    safe_email = _normalize_email(email)
    safe_username = _normalize_username(username)
    safe_password_hash = _validate_password_hash(
        password_hash,
    )
    safe_full_name = _normalize_full_name(
        full_name,
    )
    safe_role = _normalize_role(role)
    safe_metadata = _validate_metadata(metadata)

    user = UserModel(
        email=safe_email,
        username=safe_username,
        password_hash=safe_password_hash,
        full_name=safe_full_name,
        role=safe_role.value,
        tenant_id=tenant_id,
        is_active=is_active,
        is_verified=is_verified,
        metadata=safe_metadata,
    )

    try:
        db.add(user)
        await db.commit()
        await db.refresh(user)

    except IntegrityError as exc:
        await db.rollback()

        logger.warning(
            "user_create_conflict",
            extra={
                "email": mask_email(safe_email),
                "username": safe_username,
                "tenant_id": tenant_id,
            },
        )

        raise ConflictError(
            "A user with the provided email address "
            "or username already exists.",
        ) from exc

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "user_create_failed",
        )

        raise DatabaseError(
            "The user could not be created.",
        ) from exc

    logger.info(
        "user_created",
        extra={
            "user_id": str(user.id),
            "email": mask_email(safe_email),
            "username": safe_username,
            "role": user.role,
            "tenant_id": tenant_id,
        },
    )

    return user


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


async def get_user(
    db: AsyncSession,
    user_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> UserModel | None:
    """
    Retrieve a user by ID.
    """
    user_uuid = _normalize_user_id(user_id)

    query = select(UserModel).where(
        UserModel.id == user_uuid,
    )

    query = _apply_tenant_filter(
        query,
        tenant_id,
    )

    result = await db.execute(query)

    return result.scalar_one_or_none()


async def get_user_or_404(
    db: AsyncSession,
    user_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> UserModel:
    """
    Retrieve a user by ID or raise `NotFoundError`.
    """
    user = await get_user(
        db,
        user_id,
        tenant_id=tenant_id,
    )

    if user is None:
        raise NotFoundError(
            "User was not found.",
            details={
                "user_id": str(user_id),
            },
        )

    return user


async def get_user_by_email(
    db: AsyncSession,
    email: str,
    *,
    tenant_id: str | None = None,
) -> UserModel | None:
    """
    Retrieve a user by email address.
    """
    safe_email = _normalize_email(email)

    query = select(UserModel).where(
        UserModel.email == safe_email,
    )

    query = _apply_tenant_filter(
        query,
        tenant_id,
    )

    result = await db.execute(query)

    return result.scalar_one_or_none()


async def get_user_by_username(
    db: AsyncSession,
    username: str,
    *,
    tenant_id: str | None = None,
) -> UserModel | None:
    """
    Retrieve a user by username.
    """
    safe_username = _normalize_username(
        username,
    )

    query = select(UserModel).where(
        UserModel.username == safe_username,
    )

    query = _apply_tenant_filter(
        query,
        tenant_id,
    )

    result = await db.execute(query)

    return result.scalar_one_or_none()


async def get_user_by_email_or_username(
    db: AsyncSession,
    identifier: str,
    *,
    tenant_id: str | None = None,
) -> UserModel | None:
    """
    Retrieve a user by either email address or username.
    """
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValidationError(
            "Email address or username is required.",
        )

    normalized_identifier = (
        identifier.strip().lower()
    )

    query = select(UserModel).where(
        or_(
            UserModel.email
            == normalized_identifier,
            UserModel.username
            == normalized_identifier,
        ),
    )

    query = _apply_tenant_filter(
        query,
        tenant_id,
    )

    result = await db.execute(query)

    return result.scalar_one_or_none()


async def list_users(
    db: AsyncSession,
    *,
    pagination: PaginationParams,
    tenant_id: str | None = None,
    role: str | UserRole | None = None,
    is_active: bool | None = None,
    is_verified: bool | None = None,
    search: str | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
) -> Page[UserModel]:
    """
    Retrieve a paginated, filtered list of users.
    """
    safe_role = _normalize_role(role)

    filters = []

    if tenant_id:
        filters.append(
            UserModel.tenant_id == tenant_id,
        )

    if role is not None:
        filters.append(
            UserModel.role == safe_role.value,
        )

    if is_active is not None:
        filters.append(
            UserModel.is_active.is_(is_active),
        )

    if is_verified is not None:
        filters.append(
            UserModel.is_verified.is_(is_verified),
        )

    if created_after is not None:
        filters.append(
            UserModel.created_at >= created_after,
        )

    if created_before is not None:
        filters.append(
            UserModel.created_at <= created_before,
        )

    if search:
        search_term = f"%{search.strip()}%"

        filters.append(
            or_(
                UserModel.email.ilike(
                    search_term,
                ),
                UserModel.username.ilike(
                    search_term,
                ),
                UserModel.full_name.ilike(
                    search_term,
                ),
            ),
        )

    count_query = select(
        func.count(),
    ).select_from(UserModel)

    if filters:
        count_query = count_query.where(
            and_(*filters),
        )

    count_result = await db.execute(
        count_query,
    )

    total = int(
        count_result.scalar_one(),
    )

    query = select(UserModel).order_by(
        UserModel.created_at.desc(),
        UserModel.id.desc(),
    )

    if filters:
        query = query.where(
            and_(*filters),
        )

    result = await db.execute(
        query.offset(
            pagination.skip,
        ).limit(
            pagination.limit,
        ),
    )

    users = list(
        result.scalars().all(),
    )

    return Page[UserModel].create(
        items=users,
        total=total,
        skip=pagination.skip,
        limit=pagination.limit,
    )


async def count_users(
    db: AsyncSession,
    *,
    tenant_id: str | None = None,
    role: str | UserRole | None = None,
    is_active: bool | None = None,
) -> int:
    """
    Count users matching optional filters.
    """
    safe_role = _normalize_role(role)

    filters = []

    if tenant_id:
        filters.append(
            UserModel.tenant_id == tenant_id,
        )

    if role is not None:
        filters.append(
            UserModel.role == safe_role.value,
        )

    if is_active is not None:
        filters.append(
            UserModel.is_active.is_(is_active),
        )

    query = select(
        func.count(),
    ).select_from(UserModel)

    if filters:
        query = query.where(
            and_(*filters),
        )

    result = await db.execute(query)

    return int(
        result.scalar_one(),
    )


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


async def update_user_profile(
    db: AsyncSession,
    user: UserModel,
    *,
    full_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> UserModel:
    """
    Update editable user profile fields.
    """
    if full_name is not None:
        user.full_name = _normalize_full_name(
            full_name,
        )

    if metadata is not None:
        user.metadata = _validate_metadata(
            metadata,
        )

    user.updated_at = _utc_now()

    try:
        await db.commit()
        await db.refresh(user)

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "user_profile_update_failed",
        )

        raise DatabaseError(
            "The user profile could not be updated.",
        ) from exc

    logger.info(
        "user_profile_updated",
        extra={
            "user_id": str(user.id),
            "updated_fields": [
                field
                for field, value in (
                    ("full_name", full_name),
                    ("metadata", metadata),
                )
                if value is not None
            ],
        },
    )

    return user


async def update_user_role(
    db: AsyncSession,
    user_id: str | UUID,
    role: str | UserRole,
    *,
    tenant_id: str | None = None,
) -> UserModel:
    """
    Update a user's role.
    """
    safe_role = _normalize_role(role)

    user = await get_user_or_404(
        db,
        user_id,
        tenant_id=tenant_id,
    )

    previous_role = user.role
    user.role = safe_role.value
    user.updated_at = _utc_now()

    try:
        await db.commit()
        await db.refresh(user)

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "user_role_update_failed",
        )

        raise DatabaseError(
            "The user role could not be updated.",
        ) from exc

    logger.warning(
        "user_role_updated",
        extra={
            "user_id": str(user.id),
            "previous_role": previous_role,
            "role": user.role,
            "tenant_id": tenant_id,
        },
    )

    return user


async def update_password_hash(
    db: AsyncSession,
    user_id: str | UUID,
    password_hash: str,
    *,
    tenant_id: str | None = None,
) -> UserModel:
    """
    Replace a user's password hash.
    """
    safe_password_hash = _validate_password_hash(
        password_hash,
    )

    user = await get_user_or_404(
        db,
        user_id,
        tenant_id=tenant_id,
    )

    user.password_hash = safe_password_hash
    user.password_changed_at = _utc_now()
    user.failed_login_attempts = 0
    user.locked_until = None
    user.updated_at = _utc_now()

    try:
        await db.commit()
        await db.refresh(user)

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "user_password_update_failed",
        )

        raise DatabaseError(
            "The password could not be updated.",
        ) from exc

    logger.warning(
        "user_password_changed",
        extra={
            "user_id": str(user.id),
            "tenant_id": tenant_id,
        },
    )

    return user


async def activate_user(
    db: AsyncSession,
    user_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> UserModel:
    """
    Activate a user account.
    """
    user = await get_user_or_404(
        db,
        user_id,
        tenant_id=tenant_id,
    )

    user.is_active = True
    user.locked_until = None
    user.failed_login_attempts = 0
    user.updated_at = _utc_now()

    try:
        await db.commit()
        await db.refresh(user)

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "user_activation_failed",
        )

        raise DatabaseError(
            "The user could not be activated.",
        ) from exc

    logger.info(
        "user_activated",
        extra={
            "user_id": str(user.id),
            "tenant_id": tenant_id,
        },
    )

    return user


async def deactivate_user(
    db: AsyncSession,
    user_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> UserModel:
    """
    Deactivate a user account.
    """
    user = await get_user_or_404(
        db,
        user_id,
        tenant_id=tenant_id,
    )

    user.is_active = False
    user.updated_at = _utc_now()

    try:
        await db.commit()
        await db.refresh(user)

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "user_deactivation_failed",
        )

        raise DatabaseError(
            "The user could not be deactivated.",
        ) from exc

    logger.warning(
        "user_deactivated",
        extra={
            "user_id": str(user.id),
            "tenant_id": tenant_id,
        },
    )

    return user


async def mark_email_verified(
    db: AsyncSession,
    user_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> UserModel:
    """
    Mark a user's email address as verified.
    """
    user = await get_user_or_404(
        db,
        user_id,
        tenant_id=tenant_id,
    )

    user.is_verified = True
    user.email_verified_at = _utc_now()
    user.updated_at = _utc_now()

    try:
        await db.commit()
        await db.refresh(user)

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "user_email_verification_failed",
        )

        raise DatabaseError(
            "The email address could not be verified.",
        ) from exc

    logger.info(
        "user_email_verified",
        extra={
            "user_id": str(user.id),
            "tenant_id": tenant_id,
        },
    )

    return user


# ---------------------------------------------------------------------------
# Authentication telemetry
# ---------------------------------------------------------------------------


async def record_successful_login(
    db: AsyncSession,
    user_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> UserModel:
    """
    Record a successful login and reset authentication failure counters.
    """
    user = await get_user_or_404(
        db,
        user_id,
        tenant_id=tenant_id,
    )

    user.last_login_at = _utc_now()
    user.failed_login_attempts = 0
    user.locked_until = None
    user.updated_at = _utc_now()

    try:
        await db.commit()
        await db.refresh(user)

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "user_login_record_failed",
        )

        raise DatabaseError(
            "Login activity could not be recorded.",
        ) from exc

    logger.info(
        "user_login_successful",
        extra={
            "user_id": str(user.id),
            "tenant_id": tenant_id,
        },
    )

    return user


async def record_failed_login(
    db: AsyncSession,
    user_id: str | UUID,
    *,
    tenant_id: str | None = None,
    max_failed_attempts: int = 5,
    lockout_minutes: int = 15,
) -> UserModel:
    """
    Record a failed login attempt and optionally lock the account.
    """
    if max_failed_attempts < 1:
        raise ValidationError(
            "max_failed_attempts must be positive.",
        )

    if lockout_minutes < 1:
        raise ValidationError(
            "lockout_minutes must be positive.",
        )

    if (
        max_failed_attempts
        > MAX_FAILED_LOGIN_ATTEMPTS
    ):
        raise ValidationError(
            f"max_failed_attempts may not exceed "
            f"{MAX_FAILED_LOGIN_ATTEMPTS}.",
        )

    user = await get_user_or_404(
        db,
        user_id,
        tenant_id=tenant_id,
    )

    user.failed_login_attempts = (
        user.failed_login_attempts or 0
    ) + 1

    user.updated_at = _utc_now()

    if (
        user.failed_login_attempts
        >= max_failed_attempts
    ):
        user.locked_until = _utc_now() + timedelta(
            minutes=lockout_minutes,
        )

        logger.warning(
            "user_account_locked",
            extra={
                "user_id": str(user.id),
                "failed_login_attempts": (
                    user.failed_login_attempts
                ),
                "lockout_minutes": lockout_minutes,
            },
        )

    try:
        await db.commit()
        await db.refresh(user)

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "user_failed_login_record_failed",
        )

        raise DatabaseError(
            "Failed login activity could not be "
            "recorded.",
        ) from exc

    logger.warning(
        "user_login_failed",
        extra={
            "user_id": str(user.id),
            "failed_login_attempts": (
                user.failed_login_attempts
            ),
            "tenant_id": tenant_id,
        },
    )

    return user


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


async def delete_user(
    db: AsyncSession,
    user_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> None:
    """
    Permanently delete a user by ID.

    Prefer deactivation for normal offboarding. Hard deletion should be limited
    to data-subject requests, retention enforcement, or administrative cleanup.
    """
    user = await get_user_or_404(
        db,
        user_id,
        tenant_id=tenant_id,
    )

    user_id_value = str(user.id)
    masked_email = mask_email(user.email)

    try:
        await db.delete(user)
        await db.commit()

    except IntegrityError as exc:
        await db.rollback()

        logger.warning(
            "user_delete_conflict",
            extra={
                "user_id": user_id_value,
                "tenant_id": tenant_id,
            },
        )

        raise ConflictError(
            "The user cannot be deleted because "
            "related records still exist.",
        ) from exc

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "user_delete_failed",
        )

        raise DatabaseError(
            "The user could not be deleted.",
        ) from exc

    logger.warning(
        "user_deleted",
        extra={
            "user_id": user_id_value,
            "email": masked_email,
            "tenant_id": tenant_id,
        },
    )


__all__ = [
    "activate_user",
    "count_users",
    "create_user",
    "deactivate_user",
    "delete_user",
    "get_user",
    "get_user_by_email",
    "get_user_by_email_or_username",
    "get_user_by_username",
    "get_user_or_404",
    "list_users",
    "mark_email_verified",
    "record_failed_login",
    "record_successful_login",
    "update_password_hash",
    "update_user_profile",
    "update_user_role",
]