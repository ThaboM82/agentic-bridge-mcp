"""
User administration API routes.

These routes provide tenant-scoped user registration, profile retrieval,
listing, updates, role management, activation status, password changes, and
safe account deletion. Administrative operations require explicit scopes and
are recorded in the tenant audit trail.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)
from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    field_validator,
)

from app.api.dependencies import (
    CurrentPrincipal,
    DatabaseSession,
    Pagination,
    require_scoped_access,
)
from app.api.pagination import (
    build_paginated_response,
)
from app.api.schemas import (
    UserResponse,
    UserRoleUpdateRequest,
    UserStatusUpdateRequest,
)
from app.core.exceptions import ValidationError
from app.core.security import (
    hash_password,
    verify_password,
)
from app.repositories.audit_log_repository import (
    AuditLogRepository,
)
from app.repositories.user_repository import (
    UserRepository,
)


logger = logging.getLogger(
    "agentic_bridge.api.users"
)


router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class UserCreateRequest(BaseModel):
    """Request to create a tenant user."""

    email: EmailStr
    full_name: str | None = Field(
        default=None,
        max_length=200,
    )
    password: str = Field(
        min_length=12,
        max_length=128,
        description="Initial account password.",
    )
    role: str = Field(
        default="member",
        max_length=100,
    )

    @field_validator(
        "role"
    )
    @classmethod
    def validate_role(
        cls,
        value: str,
    ) -> str:
        """Normalize and validate the requested role."""
        normalized_role = (
            value.strip().lower()
        )

        allowed_roles = {
            "owner",
            "admin",
            "member",
            "viewer",
        }

        if normalized_role not in allowed_roles:
            raise ValueError(
                "Role must be one of: "
                + ", ".join(
                    sorted(
                        allowed_roles
                    )
                )
            )

        return normalized_role


class UserUpdateRequest(BaseModel):
    """Request to update a user profile."""

    full_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    email: EmailStr | None = None


class PasswordChangeRequest(BaseModel):
    """Request to change the authenticated user's password."""

    current_password: str = Field(
        min_length=1,
        max_length=128,
    )
    new_password: str = Field(
        min_length=12,
        max_length=128,
    )


# ---------------------------------------------------------------------------
# User collection
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=None,
    summary="List users",
    description=(
        "List tenant-scoped users with optional filtering. Requires the "
        "`users:read` scope."
    ),
)
async def list_users(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "users:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
    role: Annotated[
        str | None,
        Query(
            max_length=100,
            description="Filter by user role.",
        ),
    ] = None,
    is_active: Annotated[
        bool | None,
        Query(
            description="Filter by activation status.",
        ),
    ] = None,
    search: Annotated[
        str | None,
        Query(
            max_length=200,
            description="Search user email or full name.",
        ),
    ] = None,
):
    """List tenant-scoped users."""
    users, total_count = (
        await UserRepository.list_for_tenant(
            db,
            tenant_id=principal.tenant_id,
            skip=pagination.skip,
            limit=pagination.limit,
        )
    )

    if role:
        normalized_role = (
            role.strip().lower()
        )

        users = [
            user
            for user in users
            if user.role
            == normalized_role
        ]

    if is_active is not None:
        users = [
            user
            for user in users
            if user.is_active
            == is_active
        ]

    if search:
        normalized_search = (
            search.strip().lower()
        )

        users = [
            user
            for user in users
            if normalized_search
            in user.email.lower()
            or (
                user.full_name
                and normalized_search
                in user.full_name.lower()
            )
        ]

    await db.commit()

    return build_paginated_response(
        users,
        skip=pagination.skip,
        limit=pagination.limit,
        total_count=total_count,
    )


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create user",
    description=(
        "Create a new user within the authenticated tenant. Requires the "
        "`users:write` scope."
    ),
)
async def create_user(
    payload: UserCreateRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "users:write"
            )
        ),
    ],
    db: DatabaseSession,
) -> UserResponse:
    """Create a tenant-scoped user."""
    normalized_email = (
        payload.email.lower()
    )

    existing_user = (
        await UserRepository.get_by_email(
            db,
            tenant_id=principal.tenant_id,
            email=normalized_email,
        )
    )

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists.",
        )

    try:
        user = (
            await UserRepository.create_user(
                db,
                tenant_id=principal.tenant_id,
                email=normalized_email,
                full_name=payload.full_name,
                hashed_password=hash_password(
                    payload.password
                ),
                role=payload.role,
            )
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="user_created",
        actor_id=principal.user_id,
        resource_type="user",
        resource_id=user.id,
        action="create_user",
        outcome="created",
        metadata={
            "email": user.email,
            "role": user.role,
        },
    )

    await db.commit()

    logger.info(
        "User created.",
        extra={
            "user_id": str(
                user.id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
            "role": user.role,
        },
    )

    return user


# ---------------------------------------------------------------------------
# Individual users
# ---------------------------------------------------------------------------


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description=(
        "Retrieve the authenticated user's profile."
    ),
)
async def get_current_user_profile(
    principal: CurrentPrincipal,
    db: DatabaseSession,
) -> UserResponse:
    """Retrieve the authenticated user's profile."""
    user = (
        await UserRepository.get_by_id(
            db,
            record_id=principal.user_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Authenticated user not found.",
        )

    await db.commit()

    return user


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user",
    description=(
        "Retrieve a tenant-scoped user by ID. Requires the `users:read` "
        "scope."
    ),
)
async def get_user(
    user_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "users:read"
            )
        ),
    ],
    db: DatabaseSession,
) -> UserResponse:
    """Retrieve a tenant-scoped user."""
    user = (
        await UserRepository.get_by_id(
            db,
            record_id=user_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    await db.commit()

    return user


@router.patch(
    "/{user_id}",
    response_model=UserResponse,
    summary="Update user",
    description=(
        "Update a tenant-scoped user's profile. Requires the `users:write` "
        "scope."
    ),
)
async def update_user(
    user_id: UUID,
    payload: UserUpdateRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "users:write"
            )
        ),
    ],
    db: DatabaseSession,
) -> UserResponse:
    """Update a tenant-scoped user's profile."""
    existing_user = (
        await UserRepository.get_by_id(
            db,
            record_id=user_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not existing_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    update_data = payload.model_dump(
        exclude_unset=True
    )

    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one field must be provided.",
        )

    if "email" in update_data:
        normalized_email = (
            update_data["email"].lower()
        )

        conflicting_user = (
            await UserRepository.get_by_email(
                db,
                tenant_id=principal.tenant_id,
                email=normalized_email,
            )
        )

        if conflicting_user and conflicting_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A user with this email already exists.",
            )

        update_data[
            "email"
        ] = normalized_email

    try:
        user = (
            await UserRepository.update_user(
                db,
                user_id=user_id,
                tenant_id=principal.tenant_id,
                update_data=update_data,
            )
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="user_updated",
        actor_id=principal.user_id,
        resource_type="user",
        resource_id=user.id,
        action="update_user",
        outcome="updated",
        metadata={
            "updated_fields": list(
                update_data.keys()
            ),
        },
    )

    await db.commit()

    logger.info(
        "User updated.",
        extra={
            "user_id": str(
                user.id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
            "updated_fields": list(
                update_data.keys()
            ),
        },
    )

    return user


@router.delete(
    "/{user_id}",
    response_model=dict,
    summary="Delete user",
    description=(
        "Delete a tenant-scoped user. Requires the `users:admin` scope."
    ),
)
async def delete_user(
    user_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "users:admin"
            )
        ),
    ],
    db: DatabaseSession,
) -> dict:
    """Delete a tenant-scoped user."""
    if user_id == principal.user_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Users cannot delete their own account.",
        )

    existing_user = (
        await UserRepository.get_by_id(
            db,
            record_id=user_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not existing_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    deleted = (
        await UserRepository.delete_by_id(
            db,
            record_id=user_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="user_deleted",
        actor_id=principal.user_id,
        resource_type="user",
        resource_id=user_id,
        action="delete_user",
        outcome="deleted",
        metadata={
            "email": existing_user.email,
        },
    )

    await db.commit()

    logger.info(
        "User deleted.",
        extra={
            "user_id": str(
                user_id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
        },
    )

    return {
        "user_id": str(
            user_id
        ),
        "deleted": True,
    }


# ---------------------------------------------------------------------------
# Role and status administration
# ---------------------------------------------------------------------------


@router.post(
    "/{user_id}/role",
    response_model=UserResponse,
    summary="Update user role",
    description=(
        "Update a tenant-scoped user's role. Requires the `users:admin` "
        "scope."
    ),
)
async def update_user_role(
    user_id: UUID,
    payload: UserRoleUpdateRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "users:admin"
            )
        ),
    ],
    db: DatabaseSession,
) -> UserResponse:
    """Update a tenant-scoped user's role."""
    user = (
        await UserRepository.set_user_role(
            db,
            user_id=user_id,
            tenant_id=principal.tenant_id,
            role=payload.role,
        )
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="user_role_changed",
        actor_id=principal.user_id,
        resource_type="user",
        resource_id=user.id,
        action="update_user_role",
        outcome=user.role,
        metadata={
            "role": user.role,
        },
    )

    await db.commit()

    logger.info(
        "User role updated.",
        extra={
            "user_id": str(
                user.id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
            "role": user.role,
        },
    )

    return user


@router.post(
    "/{user_id}/status",
    response_model=UserResponse,
    summary="Update user status",
    description=(
        "Activate or deactivate a tenant-scoped user. Requires the "
        "`users:admin` scope."
    ),
)
async def update_user_status(
    user_id: UUID,
    payload: UserStatusUpdateRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "users:admin"
            )
        ),
    ],
    db: DatabaseSession,
) -> UserResponse:
    """Activate or deactivate a tenant-scoped user."""
    if user_id == principal.user_id and not payload.is_active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Users cannot deactivate their own account.",
        )

    user = (
        await UserRepository.set_user_status(
            db,
            user_id=user_id,
            tenant_id=principal.tenant_id,
            is_active=payload.is_active,
        )
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="user_status_changed",
        actor_id=principal.user_id,
        resource_type="user",
        resource_id=user.id,
        action="update_user_status",
        outcome="active"
        if user.is_active
        else "inactive",
        metadata={
            "is_active": user.is_active,
        },
    )

    await db.commit()

    logger.info(
        "User status updated.",
        extra={
            "user_id": str(
                user.id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
            "is_active": user.is_active,
        },
    )

    return user


# ---------------------------------------------------------------------------
# Password management
# ---------------------------------------------------------------------------


@router.post(
    "/me/password",
    response_model=dict,
    summary="Change current password",
    description=(
        "Change the authenticated user's password after verifying the "
        "current password."
    ),
)
async def change_current_user_password(
    payload: PasswordChangeRequest,
    principal: CurrentPrincipal,
    db: DatabaseSession,
) -> dict:
    """Change the authenticated user's password."""
    user = (
        await UserRepository.get_by_id(
            db,
            record_id=principal.user_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Authenticated user not found.",
        )

    if not verify_password(
        payload.current_password,
        user.hashed_password,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect.",
        )

    await UserRepository.update_user(
        db,
        user_id=user.id,
        tenant_id=principal.tenant_id,
        update_data={
            "hashed_password": hash_password(
                payload.new_password
            ),
        },
    )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="user_password_changed",
        actor_id=principal.user_id,
        resource_type="user",
        resource_id=user.id,
        action="change_password",
        outcome="completed",
        metadata={},
    )

    await db.commit()

    logger.info(
        "User password changed.",
        extra={
            "user_id": str(
                user.id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
        },
    )

    return {
        "user_id": str(
            user.id
        ),
        "password_changed": True,
    }