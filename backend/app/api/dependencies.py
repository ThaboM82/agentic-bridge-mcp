"""
Shared FastAPI dependencies.

This module provides database session injection, authenticated principal
resolution, pagination, tenant isolation, and scope-based authorization for
API routes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Any, Awaitable, Callable

from fastapi import (
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ValidationError,
)
from app.core.security import (
    decode_access_token,
)
from app.db.session import (
    get_db_session,
)
from app.repositories.tenant_repository import (
    TenantRepository,
)
from app.repositories.user_repository import (
    UserRepository,
)


logger = logging.getLogger(
    "agentic_bridge.api.dependencies"
)


bearer_scheme = HTTPBearer(
    auto_error=False,
    description="Bearer access token.",
)


# ---------------------------------------------------------------------------
# Database dependencies
# ---------------------------------------------------------------------------


DatabaseSession = Annotated[
    AsyncSession,
    Depends(get_db_session),
]


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pagination:
    """Normalized pagination parameters for list endpoints."""

    skip: int
    limit: int

    @property
    def page(self) -> int:
        """Return the one-based page number."""
        return (self.skip // self.limit) + 1

    @property
    def offset(self) -> int:
        """Return the zero-based result offset."""
        return self.skip


async def get_pagination(
    skip: Annotated[
        int,
        Query(
            ge=0,
            description="Number of records to skip.",
        ),
    ] = 0,
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=100,
            description="Maximum number of records to return.",
        ),
    ] = 25,
) -> Pagination:
    """Resolve and normalize pagination query parameters."""
    return Pagination(
        skip=skip,
        limit=limit,
    )


Pagination = Annotated[
    Pagination,
    Depends(get_pagination),
]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    """Authenticated request principal."""

    user_id: Any
    tenant_id: Any
    email: str | None
    roles: tuple[str, ...]
    scopes: frozenset[str]
    is_active: bool
    token_claims: dict[str, Any]

    def has_scope(
        self,
        required_scope: str,
    ) -> bool:
        """Return whether the principal has a required scope."""
        return required_scope in self.scopes

    def has_any_scope(
        self,
        required_scopes: list[str] | tuple[str, ...],
    ) -> bool:
        """Return whether the principal has at least one required scope."""
        return any(
            required_scope in self.scopes
            for required_scope in required_scopes
        )

    def has_all_scopes(
        self,
        required_scopes: list[str] | tuple[str, ...],
    ) -> bool:
        """Return whether the principal has every required scope."""
        return all(
            required_scope in self.scopes
            for required_scope in required_scopes
        )


async def get_bearer_credentials(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
) -> HTTPAuthorizationCredentials:
    """Require a bearer token in the Authorization header."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    if not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty bearer token.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    return credentials


async def get_current_principal(
    credentials: Annotated[
        HTTPAuthorizationCredentials,
        Depends(get_bearer_credentials),
    ],
    db: DatabaseSession,
) -> Principal:
    """Resolve the authenticated user and tenant from a bearer token."""
    try:
        token_payload = decode_access_token(
            credentials.credentials
        )
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        ) from exc

    user_id = token_payload.get(
        "sub"
    )
    tenant_id = token_payload.get(
        "tenant_id"
    )

    if not user_id or not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token is missing required identity claims.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    user = (
        await UserRepository.get_by_id(
            db,
            record_id=user_id,
            tenant_id=tenant_id,
        )
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    tenant = (
        await TenantRepository.get_by_id(
            db,
            record_id=tenant_id,
        )
    )

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated tenant not found.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    if not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant account is inactive.",
        )

    token_scopes = token_payload.get(
        "scope",
        ""
    )

    if isinstance(
        token_scopes,
        str
    ):
        scopes = tuple(
            scope
            for scope in token_scopes.split()
            if scope
        )
    else:
        scopes = tuple(
            token_scopes or []
        )

    roles = tuple(
        token_payload.get(
            "roles",
            []
        )
        or []
    )

    principal = Principal(
        user_id=user.id,
        tenant_id=user.tenant_id,
        email=user.email,
        roles=roles,
        scopes=frozenset(scopes),
        is_active=user.is_active,
        token_claims=token_payload,
    )

    logger.debug(
        "Authenticated principal resolved.",
        extra={
            "user_id": str(
                principal.user_id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "scopes": sorted(
                principal.scopes
            ),
        },
    )

    return principal


CurrentPrincipal = Annotated[
    Principal,
    Depends(get_current_principal),
]


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


def require_scoped_access(
    required_scope: str,
) -> Callable[
    [Principal],
    Awaitable[Principal],
]:
    """Create a dependency requiring a specific authorization scope."""

    async def dependency(
        principal: CurrentPrincipal,
    ) -> Principal:
        if not principal.has_scope(
            required_scope
        ):
            logger.warning(
                "Authorization denied.",
                extra={
                    "user_id": str(
                        principal.user_id
                    ),
                    "tenant_id": str(
                        principal.tenant_id
                    ),
                    "required_scope": required_scope,
                    "granted_scopes": sorted(
                        principal.scopes
                    ),
                },
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Missing required scope: "
                    f"{required_scope}."
                ),
            )

        return principal

    return dependency


def require_any_scope(
    required_scopes: list[str] | tuple[str, ...],
) -> Callable[
    [Principal],
    Awaitable[Principal],
]:
    """Create a dependency requiring at least one of several scopes."""

    async def dependency(
        principal: CurrentPrincipal,
    ) -> Principal:
        if not principal.has_any_scope(
            required_scopes
        ):
            logger.warning(
                "Authorization denied.",
                extra={
                    "user_id": str(
                        principal.user_id
                    ),
                    "tenant_id": str(
                        principal.tenant_id
                    ),
                    "required_scopes": list(
                        required_scopes
                    ),
                    "granted_scopes": sorted(
                        principal.scopes
                    ),
                },
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Missing one of required scopes: "
                    f"{', '.join(required_scopes)}."
                ),
            )

        return principal

    return dependency


def require_all_scopes(
    required_scopes: list[str] | tuple[str, ...],
) -> Callable[
    [Principal],
    Awaitable[Principal],
]:
    """Create a dependency requiring every listed scope."""

    async def dependency(
        principal: CurrentPrincipal,
    ) -> Principal:
        if not principal.has_all_scopes(
            required_scopes
        ):
            logger.warning(
                "Authorization denied.",
                extra={
                    "user_id": str(
                        principal.user_id
                    ),
                    "tenant_id": str(
                        principal.tenant_id
                    ),
                    "required_scopes": list(
                        required_scopes
                    ),
                    "granted_scopes": sorted(
                        principal.scopes
                    ),
                },
            )

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Missing required scopes: "
                    f"{', '.join(required_scopes)}."
                ),
            )

        return principal

    return dependency


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------


def require_tenant_access(
    tenant_id: Any,
    principal: Principal,
) -> None:
    """Ensure a principal can only access resources in its own tenant."""
    if principal.tenant_id != tenant_id:
        logger.warning(
            "Cross-tenant access denied.",
            extra={
                "user_id": str(
                    principal.user_id
                ),
                "principal_tenant_id": str(
                    principal.tenant_id
                ),
                "requested_tenant_id": str(
                    tenant_id
                ),
            },
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to this tenant resource is forbidden.",
        )


# ---------------------------------------------------------------------------
# Request context
# ---------------------------------------------------------------------------


async def get_request_context(
    request: Request,
) -> dict[str, Any]:
    """Extract common request metadata for logging and auditing."""
    return {
        "method": request.method,
        "path": request.url.path,
        "query_params": dict(
            request.query_params
        ),
        "client_host": (
            request.client.host
            if request.client
            else None
        ),
        "user_agent": request.headers.get(
            "user-agent"
        ),
        "request_id": request.headers.get(
            "x-request-id"
        ),
    }


RequestContext = Annotated[
    dict[str, Any],
    Depends(get_request_context),
]