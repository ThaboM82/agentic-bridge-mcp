import logging
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings


logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(
    auto_error=False,
    description="Paste a JWT access token using the Bearer scheme.",
)


class AuthError(Exception):
    """Base authentication or authorization error."""


class InvalidTokenError(AuthError):
    """Raised when a bearer token is missing, malformed, or invalid."""


class InsufficientPermissionsError(AuthError):
    """Raised when an authenticated user lacks a required role or scope."""


class CurrentUser:
    """
    Normalized authenticated identity.

    `raw_claims` retains provider-specific claims, but application code should
    rely on the typed fields wherever possible.
    """

    def __init__(
        self,
        subject: str,
        email: str | None,
        roles: list[str],
        scopes: list[str],
        raw_claims: dict[str, Any],
    ) -> None:
        self.id = subject
        self.subject = subject
        self.email = email
        self.roles = roles
        self.scopes = scopes
        self.raw_claims = raw_claims

    def has_role(self, *required_roles: str) -> bool:
        return any(role in self.roles for role in required_roles)

    def has_scope(self, *required_scopes: str) -> bool:
        return any(scope in self.scopes for scope in required_scopes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "roles": self.roles,
            "scopes": self.scopes,
        }


def _development_user(
    credentials: HTTPAuthorizationCredentials | None,
) -> CurrentUser:
    """
    Development identity used only when authentication is explicitly disabled.

    Never enable this outside local development.
    """
    if credentials and credentials.credentials:
        logger.warning(
            "Development authentication is enabled; bearer token was not validated."
        )

    return CurrentUser(
        subject="development-user",
        email="developer@localhost",
        roles=["admin", "editor", "reviewer"],
        scopes=[
            "workflow:read",
            "workflow:execute",
            "documents:read",
            "documents:write",
        ],
        raw_claims={"auth_mode": "development"},
    )


def _decode_jwt(token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT access token.

    Uses a fixed algorithm allowlist and requires expiration, issuer, and
    audience claims.
    """
    try:
        return jwt.decode(
            token,
            key=settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={
                "require": ["exp", "iss", "aud", "sub"],
            },
            leeway=settings.jwt_leeway_seconds,
        )
    except jwt.ExpiredSignatureError as exc:
        raise InvalidTokenError("Access token has expired.") from exc
    except jwt.InvalidAudienceError as exc:
        raise InvalidTokenError("Access token audience is invalid.") from exc
    except jwt.InvalidIssuerError as exc:
        raise InvalidTokenError("Access token issuer is invalid.") from exc
    except jwt.PyJWTError as exc:
        logger.warning("JWT validation failed: %s", exc)
        raise InvalidTokenError("Access token is invalid.") from exc


def _extract_roles_and_scopes(claims: dict[str, Any]) -> tuple[list[str], list[str]]:
    """
    Extract roles and scopes from common JWT claim formats.

    Supports:
    - roles: ["admin", "editor"]
    - realm_access.roles: ["admin"]  # Keycloak-style
    - scope: "workflow:execute documents:read"
    - scopes: ["workflow:execute"]
    """
    roles: list[str] = claims.get("roles") or []

    realm_access = claims.get("realm_access") or {}
    if isinstance(realm_access, dict):
        roles.extend(realm_access.get("roles") or [])

    scopes: list[str] = claims.get("scopes") or []

    raw_scope = claims.get("scope")
    if isinstance(raw_scope, str):
        scopes.extend(raw_scope.split())

    return list(dict.fromkeys(roles)), list(dict.fromkeys(scopes))


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
) -> CurrentUser:
    """
    Authenticate the caller using a Bearer JWT.

    In development mode, returns a local development identity without
    validating the token.
    """
    if settings.auth_mode.lower() == "development":
        return _development_user(credentials)

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    claims = _decode_jwt(credentials.credentials)

    subject = claims.get("sub")
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token is missing a subject claim.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    roles, scopes = _extract_roles_and_scopes(claims)

    return CurrentUser(
        subject=subject,
        email=claims.get("email"),
        roles=roles,
        scopes=scopes,
        raw_claims=claims,
    )


CurrentUserDependency = Annotated[CurrentUser, Depends(get_current_user)]


def require_roles(*required_roles: str):
    """
    Dependency factory for role-based authorization.

    Example:
        dependencies=[Depends(require_roles("admin", "publisher"))]
    """

    async def checker(
        current_user: CurrentUserDependency,
    ) -> CurrentUser:
        if not current_user.has_role(*required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to perform this action.",
            )
        return current_user

    return checker


def require_scopes(*required_scopes: str):
    """
    Dependency factory for scope-based authorization.

    Example:
        dependencies=[Depends(require_scopes("workflow:execute"))]
    """

    async def checker(
        current_user: CurrentUserDependency,
    ) -> CurrentUser:
        if not current_user.has_scope(*required_scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient scopes for this operation.",
            )
        return current_user

    return checker