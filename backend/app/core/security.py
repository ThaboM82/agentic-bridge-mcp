"""
Security primitives for AgenticBridge-MCP.

This module contains only low-level security utilities. Authentication and
authorization policies belong in `app/core/auth.py`; HTTP concerns belong in
routers and dependencies.

Security rules:
- Passwords are hashed with Argon2id and never stored, logged, or returned.
- JWT algorithms are configured server-side and never accepted from tokens.
- JWTs include issuer, audience, issuance time, expiry, and a unique `jti`.
- API keys are returned once in plaintext and stored only as hashes.
- Token and API-key comparisons use constant-time comparison where applicable.
- Weak cryptographic configuration fails closed.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import uuid
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import Any, Literal

import jwt
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import (
    Argon2Hasher,
)

from app.config import settings
from app.core.exceptions import (
    AuthenticationError,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SecurityError(Exception):
    """Base class for security-related failures."""


class InvalidCredentialsError(
    SecurityError,
):
    """Raised when supplied credentials cannot be verified."""


class InvalidTokenError(
    SecurityError,
    AuthenticationError,
):
    """
    Raised when a token is malformed, expired, or fails validation.

    This inherits from `AuthenticationError` so API and MCP authentication
    boundaries can treat invalid tokens as authentication failures.
    """


class TokenRevokedError(
    InvalidTokenError,
):
    """Raised when a token's `jti` has been revoked."""


class WeakSecretError(
    SecurityError,
):
    """Raised when a configured cryptographic secret is unsafe."""


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


class PasswordService:
    """
    Hashes and verifies user passwords using Argon2id.

    Argon2 is a memory-hard password hashing algorithm and is the recommended
    default for new applications. Never store, log, or return plaintext
    passwords.
    """

    MIN_PASSWORD_LENGTH = 12
    MAX_PASSWORD_LENGTH = 256

    def __init__(self) -> None:
        self._hasher = PasswordHash(
            [
                Argon2Hasher(),
            ]
        )

    def hash_password(
        self,
        password: str | None,
    ) -> str:
        """
        Hash a plaintext password for database storage.
        """
        self._validate_password(
            password,
        )

        return self._hasher.hash(
            password,
        )

    def verify_password(
        self,
        password: str | None,
        password_hash: str | None,
    ) -> bool:
        """
        Verify a plaintext password against a stored hash.
        """
        if not password or not password_hash:
            return False

        if not isinstance(
            password,
            str,
        ):
            return False

        if not isinstance(
            password_hash,
            str,
        ):
            return False

        try:
            return self._hasher.verify(
                password,
                password_hash,
            )

        except Exception:
            logger.warning(
                "Password verification failed.",
            )

            return False

    def needs_rehash(
        self,
        password_hash: str | None,
    ) -> bool:
        """
        Return whether an existing hash should be upgraded.
        """
        if (
            not password_hash
            or not isinstance(
                password_hash,
                str,
            )
        ):
            return False

        try:
            return (
                self._hasher.needs_rehash(
                    password_hash,
                )
            )

        except Exception:
            return False

    def _validate_password(
        self,
        password: str | None,
    ) -> None:
        """
        Validate password presence, type, and length.
        """
        if not isinstance(
            password,
            str,
        ):
            raise InvalidCredentialsError(
                "Password must be a string.",
            )

        if (
            len(password)
            < self.MIN_PASSWORD_LENGTH
        ):
            raise InvalidCredentialsError(
                "Password must contain at least "
                f"{self.MIN_PASSWORD_LENGTH} characters.",
            )

        if (
            len(password)
            > self.MAX_PASSWORD_LENGTH
        ):
            raise InvalidCredentialsError(
                "Password must contain no more than "
                f"{self.MAX_PASSWORD_LENGTH} characters.",
            )


password_service = PasswordService()


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------


TokenType = Literal[
    "access",
    "refresh",
    "service",
]


class JWTService:
    """
    Creates and validates signed JWTs.
    """

    REQUIRED_CLAIMS = {
        "sub",
        "exp",
        "iat",
        "iss",
        "aud",
        "jti",
    }

    MIN_HS_SECRET_BYTES = 32

    def __init__(
        self,
        secret_key: str | None = None,
        algorithm: str | None = None,
        issuer: str | None = None,
        audience: str | None = None,
        access_token_expire_minutes: int | None = None,
        refresh_token_expire_days: int = 14,
        leeway_seconds: int | None = None,
    ) -> None:
        """
        Initialize the JWT service.
        """
        self.secret_key = secret_key or (
            settings.JWT_SECRET_KEY.get_secret_value()
            if settings.JWT_SECRET_KEY
            else None
        )

        self.algorithm = (
            algorithm
            or settings.JWT_ALGORITHM
        )

        self.issuer = (
            issuer
            or settings.JWT_ISSUER
        )

        self.audience = (
            audience
            or settings.JWT_AUDIENCE
        )

        self.access_token_expire_minutes = (
            access_token_expire_minutes
            or settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )

        self.refresh_token_expire_days = (
            refresh_token_expire_days
        )

        self.leeway_seconds = (
            leeway_seconds
            if leeway_seconds is not None
            else settings.JWT_LEEWAY_SECONDS
        )

        self._validate_configuration()

    def create_access_token(
        self,
        subject: str | None,
        roles: list[str] | None = None,
        scopes: list[str] | None = None,
        additional_claims: dict[
            str,
            Any,
        ]
        | None = None,
        expires_in_minutes: int | None = None,
        expires_in: timedelta | None = None,
    ) -> tuple[str, str]:
        """
        Create a short-lived access token.
        """
        if expires_in is not None:
            expires_delta = expires_in

        else:
            expires_delta = timedelta(
                minutes=(
                    expires_in_minutes
                    or self.access_token_expire_minutes
                ),
            )

        return self._create_token(
            subject=subject,
            token_type="access",
            expires_delta=expires_delta,
            roles=roles,
            scopes=scopes,
            additional_claims=additional_claims,
        )

    def create_refresh_token(
        self,
        subject: str | None,
        roles: list[str] | None = None,
        scopes: list[str] | None = None,
        additional_claims: dict[
            str,
            Any,
        ]
        | None = None,
        expires_in_days: int | None = None,
        expires_in: timedelta | None = None,
    ) -> tuple[str, str]:
        """
        Create a longer-lived refresh token.
        """
        if expires_in is not None:
            expires_delta = expires_in

        else:
            expires_delta = timedelta(
                days=(
                    expires_in_days
                    or self.refresh_token_expire_days
                ),
            )

        return self._create_token(
            subject=subject,
            token_type="refresh",
            expires_delta=expires_delta,
            roles=roles,
            scopes=scopes,
            additional_claims=additional_claims,
        )

    def create_service_token(
        self,
        subject: str | None,
        scopes: list[str] | None = None,
        expires_in_minutes: int = 15,
        expires_in: timedelta | None = None,
    ) -> tuple[str, str]:
        """
        Create a short-lived token for server-to-server communication.
        """
        if expires_in is not None:
            expires_delta = expires_in

        else:
            expires_delta = timedelta(
                minutes=expires_in_minutes,
            )

        return self._create_token(
            subject=subject,
            token_type="service",
            expires_delta=expires_delta,
            roles=["service"],
            scopes=scopes,
        )

    def decode_token(
        self,
        token: str | None,
        expected_type: TokenType | None = None,
        revoked_jti_checker: Any = None,
    ) -> dict[str, Any]:
        """
        Decode and validate a JWT.
        """
        if not token or not isinstance(
            token,
            str,
        ):
            raise InvalidTokenError(
                "Token is missing.",
            )

        try:
            claims = jwt.decode(
                token,
                key=self.secret_key,
                algorithms=[
                    self.algorithm,
                ],
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.leeway_seconds,
                options={
                    "require": sorted(
                        self.REQUIRED_CLAIMS,
                    ),
                },
            )

        except jwt.ExpiredSignatureError as exc:
            raise InvalidTokenError(
                "Token has expired.",
            ) from exc

        except jwt.InvalidAudienceError as exc:
            raise InvalidTokenError(
                "Token audience is invalid.",
            ) from exc

        except jwt.InvalidIssuerError as exc:
            raise InvalidTokenError(
                "Token issuer is invalid.",
            ) from exc

        except jwt.PyJWTError as exc:
            logger.warning(
                "JWT validation failed: %s",
                exc,
            )

            raise InvalidTokenError(
                "Token is invalid.",
            ) from exc

        token_type = claims.get(
            "token_type",
        )

        if (
            expected_type
            and token_type != expected_type
        ):
            raise InvalidTokenError(
                "Token type mismatch; expected "
                f"'{expected_type}'.",
            )

        jti = claims.get("jti")

        if revoked_jti_checker:
            revoked = revoked_jti_checker(
                jti,
            )

            if hasattr(
                revoked,
                "__await__",
            ):
                raise InvalidTokenError(
                    "Async revocation checker must "
                    "be awaited by the caller.",
                )

            if revoked:
                raise TokenRevokedError(
                    "Token has been revoked.",
                )

        return claims

    def _create_token(
        self,
        subject: str | None,
        token_type: TokenType,
        expires_delta: timedelta,
        roles: list[str] | None,
        scopes: list[str] | None,
        additional_claims: dict[
            str,
            Any,
        ]
        | None,
    ) -> tuple[str, str]:
        """
        Create and sign a JWT with standard security claims.
        """
        if not subject or not isinstance(
            subject,
            str,
        ):
            raise InvalidTokenError(
                "Token subject cannot be empty.",
            )

        if expires_delta.total_seconds() <= 0:
            raise InvalidTokenError(
                "Token lifetime must be positive.",
            )

        now = datetime.now(
            timezone.utc,
        )

        expires_at = now + expires_delta

        jti = str(uuid.uuid4())

        claims: dict[str, Any] = {
            "sub": subject,
            "token_type": token_type,
            "roles": roles or [],
            "scopes": scopes or [],
            "iat": now,
            "exp": expires_at,
            "iss": self.issuer,
            "aud": self.audience,
            "jti": jti,
        }

        if additional_claims:
            claims.update(
                additional_claims,
            )

        token = jwt.encode(
            claims,
            key=self.secret_key,
            algorithm=self.algorithm,
        )

        return token, jti

    def _validate_configuration(
        self,
    ) -> None:
        """
        Validate JWT cryptographic configuration.
        """
        if (
            not self.secret_key
            or not isinstance(
                self.secret_key,
                str,
            )
        ):
            raise WeakSecretError(
                "JWT secret key is not configured.",
            )

        if self.algorithm.startswith(
            "HS",
        ) and len(
            self.secret_key.encode(
                "utf-8",
            ),
        ) < self.MIN_HS_SECRET_BYTES:
            raise WeakSecretError(
                "HS JWT secret keys must contain at least "
                f"{self.MIN_HS_SECRET_BYTES} bytes.",
            )


jwt_service = JWTService()


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------


class APIKeyService:
    """
    Creates and verifies hashed API keys.
    """

    KEY_PREFIX = "abmcp"
    KEY_BYTES = 32

    def generate_api_key(
        self,
    ) -> tuple[str, str, str]:
        """
        Generate a new API key.
        """
        raw_key = secrets.token_urlsafe(
            self.KEY_BYTES,
        )

        plaintext_key = (
            f"{self.KEY_PREFIX}_{raw_key}"
        )

        key_id = str(uuid.uuid4())

        key_hash = self.hash_api_key(
            plaintext_key,
        )

        return (
            plaintext_key,
            key_id,
            key_hash,
        )

    def hash_api_key(
        self,
        api_key: str | None,
    ) -> str:
        """
        Hash an API key using SHA-256.
        """
        if not api_key or not isinstance(
            api_key,
            str,
        ):
            raise InvalidCredentialsError(
                "API key cannot be empty.",
            )

        return hashlib.sha256(
            api_key.encode(
                "utf-8",
            ),
        ).hexdigest()

    def verify_api_key(
        self,
        provided_key: str | None,
        stored_hash: str | None,
    ) -> bool:
        """
        Verify an API key in constant time.
        """
        if (
            not provided_key
            or not stored_hash
        ):
            return False

        if not isinstance(
            provided_key,
            str,
        ) or not isinstance(
            stored_hash,
            str,
        ):
            return False

        try:
            provided_hash = (
                self.hash_api_key(
                    provided_key,
                )
            )

        except InvalidCredentialsError:
            return False

        return hmac.compare_digest(
            provided_hash,
            stored_hash,
        )


api_key_service = APIKeyService()


# ---------------------------------------------------------------------------
# General cryptographic helpers
# ---------------------------------------------------------------------------


def generate_secure_token(
    byte_length: int | None = 32,
) -> str:
    """
    Generate a URL-safe, cryptographically secure random token.
    """
    if not isinstance(
        byte_length,
        int,
    ):
        raise ValueError(
            "Token byte length must be an integer.",
        )

    if byte_length < 16:
        raise ValueError(
            "Token byte length must be at least 16.",
        )

    return secrets.token_urlsafe(
        byte_length,
    )


def generate_numeric_confirmation_code(
    length: int | None = 6,
) -> str:
    """
    Generate a cryptographically secure numeric confirmation code.
    """
    if not isinstance(
        length,
        int,
    ):
        raise ValueError(
            "Confirmation code length must be an integer.",
        )

    if not 4 <= length <= 10:
        raise ValueError(
            "Confirmation code length must be "
            "between 4 and 10.",
        )

    return "".join(
        str(secrets.randbelow(10))
        for _ in range(length)
    )


def constant_time_equals(
    left: str | None,
    right: str | None,
) -> bool:
    """
    Compare two strings without leaking timing information.
    """
    if not isinstance(
        left,
        str,
    ) or not isinstance(
        right,
        str,
    ):
        return False

    if not left or not right:
        return False

    return hmac.compare_digest(
        left.encode(
            "utf-8",
        ),
        right.encode(
            "utf-8",
        ),
    )


__all__ = [
    "APIKeyService",
    "InvalidCredentialsError",
    "InvalidTokenError",
    "JWTService",
    "PasswordService",
    "SecurityError",
    "TokenRevokedError",
    "TokenType",
    "WeakSecretError",
    "api_key_service",
    "constant_time_equals",
    "generate_numeric_confirmation_code",
    "generate_secure_token",
    "jwt_service",
    "password_service",
]