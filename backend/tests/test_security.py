"""
Comprehensive tests for security primitives.

These tests cover password hashing and verification, JWT access and refresh
tokens, token claims and expiration, token tampering, weak-secret rejection,
API-key generation and verification, secure token generation, and constant-time
comparison behavior.
"""

from __future__ import annotations

import asyncio
import base64
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from app.core.exceptions import AuthenticationError
from app.core.security import (
    APIKeyService,
    InvalidTokenError,
    JWTService,
    PasswordService,
    WeakSecretError,
    api_key_service,
    constant_time_equals,
    generate_secure_token,
    password_service,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jwt_service() -> JWTService:
    """Return a JWT service configured with a strong test secret."""
    return JWTService(
        secret_key="a" * 64,
        issuer="https://auth.example.com",
        audience="agentic-bridge-mcp",
    )


@pytest.fixture
def valid_password() -> str:
    """Return a strong test password."""
    return "Correct-Horse-Battery-Staple"


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def test_password_hashing_round_trip(
    valid_password: str,
):
    hashed = password_service.hash_password(
        valid_password
    )

    assert hashed != valid_password
    assert (
        password_service.verify_password(
            valid_password,
            hashed,
        )
        is True
    )
    assert (
        password_service.verify_password(
            "wrong-password",
            hashed,
        )
        is False
    )


def test_password_hash_is_unique(
    valid_password: str,
):
    first_hash = (
        password_service.hash_password(
            valid_password
        )
    )

    second_hash = (
        password_service.hash_password(
            valid_password
        )
    )

    assert first_hash != second_hash


def test_password_hash_does_not_contain_plaintext(
    valid_password: str,
):
    hashed = password_service.hash_password(
        valid_password
    )

    assert valid_password not in hashed


def test_password_verification_rejects_empty_password(
    valid_password: str,
):
    hashed = password_service.hash_password(
        valid_password
    )

    with pytest.raises(Exception):
        password_service.verify_password(
            "",
            hashed,
        )


def test_password_verification_rejects_empty_hash(
    valid_password: str,
):
    with pytest.raises(Exception):
        password_service.verify_password(
            valid_password,
            "",
        )


def test_password_verification_rejects_invalid_hash(
    valid_password: str,
):
    with pytest.raises(Exception):
        password_service.verify_password(
            valid_password,
            "not-a-password-hash",
        )


def test_short_password_is_rejected():
    with pytest.raises(Exception):
        password_service.hash_password(
            "short"
        )


def test_empty_password_is_rejected():
    with pytest.raises(Exception):
        password_service.hash_password("")


def test_whitespace_password_is_rejected():
    with pytest.raises(Exception):
        password_service.hash_password(
            "   "
        )


def test_none_password_is_rejected():
    with pytest.raises(Exception):
        password_service.hash_password(None)


@pytest.mark.parametrize(
    "weak_password",
    [
        "password",
        "12345678",
        "abcdefgh",
        "PASSWORD",
    ],
)
def test_common_weak_passwords_are_rejected(
    weak_password: str,
):
    with pytest.raises(Exception):
        password_service.hash_password(
            weak_password
        )


# ---------------------------------------------------------------------------
# JWT access tokens
# ---------------------------------------------------------------------------


def test_jwt_access_token_round_trip(
    jwt_service: JWTService,
):
    token, jti = (
        jwt_service.create_access_token(
            subject="user_123",
            roles=["editor"],
            scopes=["workflow:execute"],
        )
    )

    claims = jwt_service.decode_token(
        token,
        expected_type="access",
    )

    assert claims["sub"] == "user_123"
    assert (
        claims["token_type"] == "access"
    )
    assert claims["roles"] == ["editor"]
    assert claims["scopes"] == [
        "workflow:execute"
    ]
    assert claims["jti"] == jti
    assert (
        claims["iss"]
        == "https://auth.example.com"
    )
    assert (
        claims["aud"]
        == "agentic-bridge-mcp"
    )


def test_jwt_access_token_contains_expiration(
    jwt_service: JWTService,
):
    token, _ = jwt_service.create_access_token(
        subject="user_123",
    )

    claims = jwt_service.decode_token(
        token,
        expected_type="access",
    )

    assert "exp" in claims
    assert claims["exp"] > time.time()


def test_jwt_access_token_contains_issued_at(
    jwt_service: JWTService,
):
    token, _ = jwt_service.create_access_token(
        subject="user_123",
    )

    claims = jwt_service.decode_token(
        token,
        expected_type="access",
    )

    assert "iat" in claims
    assert claims["iat"] <= time.time()


def test_jwt_access_token_contains_unique_jti(
    jwt_service: JWTService,
):
    first_token, first_jti = (
        jwt_service.create_access_token(
            subject="user_123",
        )
    )

    second_token, second_jti = (
        jwt_service.create_access_token(
            subject="user_123",
        )
    )

    assert first_jti != second_jti
    assert first_token != second_token


def test_jwt_access_token_rejects_empty_subject(
    jwt_service: JWTService,
):
    with pytest.raises(Exception):
        jwt_service.create_access_token(
            subject=""
        )


def test_jwt_access_token_rejects_none_subject(
    jwt_service: JWTService,
):
    with pytest.raises(Exception):
        jwt_service.create_access_token(
            subject=None
        )


# ---------------------------------------------------------------------------
# JWT refresh tokens
# ---------------------------------------------------------------------------


def test_jwt_refresh_token_has_refresh_type(
    jwt_service: JWTService,
):
    token, _ = (
        jwt_service.create_refresh_token(
            subject="user_123"
        )
    )

    claims = jwt_service.decode_token(
        token,
        expected_type="refresh",
    )

    assert (
        claims["token_type"] == "refresh"
    )
    assert claims["sub"] == "user_123"


def test_jwt_refresh_token_contains_expiration(
    jwt_service: JWTService,
):
    token, _ = (
        jwt_service.create_refresh_token(
            subject="user_123"
        )
    )

    claims = jwt_service.decode_token(
        token,
        expected_type="refresh",
    )

    assert "exp" in claims
    assert claims["exp"] > time.time()


def test_jwt_refresh_token_contains_unique_jti(
    jwt_service: JWTService,
):
    _, first_jti = (
        jwt_service.create_refresh_token(
            subject="user_123"
        )
    )

    _, second_jti = (
        jwt_service.create_refresh_token(
            subject="user_123"
        )
    )

    assert first_jti != second_jti


def test_jwt_refresh_token_rejects_empty_subject(
    jwt_service: JWTService,
):
    with pytest.raises(Exception):
        jwt_service.create_refresh_token(
            subject=""
        )


# ---------------------------------------------------------------------------
# JWT validation failures
# ---------------------------------------------------------------------------


def test_jwt_rejects_wrong_token_type(
    jwt_service: JWTService,
):
    token, _ = jwt_service.create_access_token(
        subject="user_123"
    )

    with pytest.raises(InvalidTokenError):
        jwt_service.decode_token(
            token,
            expected_type="refresh",
        )


def test_jwt_rejects_tampered_token(
    jwt_service: JWTService,
):
    token, _ = jwt_service.create_access_token(
        subject="user_123"
    )

    header, payload, signature = token.split(
        "."
    )

    decoded_payload = json.loads(
        base64.urlsafe_b64decode(
            payload + "=="
        )
    )

    decoded_payload["sub"] = "attacker"

    tampered_payload = (
        base64.urlsafe_b64encode(
            json.dumps(
                decoded_payload
            ).encode()
        )
        .decode()
        .rstrip("=")
    )

    tampered_token = (
        f"{header}.{tampered_payload}.{signature}"
    )

    with pytest.raises(InvalidTokenError):
        jwt_service.decode_token(
            tampered_token,
            expected_type="access",
        )


def test_jwt_rejects_expired_token(
    jwt_service: JWTService,
):
    token, _ = jwt_service.create_access_token(
        subject="user_123",
        expires_in=timedelta(seconds=-1),
    )

    with pytest.raises(InvalidTokenError):
        jwt_service.decode_token(
            token,
            expected_type="access",
        )


def test_jwt_rejects_invalid_token_string(
    jwt_service: JWTService,
):
    with pytest.raises(InvalidTokenError):
        jwt_service.decode_token(
            "not-a-jwt",
            expected_type="access",
        )


def test_jwt_rejects_empty_token(
    jwt_service: JWTService,
):
    with pytest.raises(InvalidTokenError):
        jwt_service.decode_token(
            "",
            expected_type="access",
        )


def test_jwt_rejects_none_token(
    jwt_service: JWTService,
):
    with pytest.raises(InvalidTokenError):
        jwt_service.decode_token(
            None,
            expected_type="access",
        )


def test_jwt_rejects_wrong_secret(
    jwt_service: JWTService,
):
    token, _ = jwt_service.create_access_token(
        subject="user_123"
    )

    other_service = JWTService(
        secret_key="b" * 64,
        issuer="https://auth.example.com",
        audience="agentic-bridge-mcp",
    )

    with pytest.raises(InvalidTokenError):
        other_service.decode_token(
            token,
            expected_type="access",
        )


def test_jwt_rejects_wrong_issuer(
    jwt_service: JWTService,
):
    token, _ = jwt_service.create_access_token(
        subject="user_123"
    )

    other_service = JWTService(
        secret_key="a" * 64,
        issuer="https://evil.example.com",
        audience="agentic-bridge-mcp",
    )

    with pytest.raises(InvalidTokenError):
        other_service.decode_token(
            token,
            expected_type="access",
        )


def test_jwt_rejects_wrong_audience(
    jwt_service: JWTService,
):
    token, _ = jwt_service.create_access_token(
        subject="user_123"
    )

    other_service = JWTService(
        secret_key="a" * 64,
        issuer="https://auth.example.com",
        audience="other-service",
    )

    with pytest.raises(InvalidTokenError):
        other_service.decode_token(
            token,
            expected_type="access",
        )


# ---------------------------------------------------------------------------
# JWT secret validation
# ---------------------------------------------------------------------------


def test_jwt_rejects_weak_secret():
    with pytest.raises(WeakSecretError):
        JWTService(
            secret_key="too-short"
        )


def test_jwt_rejects_empty_secret():
    with pytest.raises(WeakSecretError):
        JWTService(secret_key="")


def test_jwt_rejects_none_secret():
    with pytest.raises(WeakSecretError):
        JWTService(secret_key=None)


def test_jwt_rejects_low_entropy_secret():
    with pytest.raises(WeakSecretError):
        JWTService(secret_key="a" * 10)


def test_jwt_accepts_strong_secret():
    service = JWTService(
        secret_key="a" * 64
    )

    assert service is not None


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------


def test_api_key_round_trip():
    plaintext_key, key_id, key_hash = (
        api_key_service.generate_api_key()
    )

    assert plaintext_key.startswith(
        "abmcp_"
    )
    assert key_id
    assert (
        api_key_service.verify_api_key(
            plaintext_key,
            key_hash,
        )
        is True
    )
    assert (
        api_key_service.verify_api_key(
            "invalid-key",
            key_hash,
        )
        is False
    )


def test_api_key_hash_does_not_contain_plaintext():
    plaintext_key, _, key_hash = (
        api_key_service.generate_api_key()
    )

    assert plaintext_key not in key_hash


def test_api_key_generation_is_unique():
    first_key, first_key_id, _ = (
        api_key_service.generate_api_key()
    )

    second_key, second_key_id, _ = (
        api_key_service.generate_api_key()
    )

    assert first_key != second_key
    assert first_key_id != second_key_id


def test_api_key_rejects_empty_plaintext():
    _, _, key_hash = (
        api_key_service.generate_api_key()
    )

    with pytest.raises(Exception):
        api_key_service.verify_api_key(
            "",
            key_hash,
        )


def test_api_key_rejects_empty_hash():
    plaintext_key, _, _ = (
        api_key_service.generate_api_key()
    )

    with pytest.raises(Exception):
        api_key_service.verify_api_key(
            plaintext_key,
            "",
        )


def test_api_key_rejects_invalid_hash():
    plaintext_key, _, _ = (
        api_key_service.generate_api_key()
    )

    with pytest.raises(Exception):
        api_key_service.verify_api_key(
            plaintext_key,
            "not-a-key-hash",
        )


def test_api_key_service_can_be_instantiated():
    service = APIKeyService()

    assert service is not None


# ---------------------------------------------------------------------------
# Secure token generation
# ---------------------------------------------------------------------------


def test_secure_token_is_url_safe():
    token = generate_secure_token(32)

    assert len(token) >= 32
    assert "+" not in token
    assert "/" not in token


def test_secure_token_is_unique():
    first_token = generate_secure_token(32)
    second_token = generate_secure_token(32)

    assert first_token != second_token


def test_secure_token_has_expected_minimum_length():
    token = generate_secure_token(64)

    assert len(token) >= 64


def test_secure_token_rejects_invalid_length():
    with pytest.raises(Exception):
        generate_secure_token(0)


def test_secure_token_rejects_negative_length():
    with pytest.raises(Exception):
        generate_secure_token(-1)


def test_secure_token_rejects_none_length():
    with pytest.raises(Exception):
        generate_secure_token(None)


# ---------------------------------------------------------------------------
# Constant-time comparison
# ---------------------------------------------------------------------------


def test_constant_time_equals_matches():
    assert (
        constant_time_equals(
            "same-value",
            "same-value",
        )
        is True
    )


def test_constant_time_equals_rejects_mismatch():
    assert (
        constant_time_equals(
            "value-one",
            "value-two",
        )
        is False
    )


def test_constant_time_equals_rejects_empty_values():
    assert (
        constant_time_equals("", "")
        is False
    )


def test_constant_time_equals_rejects_none_values():
    assert (
        constant_time_equals(None, None)
        is False
    )


def test_constant_time_equals_rejects_different_lengths():
    assert (
        constant_time_equals(
            "short",
            "a-much-longer-value",
        )
        is False
    )


# ---------------------------------------------------------------------------
# Authentication error integration
# ---------------------------------------------------------------------------


def test_invalid_token_error_is_authentication_error():
    assert issubclass(
        InvalidTokenError,
        AuthenticationError,
    )


def test_weak_secret_error_is_not_authentication_error():
    assert not issubclass(
        WeakSecretError,
        AuthenticationError,
    )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    asyncio.run(
        pytest.main(
            [
                __file__,
                "-v",
            ]
        )
    )
