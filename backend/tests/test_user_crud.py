"""
Comprehensive tests for user CRUD normalization and validation helpers.

These tests cover email, username, full-name, role, user-ID, password-hash,
and metadata validation. They also verify security boundaries, including
sensitive-metadata rejection, whitespace handling, canonicalization, and
fail-closed validation behavior.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.core.enums import UserRole
from app.core.exceptions import ValidationError
from app.crud.user_crud import (
    _normalize_email,
    _normalize_full_name,
    _normalize_role,
    _normalize_user_id,
    _normalize_username,
    _validate_metadata,
    _validate_password_hash,
)


# ---------------------------------------------------------------------------
# Email normalization
# ---------------------------------------------------------------------------


def test_normalize_email():
    assert (
        _normalize_email(
            "  User@Example.COM "
        )
        == "user@example.com"
    )


def test_normalize_email_removes_whitespace():
    assert (
        _normalize_email(
            "  thabo@example.com  "
        )
        == "thabo@example.com"
    )


def test_normalize_email_lowercases_domain_and_local_part():
    assert (
        _normalize_email(
            "Thabo.Dev@Example.COM"
        )
        == "thabo.dev@example.com"
    )


@pytest.mark.parametrize(
    "email",
    [
        "user@example.com",
        "user.name@example.com",
        "user+tag@example.com",
        "user_name@example.co.uk",
        "user@subdomain.example.com",
    ],
)
def test_normalize_email_accepts_valid_addresses(
    email: str,
):
    normalized = _normalize_email(email)

    assert normalized == email.lower()


@pytest.mark.parametrize(
    "invalid_email",
    [
        "",
        "   ",
        "not-an-email",
        "user",
        "user@",
        "@example.com",
        "user@example",
        "user@@example.com",
        "user@example..com",
        "user name@example.com",
        "user@exa mple.com",
        "user@example.com ",
        None,
    ],
)
def test_invalid_email(invalid_email):
    with pytest.raises(ValidationError):
        _normalize_email(invalid_email)


def test_normalize_email_rejects_overly_long_email():
    long_email = (
        "a" * 250 + "@example.com"
    )

    with pytest.raises(ValidationError):
        _normalize_email(long_email)


# ---------------------------------------------------------------------------
# Username normalization
# ---------------------------------------------------------------------------


def test_normalize_username():
    assert (
        _normalize_username("  Thabo_Dev ")
        == "thabo_dev"
    )


def test_normalize_username_lowercases_value():
    assert (
        _normalize_username("ThaboDev")
        == "thabodev"
    )


def test_normalize_username_removes_whitespace():
    assert (
        _normalize_username(
            "  thabo_dev  "
        )
        == "thabo_dev"
    )


@pytest.mark.parametrize(
    "username",
    [
        "thabo_dev",
        "thabo-dev",
        "thabo.dev",
        "thabo123",
        "123thabo",
        "a" * 3,
        "a" * 30,
    ],
)
def test_normalize_username_accepts_valid_usernames(
    username: str,
):
    normalized = _normalize_username(
        username
    )

    assert normalized == username.lower()


@pytest.mark.parametrize(
    "invalid_username",
    [
        "",
        "   ",
        "invalid username",
        "user name",
        "user@name",
        "user#name",
        "user!name",
        "user$name",
        "user%name",
        "user/name",
        "user\\name",
        "user+name",
        "th",
        None,
    ],
)
def test_invalid_username(invalid_username):
    with pytest.raises(ValidationError):
        _normalize_username(
            invalid_username
        )


def test_normalize_username_rejects_overly_long_username():
    with pytest.raises(ValidationError):
        _normalize_username("a" * 65)


# ---------------------------------------------------------------------------
# Full-name normalization
# ---------------------------------------------------------------------------


def test_normalize_full_name():
    assert (
        _normalize_full_name("  Thabo M  ")
        == "Thabo M"
    )
    assert (
        _normalize_full_name(None) is None
    )


def test_normalize_full_name_preserves_internal_spaces():
    assert (
        _normalize_full_name(
            "Thabo   Mokoena"
        )
        == "Thabo Mokoena"
    )


def test_normalize_full_name_collapses_whitespace():
    assert (
        _normalize_full_name(
            "  Thabo\tMokoena\n  "
        )
        == "Thabo Mokoena"
    )


@pytest.mark.parametrize(
    "valid_name",
    [
        "Thabo",
        "Thabo Mokoena",
        "Thabo M Mokoena",
        "Mary-Jane O'Connor",
        "JosÃ© GarcÃ­a",
    ],
)
def test_normalize_full_name_accepts_valid_names(
    valid_name: str,
):
    assert (
        _normalize_full_name(valid_name)
        == valid_name.strip()
    )


@pytest.mark.parametrize(
    "invalid_name",
    [
        "",
        "   ",
        "123",
        "Thabo123",
        "Thabo!",
        "Thabo@Mokoena",
        "Thabo_Mokoena",
    ],
)
def test_normalize_full_name_rejects_invalid_names(
    invalid_name: str,
):
    with pytest.raises(ValidationError):
        _normalize_full_name(
            invalid_name
        )


# ---------------------------------------------------------------------------
# Role normalization
# ---------------------------------------------------------------------------


def test_normalize_role():
    assert (
        _normalize_role("admin")
        == UserRole.ADMIN
    )
    assert (
        _normalize_role(None)
        == UserRole.VIEWER
    )


def test_normalize_role_is_case_insensitive():
    assert (
        _normalize_role("ADMIN")
        == UserRole.ADMIN
    )
    assert (
        _normalize_role("Admin")
        == UserRole.ADMIN
    )


def test_normalize_role_removes_whitespace():
    assert (
        _normalize_role("  admin  ")
        == UserRole.ADMIN
    )


@pytest.mark.parametrize(
    "role,expected",
    [
        ("admin", UserRole.ADMIN),
        ("editor", UserRole.EDITOR),
        ("viewer", UserRole.VIEWER),
        ("ADMIN", UserRole.ADMIN),
        ("Editor", UserRole.EDITOR),
        ("VIEWER", UserRole.VIEWER),
    ],
)
def test_normalize_role_accepts_supported_roles(
    role: str,
    expected: UserRole,
):
    assert (
        _normalize_role(role)
        == expected
    )


@pytest.mark.parametrize(
    "invalid_role",
    [
        "",
        "   ",
        "unknown-role",
        "superadmin",
        "owner",
        "root",
        "administrator",
        "user",
        None if False else "123",
    ],
)
def test_invalid_role(invalid_role):
    with pytest.raises(ValidationError):
        _normalize_role(invalid_role)


# ---------------------------------------------------------------------------
# User-ID normalization
# ---------------------------------------------------------------------------


def test_normalize_user_id():
    user_id = "123e4567-e89b-12d3-a456-426614174000"

    assert _normalize_user_id(user_id)


def test_normalize_user_id_accepts_uuid_object():
    user_id = uuid4()

    assert (
        _normalize_user_id(user_id)
        == str(user_id)
    )


def test_normalize_user_id_strips_whitespace():
    user_id = str(uuid4())

    assert (
        _normalize_user_id(
            f"  {user_id}  "
        )
        == user_id
    )


def test_normalize_user_id_returns_canonical_uuid_string():
    user_id = uuid4()

    normalized = _normalize_user_id(
        str(user_id).upper()
    )

    assert normalized == str(user_id)


@pytest.mark.parametrize(
    "invalid_user_id",
    [
        "",
        "   ",
        "not-a-uuid",
        "123",
        "123e4567-e89b-12d3-a456",
        "123e4567-e89b-12d3-a456-42661417400",
        "123e4567-e89b-12d3-a456-4266141740000",
        "zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz",
        None,
    ],
)
def test_invalid_user_id(invalid_user_id):
    with pytest.raises(ValidationError):
        _normalize_user_id(
            invalid_user_id
        )


# ---------------------------------------------------------------------------
# Password-hash validation
# ---------------------------------------------------------------------------


def test_validate_password_hash():
    assert _validate_password_hash(
        "$argon2id$v=19$m=65536,t=3,p=4$validhash"
    )


@pytest.mark.parametrize(
    "valid_hash",
    [
        "$argon2id$v=19$m=65536,t=3,p=4$validhash",
        "$argon2i$v=19$m=65536,t=3,p=4$validhash",
        "$argon2d$v=19$m=65536,t=3,p=4$validhash",
        "$2b$12$validbcryptpasswordhashvalue",
        "$2y$12$validbcryptpasswordhashvalue",
    ],
)
def test_validate_password_hash_accepts_supported_hashes(
    valid_hash: str,
):
    assert (
        _validate_password_hash(
            valid_hash
        )
        is True
    )


@pytest.mark.parametrize(
    "invalid_hash",
    [
        "",
        "   ",
        "short",
        "password",
        "sha256",
        "$argon2id",
        "$2b$12",
        "$unknown$v=19$hash",
        None,
    ],
)
def test_invalid_password_hash(invalid_hash):
    with pytest.raises(ValidationError):
        _validate_password_hash(
            invalid_hash
        )


def test_validate_password_hash_rejects_plaintext_password():
    with pytest.raises(ValidationError):
        _validate_password_hash(
            "SuperSecretPassword123!"
        )


def test_validate_password_hash_rejects_whitespace_only_hash():
    with pytest.raises(ValidationError):
        _validate_password_hash(
            "                    "
        )


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------


def test_metadata_rejects_sensitive_keys():
    with pytest.raises(ValidationError):
        _validate_metadata(
            {"api_key": "secret"}
        )


@pytest.mark.parametrize(
    "sensitive_key",
    [
        "api_key",
        "apikey",
        "API_KEY",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "credential",
        "cookie",
        "private_key",
        "database_url",
        "connection_string",
        "prompt",
        "completion",
        "messages",
        "embedding",
        "vector",
        "generated_code",
        "signed_url",
        "storage_url",
        "file_url",
        "download_url",
    ],
)
def test_metadata_rejects_common_sensitive_keys(
    sensitive_key: str,
):
    with pytest.raises(ValidationError):
        _validate_metadata(
            {
                "safe": "value",
                sensitive_key: "should-not-leak",
            }
        )


def test_metadata_accepts_safe_values():
    assert (
        _validate_metadata(
            {"department": "engineering"}
        )
        == {
            "department": "engineering"
        }
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"department": "engineering"},
        {"location": "Johannesburg"},
        {"job_title": "Platform Engineer"},
        {"timezone": "Africa/Johannesburg"},
        {"preferences": {"theme": "dark"}},
    ],
)
def test_metadata_accepts_safe_metadata(
    metadata: dict,
):
    assert (
        _validate_metadata(metadata)
        == metadata
    )


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        "not-a-dictionary",
        123,
        [],
        None,
    ],
)
def test_metadata_rejects_non_dict_metadata(
    invalid_metadata,
):
    with pytest.raises(ValidationError):
        _validate_metadata(
            invalid_metadata
        )


def test_metadata_rejects_non_string_values():
    with pytest.raises(ValidationError):
        _validate_metadata(
            {"department": 123}
        )


def test_metadata_rejects_nested_sensitive_values():
    with pytest.raises(ValidationError):
        _validate_metadata(
            {
                "profile": {
                    "api_key": "should-not-leak"
                }
            }
        )


def test_metadata_rejects_overly_large_metadata():
    metadata = {
        f"key_{index}": f"value-{index}"
        for index in range(100)
    }

    with pytest.raises(ValidationError):
        _validate_metadata(metadata)


# ---------------------------------------------------------------------------
# Cross-field consistency
# ---------------------------------------------------------------------------


def test_email_and_username_normalization_are_independent():
    email = _normalize_email(
        "User@Example.COM"
    )
    username = _normalize_username(
        "  Thabo_Dev "
    )

    assert email == "user@example.com"
    assert username == "thabo_dev"


def test_role_defaults_to_viewer_when_missing():
    assert (
        _normalize_role(None)
        == UserRole.VIEWER
    )


def test_full_name_defaults_to_none_when_missing():
    assert (
        _normalize_full_name(None)
        is None
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
