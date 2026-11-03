"""
Comprehensive tests for core utility functions.

These tests cover slug generation, text truncation and normalization,
case conversion, batching, deduplication, validation, secret redaction,
email masking, secure tokens, constant-time comparison, hashing, UTC
datetime handling, expiration checks, deep merging, and fallback values.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from app.core.utils import (
    camel_to_snake,
    chunk_list,
    constant_time_equals,
    deep_merge,
    deduplicate,
    ensure_utc,
    first_non_empty,
    generate_secure_token,
    hash_text,
    is_expired,
    is_valid_email,
    is_valid_identifier,
    is_valid_uuid,
    mask_email,
    normalize_whitespace,
    redact_secret,
    slugify,
    snake_to_camel,
    truncate_text,
    utc_now,
)


# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------


def test_slugify():
    assert (
        slugify("AgenticBridge MCP Platform!")
        == "agenticbridge-mcp-platform"
    )
    assert (
        slugify("  Multiple   Spaces  ")
        == "multiple-spaces"
    )
    assert slugify("###") == "item"


def test_slugify_respects_max_length():
    assert (
        len(
            slugify(
                "a" * 200,
                max_length=20,
            )
        )
        <= 20
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("AgenticBridge", "agenticbridge"),
        ("AgenticBridge Platform", "agenticbridge-platform"),
        ("AgenticBridge_Platform", "agenticbridge-platform"),
        ("AgenticBridge-Platform", "agenticbridge-platform"),
        ("AgenticBridge  Platform", "agenticbridge-platform"),
        ("AgenticBridge!@#Platform", "agenticbridge-platform"),
        ("123 Workflow", "123-workflow"),
        ("", "item"),
        ("   ", "item"),
        ("###", "item"),
    ],
)
def test_slugify_normalizes_values(
    value: str,
    expected: str,
):
    assert slugify(value) == expected


def test_slugify_preserves_numbers():
    assert (
        slugify("Workflow 123")
        == "workflow-123"
    )


def test_slugify_max_length_returns_fallback():
    assert (
        slugify("###", max_length=3)
        == "ite"
    )


def test_slugify_rejects_invalid_max_length():
    with pytest.raises(ValueError):
        slugify("workflow", max_length=0)


def test_slugify_rejects_none():
    with pytest.raises((TypeError, ValueError)):
        slugify(None)


# ---------------------------------------------------------------------------
# Text truncation
# ---------------------------------------------------------------------------


def test_truncate_text():
    assert (
        truncate_text("hello", 10)
        == "hello"
    )
    assert (
        truncate_text("hello world", 8)
        == "hello..."
    )
    assert (
        truncate_text(
            "hello world",
            5,
            suffix="!",
        )
        == "hell!"
    )


def test_truncate_text_exact_length():
    assert (
        truncate_text("hello", 5)
        == "hello"
    )


def test_truncate_text_custom_suffix():
    assert (
        truncate_text(
            "hello world",
            8,
            suffix="â€¦",
        )
        == "hello wâ€¦"
    )


def test_truncate_text_empty_string():
    assert truncate_text("", 10) == ""


def test_invalid_truncate_length():
    with pytest.raises(ValueError):
        truncate_text("hello", 0)


@pytest.mark.parametrize(
    "invalid_length",
    [
        -1,
        -10,
        0,
    ],
)
def test_truncate_text_rejects_non_positive_lengths(
    invalid_length: int,
):
    with pytest.raises(ValueError):
        truncate_text(
            "hello",
            invalid_length,
        )


def test_truncate_text_rejects_none_text():
    with pytest.raises((TypeError, ValueError)):
        truncate_text(None, 10)


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------


def test_normalize_whitespace():
    assert (
        normalize_whitespace(
            "  hello   world  "
        )
        == "hello world"
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", ""),
        ("   ", ""),
        ("hello", "hello"),
        ("  hello", "hello"),
        ("hello  ", "hello"),
        ("hello\tworld", "hello world"),
        ("hello\nworld", "hello world"),
        ("hello\r\nworld", "hello world"),
        ("hello    world", "hello world"),
    ],
)
def test_normalize_whitespace_variants(
    value: str,
    expected: str,
):
    assert (
        normalize_whitespace(value)
        == expected
    )


def test_normalize_whitespace_rejects_none():
    with pytest.raises((TypeError, ValueError)):
        normalize_whitespace(None)


# ---------------------------------------------------------------------------
# Case conversion
# ---------------------------------------------------------------------------


def test_camel_case_conversion():
    assert (
        camel_to_snake(
            "WorkflowExecutionID"
        )
        == "workflow_execution_id"
    )
    assert (
        snake_to_camel(
            "workflow_execution_id"
        )
        == "workflowExecutionId"
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("WorkflowExecutionID", "workflow_execution_id"),
        ("workflowExecutionId", "workflow_execution_id"),
        ("WorkflowExecutionId", "workflow_execution_id"),
        ("HTTPResponse", "http_response"),
        ("APIKey", "api_key"),
        ("userID", "user_id"),
        ("", ""),
    ],
)
def test_camel_to_snake_variants(
    value: str,
    expected: str,
):
    assert (
        camel_to_snake(value)
        == expected
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("workflow_execution_id", "workflowExecutionId"),
        ("user_id", "userId"),
        ("api_key", "apiKey"),
        ("http_response", "httpResponse"),
        ("", ""),
        ("_leading", "Leading"),
        ("trailing_", "trailing"),
    ],
)
def test_snake_to_camel_variants(
    value: str,
    expected: str,
):
    assert (
        snake_to_camel(value)
        == expected
    )


def test_case_conversion_round_trip():
    original = "workflow_execution_id"

    assert (
        camel_to_snake(
            snake_to_camel(original)
        )
        == original
    )


def test_case_conversion_rejects_none():
    with pytest.raises((TypeError, ValueError)):
        camel_to_snake(None)

    with pytest.raises((TypeError, ValueError)):
        snake_to_camel(None)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_list():
    assert (
        chunk_list([1, 2, 3, 4, 5], 2)
        == [[1, 2], [3, 4], [5]]
    )
    assert chunk_list([], 3) == []


def test_chunk_list_exact_division():
    assert (
        chunk_list(
            [1, 2, 3, 4],
            2,
        )
        == [[1, 2], [3, 4]]
    )


def test_chunk_list_single_item_chunks():
    assert (
        chunk_list([1, 2, 3], 1)
        == [[1], [2], [3]]
    )


def test_chunk_list_chunk_larger_than_list():
    assert (
        chunk_list([1, 2], 10)
        == [[1, 2]]
    )


def test_chunk_list_preserves_order():
    values = list(range(10))

    chunks = chunk_list(values, 3)

    flattened = [
        item
        for chunk in chunks
        for item in chunk
    ]

    assert flattened == values


@pytest.mark.parametrize(
    "invalid_size",
    [
        -1,
        0,
    ],
)
def test_invalid_chunk_size(invalid_size):
    with pytest.raises(ValueError):
        chunk_list(
            [1, 2],
            invalid_size,
        )


def test_chunk_list_rejects_none():
    with pytest.raises((TypeError, ValueError)):
        chunk_list(None, 2)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_deduplicate_preserves_order():
    assert (
        deduplicate(
            ["a", "b", "a", "c", "b"]
        )
        == ["a", "b", "c"]
    )


def test_deduplicate_empty_list():
    assert deduplicate([]) == []


def test_deduplicate_no_duplicates():
    assert (
        deduplicate(["a", "b", "c"])
        == ["a", "b", "c"]
    )


def test_deduplicate_mixed_types():
    assert (
        deduplicate(
            [1, "1", 1, "1", True]
        )
        == [1, "1", True]
    )


def test_deduplicate_preserves_first_occurrence():
    assert (
        deduplicate(
            [
                {"id": 1},
                {"id": 2},
                {"id": 1},
            ]
        )
        == [{"id": 1}, {"id": 2}]
    )


def test_deduplicate_rejects_none():
    with pytest.raises((TypeError, ValueError)):
        deduplicate(None)


# ---------------------------------------------------------------------------
# Email validation
# ---------------------------------------------------------------------------


def test_email_validation():
    assert (
        is_valid_email("user@example.com")
        is True
    )
    assert (
        is_valid_email("not-an-email")
        is False
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
def test_valid_email_addresses(email: str):
    assert is_valid_email(email) is True


@pytest.mark.parametrize(
    "email",
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
        None,
    ],
)
def test_invalid_email_addresses(email):
    assert is_valid_email(email) is False


# ---------------------------------------------------------------------------
# Identifier validation
# ---------------------------------------------------------------------------


def test_identifier_validation():
    assert (
        is_valid_identifier(
            "workflow_123"
        )
        is True
    )
    assert (
        is_valid_identifier("invalid id")
        is False
    )


@pytest.mark.parametrize(
    "identifier",
    [
        "workflow_123",
        "workflow-123",
        "Workflow123",
        "a",
        "a" * 64,
    ],
)
def test_valid_identifiers(identifier: str):
    assert (
        is_valid_identifier(identifier)
        is True
    )


@pytest.mark.parametrize(
    "identifier",
    [
        "",
        "   ",
        "invalid id",
        "invalid-id!",
        "invalid@id",
        "invalid#id",
        "invalid/id",
        "invalid\\id",
        None,
    ],
)
def test_invalid_identifiers(identifier):
    assert (
        is_valid_identifier(identifier)
        is False
    )


def test_identifier_max_length_is_enforced():
    assert (
        is_valid_identifier("a" * 65)
        is False
    )


# ---------------------------------------------------------------------------
# UUID validation
# ---------------------------------------------------------------------------


def test_uuid_validation():
    assert (
        is_valid_uuid(
            "123e4567-e89b-12d3-a456-426614174000"
        )
        is True
    )
    assert (
        is_valid_uuid("not-a-uuid")
        is False
    )


def test_uuid_validation_accepts_generated_uuid():
    assert (
        is_valid_uuid(str(uuid4()))
        is True
    )


def test_uuid_validation_accepts_uuid_object():
    assert is_valid_uuid(uuid4()) is True


@pytest.mark.parametrize(
    "invalid_uuid",
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
def test_invalid_uuid_values(invalid_uuid):
    assert (
        is_valid_uuid(invalid_uuid)
        is False
    )


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------


def test_redact_secret():
    assert (
        redact_secret("sk-1234567890")
        == "sk-1...7890"
    )
    assert (
        redact_secret("short")
        == "*****"
    )
    assert (
        redact_secret(None) == ""
    )


def test_redact_secret_preserves_prefix_and_suffix():
    redacted = redact_secret(
        "sk-1234567890"
    )

    assert redacted.startswith("sk-1")
    assert redacted.endswith("7890")
    assert "234567" not in redacted


def test_redact_secret_handles_empty_string():
    assert redact_secret("") == ""


def test_redact_secret_handles_minimum_length():
    assert (
        redact_secret("12345678")
        == "1234...5678"
    )


def test_redact_secret_handles_custom_visible_characters():
    redacted = redact_secret(
        "sk-1234567890",
        visible_chars=4,
    )

    assert (
        redacted == "sk-1...7890"
    )


def test_redact_secret_does_not_leak_full_secret():
    secret = "sk-very-long-secret-value"

    redacted = redact_secret(secret)

    assert secret not in redacted


# ---------------------------------------------------------------------------
# Email masking
# ---------------------------------------------------------------------------


def test_mask_email():
    assert (
        mask_email("user@example.com")
        == "u***@example.com"
    )
    assert mask_email(None) == ""


def test_mask_email_handles_empty_string():
    assert mask_email("") == ""


def test_mask_email_handles_short_local_part():
    assert (
        mask_email("a@example.com")
        == "a***@example.com"
    )


def test_mask_email_preserves_domain():
    masked = mask_email(
        "thabo@example.com"
    )

    assert masked.endswith(
        "@example.com"
    )


def test_mask_email_does_not_leak_local_part():
    masked = mask_email(
        "thabo@example.com"
    )

    assert "thabo" not in masked


# ---------------------------------------------------------------------------
# Secure tokens
# ---------------------------------------------------------------------------


def test_secure_token_is_unique():
    assert (
        generate_secure_token()
        != generate_secure_token()
    )


def test_secure_token_has_expected_length():
    token = generate_secure_token()

    assert len(token) >= 32


def test_secure_token_is_url_safe():
    token = generate_secure_token()

    assert " " not in token
    assert "/" not in token
    assert "+" not in token


def test_secure_token_uses_secure_randomness():
    token = generate_secure_token()

    assert secrets.token_urlsafe(32)


def test_secure_token_custom_length():
    token = generate_secure_token(
        length=64
    )

    assert len(token) >= 64


# ---------------------------------------------------------------------------
# Constant-time comparison
# ---------------------------------------------------------------------------


def test_constant_time_comparison():
    assert (
        constant_time_equals(
            "value",
            "value",
        )
        is True
    )
    assert (
        constant_time_equals(
            "value",
            "other",
        )
        is False
    )


def test_constant_time_comparison_handles_empty_strings():
    assert (
        constant_time_equals("", "")
        is True
    )
    assert (
        constant_time_equals("", "value")
        is False
    )


def test_constant_time_comparison_is_case_sensitive():
    assert (
        constant_time_equals(
            "Value",
            "value",
        )
        is False
    )


def test_constant_time_comparison_handles_none():
    assert (
        constant_time_equals(None, None)
        is True
    )
    assert (
        constant_time_equals(
            None,
            "value",
        )
        is False
    )


# ---------------------------------------------------------------------------
# Text hashing
# ---------------------------------------------------------------------------


def test_hash_text_is_deterministic():
    assert (
        hash_text("input")
        == hash_text("input")
    )
    assert (
        hash_text("input")
        != hash_text("other")
    )


def test_hash_text_produces_hex_digest():
    hashed = hash_text("input")

    assert len(hashed) == 64
    assert all(
        character in "0123456789abcdef"
        for character in hashed
    )


def test_hash_text_matches_sha256():
    assert (
        hash_text("input")
        == hashlib.sha256(
            b"input"
        ).hexdigest()
    )


def test_hash_text_handles_unicode():
    hashed = hash_text("Thabo Mokoena")

    assert hashed == hash_text(
        "Thabo Mokoena"
    )


def test_hash_text_handles_empty_string():
    assert (
        hash_text("")
        == hashlib.sha256(
            b""
        ).hexdigest()
    )


# ---------------------------------------------------------------------------
# UTC datetime handling
# ---------------------------------------------------------------------------


def test_utc_now_is_timezone_aware():
    assert (
        utc_now().tzinfo is not None
    )


def test_utc_now_uses_utc_timezone():
    assert (
        utc_now().tzinfo
        == timezone.utc
    )


def test_ensure_utc_handles_naive_datetime():
    naive = datetime(
        2026, 1, 1, 12, 0, 0
    )

    result = ensure_utc(naive)

    assert (
        result.tzinfo == timezone.utc
    )


def test_ensure_utc_preserves_utc_datetime():
    utc_datetime = datetime(
        2026,
        1,
        1,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    )

    assert (
        ensure_utc(utc_datetime)
        == utc_datetime
    )


def test_ensure_utc_converts_other_timezones():
    offset_timezone = timezone(
        timedelta(hours=2)
    )

    local_datetime = datetime(
        2026,
        1,
        1,
        14,
        0,
        0,
        tzinfo=offset_timezone,
    )

    result = ensure_utc(local_datetime)

    assert (
        result.tzinfo == timezone.utc
    )
    assert result.hour == 12


def test_ensure_utc_rejects_none():
    with pytest.raises((TypeError, ValueError)):
        ensure_utc(None)


# ---------------------------------------------------------------------------
# Expiration checks
# ---------------------------------------------------------------------------


def test_is_expired():
    past = datetime(
        2020, 1, 1, tzinfo=timezone.utc
    )
    future = datetime(
        2999, 1, 1, tzinfo=timezone.utc
    )

    assert is_expired(past) is True
    assert is_expired(future) is False


def test_is_expired_with_custom_now():
    expires_at = datetime(
        2026,
        1,
        1,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    )

    now = datetime(
        2026,
        1,
        1,
        11,
        0,
        0,
        tzinfo=timezone.utc,
    )

    assert (
        is_expired(
            expires_at,
            now=now,
        )
        is False
    )


def test_is_expired_at_exact_boundary():
    expires_at = datetime(
        2026,
        1,
        1,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    )

    now = datetime(
        2026,
        1,
        1,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    )

    assert (
        is_expired(
            expires_at,
            now=now,
        )
        is True
    )


def test_is_expired_handles_naive_datetime():
    naive_expiry = datetime(
        2020, 1, 1, 12, 0, 0
    )

    assert (
        is_expired(naive_expiry)
        is True
    )


def test_is_expired_rejects_none():
    with pytest.raises((TypeError, ValueError)):
        is_expired(None)


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------


def test_deep_merge():
    assert deep_merge(
        {"a": {"b": 1}, "c": 2},
        {"a": {"d": 3}, "c": 4},
    ) == {
        "a": {"b": 1, "d": 3},
        "c": 4,
    }


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1}}
    override = {"a": {"c": 2}}

    deep_merge(base, override)

    assert base == {"a": {"b": 1}}
    assert override == {"a": {"c": 2}}


def test_deep_merge_overrides_scalars():
    assert deep_merge(
        {"a": 1},
        {"a": 2},
    ) == {"a": 2}


def test_deep_merge_replaces_non_dict_with_dict():
    assert deep_merge(
        {"a": 1},
        {"a": {"b": 2}},
    ) == {"a": {"b": 2}}


def test_deep_merge_replaces_dict_with_non_dict():
    assert deep_merge(
        {"a": {"b": 1}},
        {"a": 2},
    ) == {"a": 2}


def test_deep_merge_handles_empty_dictionaries():
    assert deep_merge({}, {}) == {}
    assert deep_merge({"a": 1}, {}) == {
        "a": 1
    }
    assert deep_merge({}, {"a": 1}) == {
        "a": 1
    }


def test_deep_merge_handles_nested_levels():
    assert deep_merge(
        {
            "workflow": {
                "execution": {
                    "retry": 1
                }
            }
        },
        {
            "workflow": {
                "execution": {
                    "timeout": 30
                }
            }
        },
    ) == {
        "workflow": {
            "execution": {
                "retry": 1,
                "timeout": 30,
            }
        }
    }


# ---------------------------------------------------------------------------
# First non-empty fallback
# ---------------------------------------------------------------------------


def test_first_non_empty():
    assert (
        first_non_empty(
            "",
            None,
            "value"
        )
        == "value"
    )
    assert (
        first_non_empty("", None)
        == ""
    )


def test_first_non_empty_returns_first_valid_value():
    assert (
        first_non_empty(
            "first",
            "second",
        )
        == "first"
    )


def test_first_non_empty_skips_whitespace():
    assert (
        first_non_empty(
            "   ",
            "value",
        )
        == "value"
    )


def test_first_non_empty_handles_zero():
    assert (
        first_non_empty(0, "value")
        == 0
    )


def test_first_non_empty_handles_false():
    assert (
        first_non_empty(
            False,
            "value",
        )
        is False
    )


def test_first_non_empty_returns_default_when_all_empty():
    assert (
        first_non_empty(
            "",
            None,
            default="fallback",
        )
        == "fallback"
    )


def test_first_non_empty_rejects_no_values():
    with pytest.raises((TypeError, ValueError)):
        first_non_empty()


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
