"""
Small, reusable, side-effect-free helpers.

These utilities must remain deterministic, dependency-light, and safe to use
from API routes, LangGraph nodes, background jobs, and tests. Never place
database access, network calls, filesystem access, or global mutable state in
this module.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import unicodedata
from datetime import date, datetime, timezone
from typing import Any, Iterable, TypeVar
from uuid import UUID


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

SLUG_INVALID_CHARACTERS = re.compile(r"[^a-z0-9]+")
MULTIPLE_DASHES = re.compile(r"-+")

EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
)

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

WHITESPACE_PATTERN = re.compile(r"\s+")

CAMEL_CASE_PATTERN = re.compile(r"(?<!^)(?=[A-Z])")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def slugify(value: str, *, max_length: int = 80) -> str:
    """
    Convert text into a URL-safe slug.

    Args:
        value: Input text.
        max_length: Maximum slug length.

    Returns:
        A lowercase, hyphenated slug. Returns `item` when no usable characters
        remain.
    """
    if not isinstance(value, str):
        raise TypeError("Value must be a string.")

    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")

    slug = SLUG_INVALID_CHARACTERS.sub("-", ascii_value.lower()).strip("-")
    slug = MULTIPLE_DASHES.sub("-", slug)

    return slug[:max_length].rstrip("-") or "item"


def truncate_text(
    value: str,
    max_length: int,
    *,
    suffix: str = "...",
) -> str:
    """
    Safely truncate text for display, logging, or storage.

    Args:
        value: Text to truncate.
        max_length: Maximum returned length, including the suffix.
        suffix: Text appended when truncation occurs.

    Returns:
        The original text when it fits, otherwise truncated text plus suffix.
    """
    if max_length <= 0:
        raise ValueError("max_length must be greater than zero.")

    if len(suffix) >= max_length:
        raise ValueError("suffix must be shorter than max_length.")

    if not isinstance(value, str):
        value = str(value)

    if len(value) <= max_length:
        return value

    return value[: max_length - len(suffix)] + suffix


def normalize_whitespace(value: str) -> str:
    """Collapse repeated whitespace and trim leading/trailing spaces."""
    if not isinstance(value, str):
        raise TypeError("Value must be a string.")

    return WHITESPACE_PATTERN.sub(" ", value).strip()


def remove_control_characters(value: str) -> str:
    """Remove non-printable control characters from text."""
    if not isinstance(value, str):
        raise TypeError("Value must be a string.")

    return CONTROL_CHARACTERS.sub("", value)


def camel_to_snake(value: str) -> str:
    """Convert a CamelCase or PascalCase string to snake_case."""
    if not isinstance(value, str):
        raise TypeError("Value must be a string.")

    return CAMEL_CASE_PATTERN.sub("_", value).lower()


def snake_to_camel(value: str) -> str:
    """Convert a snake_case string to camelCase."""
    if not isinstance(value, str):
        raise TypeError("Value must be a string.")

    parts = value.split("_")
    first, *remaining = parts

    return first + "".join(part.capitalize() for part in remaining)


def pluralize(
    count: int,
    singular: str,
    plural: str | None = None,
) -> str:
    """Return the singular or plural form of a word based on count."""
    if count == 1:
        return singular

    return plural or f"{singular}s"


def chunk_list(items: list[T], size: int) -> list[list[T]]:
    """
    Split a list into fixed-size chunks.

    Args:
        items: Items to split.
        size: Maximum number of items per chunk.

    Returns:
        A list of chunks. The final chunk may contain fewer than `size` items.
    """
    if size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    return [items[index : index + size] for index in range(0, len(items), size)]


def deduplicate(items: Iterable[T]) -> list[T]:
    """Remove duplicate values while preserving original order."""
    seen: set[Any] = set()
    result: list[T] = []

    for item in items:
        try:
            key = item
        except TypeError:
            key = repr(item)

        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def is_valid_email(value: str) -> bool:
    """Perform conservative email-format validation."""
    if not isinstance(value, str) or len(value) > 254:
        return False

    return bool(EMAIL_PATTERN.fullmatch(value))


def is_valid_identifier(value: str, *, max_length: int = 64) -> bool:
    """Validate an internal identifier containing letters, numbers, hyphens, and underscores."""
    if not isinstance(value, str) or not value or len(value) > max_length:
        return False

    return bool(IDENTIFIER_PATTERN.fullmatch(value))


def is_valid_uuid(value: str) -> bool:
    """Validate whether a string is a UUID."""
    if not isinstance(value, str):
        return False

    try:
        UUID(value)
    except ValueError:
        return False

    return True


def ensure_non_empty(value: str | None, field_name: str) -> str:
    """Ensure a string value is non-empty after trimming."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")

    return value.strip()


def ensure_max_length(
    value: str,
    max_length: int,
    field_name: str,
) -> str:
    """Ensure a string value does not exceed a maximum length."""
    if len(value) > max_length:
        raise ValueError(
            f"{field_name} exceeds the maximum allowed length of {max_length}."
        )

    return value


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------


def redact_secret(
    value: str | None,
    *,
    visible_characters: int = 4,
) -> str:
    """
    Redact a secret for safe logging.

    Never log full API keys, passwords, tokens, or connection strings.
    """
    if not value:
        return ""

    if visible_characters < 0:
        raise ValueError("visible_characters cannot be negative.")

    if len(value) <= visible_characters:
        return "*" * len(value)

    return f"{value[:visible_characters]}...{value[-visible_characters:]}"


def mask_email(value: str | None) -> str:
    """
    Mask an email address for logs or user-facing diagnostics.

    Example:
        `user@example.com` becomes `u***@example.com`.
    """
    if not value or "@" not in value:
        return ""

    local_part, domain = value.split("@", 1)

    if not local_part:
        return f"***@{domain}"

    return f"{local_part[0]}***@{domain}"


def generate_secure_token(byte_length: int = 32) -> str:
    """Generate a cryptographically secure URL-safe token."""
    if byte_length <= 0:
        raise ValueError("byte_length must be greater than zero.")

    return secrets.token_urlsafe(byte_length)


def constant_time_equals(left: str, right: str) -> bool:
    """Compare two strings in constant time to reduce timing attacks."""
    return hmac.compare_digest(
        left.encode("utf-8"),
        right.encode("utf-8"),
    )


def hash_text(value: str, *, algorithm: str = "sha256") -> str:
    """
    Generate a deterministic hash for non-secret identifiers or cache keys.

    Do not use this function to store passwords; use a password-specific
    hashing algorithm such as Argon2 or bcrypt.
    """
    if not isinstance(value, str):
        raise TypeError("Value must be a string.")

    return hashlib.new(algorithm, value.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Date and time helpers
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_timestamp() -> int:
    """Return the current Unix timestamp in seconds."""
    return int(utc_now().timestamp())


def iso_utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return utc_now().isoformat()


def ensure_utc(value: datetime) -> datetime:
    """
    Ensure a datetime is timezone-aware and expressed in UTC.

    Naive datetimes are assumed to represent UTC.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def parse_iso_datetime(value: str) -> datetime:
    """Parse an ISO-8601 datetime string and normalize it to UTC."""
    if not isinstance(value, str):
        raise TypeError("Value must be a string.")

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))

    return ensure_utc(parsed)


def format_date(value: date) -> str:
    """Format a date using an unambiguous ISO-8601 representation."""
    return value.isoformat()


def is_expired(
    expires_at: datetime,
    *,
    now: datetime | None = None,
) -> bool:
    """Determine whether a timestamp has expired."""
    current_time = ensure_utc(now) if now else utc_now()

    return ensure_utc(expires_at) <= current_time


# ---------------------------------------------------------------------------
# Collection and dictionary helpers
# ---------------------------------------------------------------------------


def safe_get(
    mapping: dict[str, Any] | None,
    key: str,
    default: Any = None,
) -> Any:
    """Safely retrieve a value from a possibly missing dictionary."""
    if not isinstance(mapping, dict):
        return default

    return mapping.get(key, default)


def deep_merge(
    base: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Values from `updates` override values from `base`. Nested dictionaries are
    merged recursively; all other values are replaced.
    """
    merged = dict(base)

    for key, value in updates.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def first_non_empty(*values: str | None) -> str:
    """Return the first non-empty string value."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""