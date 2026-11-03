"""
MCP error mapping, result sanitization, and safe tool boundary.

This module converts application exceptions into stable, client-safe MCP error
responses. It also provides the decorator used by every MCP tool to prevent
unexpected failures from leaking internal details.

Security rules:
- Never expose stack traces, provider payloads, database URLs, credentials,
  prompts, completions, generated code, embeddings, or customer content.
- Treat tool arguments and results as untrusted input.
- Redact sensitive-looking keys and values.
- Bound all strings, lists, and metadata maps.
- Log only stable error codes, tool names, and correlation IDs.
- Tool failures must never expose internal implementation details.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from app.core.exceptions import (
    AppError,
    AuthenticationError,
    AuthorizationError,
    ExternalProviderError,
    NotFoundError,
    ProviderTimeoutError,
    RateLimitError,
    ValidationError,
    WorkflowStateError,
)
from app.core.utils import truncate_text


logger = logging.getLogger("agentic_bridge.mcp_server.errors")

P = ParamSpec("P")
R = TypeVar("R")

MAX_ERROR_MESSAGE_CHARS = 1_000
MAX_TOOL_RESULT_CHARS = 20_000
MAX_TOOL_ARGUMENT_CHARS = 20_000
MAX_METADATA_KEYS = 50
MAX_METADATA_VALUE_CHARS = 1_000
MAX_LIST_ITEMS = 50
MAX_ERROR_CODE_CHARS = 100

DEFAULT_ERROR_MESSAGE = (
    "The requested MCP operation could not be completed."
)

UNEXPECTED_ERROR_MESSAGE = (
    "The requested MCP operation failed due to an unexpected server error."
)

PROVIDER_TIMEOUT_MESSAGE = (
    "The requested operation timed out while communicating with an external "
    "provider."
)

PROVIDER_UNAVAILABLE_MESSAGE = (
    "An external provider is temporarily unavailable."
)

REDACTED_VALUE = "[REDACTED]"
UNSERIALIZABLE_VALUE = "[UNSERIALIZABLE]"

SENSITIVE_KEY_FRAGMENTS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "cookie",
    "session",
    "private_key",
    "access_key",
    "refresh_token",
    "signed_url",
    "storage_url",
    "file_url",
    "download_url",
    "prompt",
    "completion",
    "messages",
    "input",
    "output",
    "embedding",
    "vector",
    "generated_code",
    "database_url",
    "connection_string",
    "stack_trace",
    "traceback",
)

SENSITIVE_ARGUMENT_KEYS: frozenset[str] = frozenset(
    fragment
    for fragment in SENSITIVE_KEY_FRAGMENTS
)

SECRET_VALUE_PATTERNS = None  # Populated lazily to avoid import cost at module import.


# ---------------------------------------------------------------------------
# Exception classification
# ---------------------------------------------------------------------------


def get_error_status_code(
    exc: BaseException,
) -> int:
    """
    Map an exception to an HTTP-compatible status code.

    Args:
        exc: Application or unexpected exception.

    Returns:
        A stable HTTP status code.
    """
    if isinstance(exc, ValidationError):
        return 422

    if isinstance(exc, AuthenticationError):
        return 401

    if isinstance(exc, AuthorizationError):
        return 403

    if isinstance(exc, NotFoundError):
        return 404

    if isinstance(exc, RateLimitError):
        return 429

    if isinstance(exc, ProviderTimeoutError):
        return 504

    if isinstance(exc, ExternalProviderError):
        return 502

    if isinstance(exc, (AppError, WorkflowStateError)):
        return 400

    return 500


def get_error_code(
    exc: BaseException,
) -> str:
    """
    Map an exception to a stable machine-readable error code.

    Args:
        exc: Application or unexpected exception.

    Returns:
        A stable error code.

    Raises:
        ValidationError: If an application error exposes an invalid code.
    """
    if isinstance(exc, ValidationError):
        return "validation_error"

    if isinstance(exc, AuthenticationError):
        return "authentication_error"

    if isinstance(exc, AuthorizationError):
        return "authorization_error"

    if isinstance(exc, NotFoundError):
        return "not_found"

    if isinstance(exc, RateLimitError):
        return "rate_limit_exceeded"

    if isinstance(exc, ProviderTimeoutError):
        return "provider_timeout"

    if isinstance(exc, ExternalProviderError):
        return "external_provider_error"

    if isinstance(exc, WorkflowStateError):
        return "workflow_error"

    if isinstance(exc, AppError):
        error_code = getattr(
            exc,
            "code",
            None,
        )

        if not error_code:
            return "workflow_error"

        normalized_code = str(error_code).strip().lower()

        if not normalized_code:
            return "workflow_error"

        return truncate_text(
            normalized_code,
            MAX_ERROR_CODE_CHARS,
        )

    return "internal_error"


def get_safe_error_message(
    exc: BaseException,
) -> str:
    """
    Return a client-safe error message.

    Args:
        exc: Application or unexpected exception.

    Returns:
        A bounded, non-sensitive error message.
    """
    if isinstance(exc, ProviderTimeoutError):
        return PROVIDER_TIMEOUT_MESSAGE

    if isinstance(exc, ExternalProviderError):
        return PROVIDER_UNAVAILABLE_MESSAGE

    if isinstance(
        exc,
        (
            ValidationError,
            AuthenticationError,
            AuthorizationError,
            NotFoundError,
            RateLimitError,
        ),
    ):
        message = str(exc).strip()

        if message:
            return truncate_text(
                message,
                MAX_ERROR_MESSAGE_CHARS,
            )

    if isinstance(exc, (AppError, WorkflowStateError)):
        return DEFAULT_ERROR_MESSAGE

    return UNEXPECTED_ERROR_MESSAGE


def is_retryable_error(
    exc: BaseException,
) -> bool:
    """
    Return whether an error may be retried automatically.

    Args:
        exc: Application or unexpected exception.

    Returns:
        `True` for transient provider, timeout, or rate-limit failures.
    """
    return isinstance(
        exc,
        (
            ProviderTimeoutError,
            ExternalProviderError,
            RateLimitError,
        ),
    )


def get_retry_after_seconds(
    exc: BaseException,
) -> int | None:
    """
    Return a safe retry delay for rate-limit and provider failures.

    Args:
        exc: Application or unexpected exception.

    Returns:
        Retry delay in seconds, or `None`.
    """
    if isinstance(exc, RateLimitError):
        retry_after = getattr(
            exc,
            "retry_after_seconds",
            None,
        )

        if isinstance(retry_after, int) and retry_after > 0:
            return min(retry_after, 3_600)

        return 60

    if isinstance(
        exc,
        (
            ProviderTimeoutError,
            ExternalProviderError,
        ),
    ):
        return 30

    return None


# ---------------------------------------------------------------------------
# Result and metadata sanitization
# ---------------------------------------------------------------------------


def _redact_sensitive_value(
    value: str,
) -> str:
    """Redact obvious credential-like strings."""
    normalized_value = value

    replacements = (
        (
            "postgresql://",
            "postgresql://[REDACTED]@",
        ),
        (
            "postgresql+asyncpg://",
            "postgresql+asyncpg://[REDACTED]@",
        ),
        (
            "redis://",
            "redis://[REDACTED]@",
        ),
    )

    for pattern, replacement in replacements:
        if pattern in normalized_value and "@" in normalized_value:
            scheme, remainder = normalized_value.split("://", 1)

            if "@" in remainder:
                _, host_part = remainder.split("@", 1)
                normalized_value = (
                    f"{scheme}://{REDACTED_VALUE}@{host_part}"
                )

    return normalized_value


def sanitize_error_message(
    message: str | None,
) -> str:
    """
    Sanitize and bound an error message.

    Args:
        message: Raw error message.

    Returns:
        A bounded, redacted error message.
    """
    if not message or not isinstance(message, str):
        return UNEXPECTED_ERROR_MESSAGE

    normalized_message = message.strip()

    if not normalized_message:
        return UNEXPECTED_ERROR_MESSAGE

    return truncate_text(
        _redact_sensitive_value(normalized_message),
        MAX_ERROR_MESSAGE_CHARS,
    )


def sanitize_tool_value(
    value: Any,
    *,
    max_length: int = MAX_METADATA_VALUE_CHARS,
) -> Any:
    """
    Sanitize and bound an MCP tool result value.

    Args:
        value: Arbitrary result value.
        max_length: Maximum string length.

    Returns:
        A JSON-safe, bounded representation of the value.
    """
    if value is None or isinstance(
        value,
        (bool, int, float),
    ):
        return value

    if isinstance(value, str):
        return truncate_text(
            _redact_sensitive_value(value),
            max_length,
        )

    if isinstance(value, dict):
        return sanitize_tool_metadata(value)

    if isinstance(value, (list, tuple, set)):
        return [
            sanitize_tool_value(
                item,
                max_length=min(
                    max_length,
                    MAX_TOOL_RESULT_CHARS,
                ),
            )
            for item in list(value)[:MAX_LIST_ITEMS]
        ]

    return UNSERIALIZABLE_VALUE


def sanitize_tool_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Sanitize and bound MCP metadata or result mappings.

    Args:
        metadata: Arbitrary metadata supplied by a tool or service.

    Returns:
        A bounded metadata dictionary with sensitive values redacted.
    """
    if not metadata or not isinstance(metadata, dict):
        return {}

    safe_metadata: dict[str, Any] = {}

    for key, value in list(metadata.items())[:MAX_METADATA_KEYS]:
        if not isinstance(key, str):
            continue

        normalized_key = key.strip()

        if not normalized_key:
            continue

        key_lower = normalized_key.lower()

        if any(
            fragment in key_lower
            for fragment in SENSITIVE_KEY_FRAGMENTS
        ):
            safe_metadata[normalized_key] = REDACTED_VALUE
            continue

        safe_metadata[normalized_key] = sanitize_tool_value(
            value
        )

    return safe_metadata


def sanitize_tool_result(
    result: Any,
) -> Any:
    """
    Sanitize and bound an MCP tool result before returning it to a client.

    Args:
        result: Value returned by a workflow service, runner, or tool.

    Returns:
        A JSON-safe, bounded result.
    """
    return sanitize_tool_value(
        result,
        max_length=MAX_TOOL_RESULT_CHARS,
    )


def sanitize_tool_arguments(
    arguments: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Sanitize and bound MCP tool arguments for safe diagnostics.

    Args:
        arguments: Tool arguments supplied by an MCP client.

    Returns:
        A bounded argument summary with sensitive values redacted.
    """
    if not arguments or not isinstance(arguments, dict):
        return {}

    safe_arguments: dict[str, Any] = {}

    for key, value in list(arguments.items())[:MAX_METADATA_KEYS]:
        if not isinstance(key, str):
            continue

        normalized_key = key.strip()

        if not normalized_key:
            continue

        key_lower = normalized_key.lower()

        if key_lower in SENSITIVE_ARGUMENT_KEYS or any(
            fragment in key_lower
            for fragment in SENSITIVE_KEY_FRAGMENTS
        ):
            safe_arguments[normalized_key] = REDACTED_VALUE
            continue

        if isinstance(value, str):
            safe_arguments[normalized_key] = truncate_text(
                _redact_sensitive_value(value),
                min(
                    MAX_METADATA_VALUE_CHARS,
                    MAX_TOOL_ARGUMENT_CHARS,
                ),
            )
        elif value is None or isinstance(
            value,
            (bool, int, float),
        ):
            safe_arguments[normalized_key] = value
        else:
            safe_arguments[normalized_key] = UNSERIALIZABLE_VALUE

    return safe_arguments


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------


def build_error_response(
    exc: BaseException,
    *,
    request_id: str | None = None,
    tool_name: str | None = None,
) -> dict[str, Any]:
    """
    Build a safe MCP error response.

    Args:
        exc: Application or unexpected exception.
        request_id: Optional request correlation ID.
        tool_name: Optional MCP tool name.

    Returns:
        A JSON-safe error response.
    """
    error_code = get_error_code(exc)
    safe_message = get_safe_error_message(exc)
    status_code = get_error_status_code(exc)
    retry_after_seconds = get_retry_after_seconds(exc)

    log_context = {
        "error_code": error_code,
        "request_id": request_id,
        "tool_name": tool_name,
        "retryable": is_retryable_error(exc),
    }

    if isinstance(
        exc,
        (
            AppError,
            WorkflowStateError,
            ValidationError,
            AuthenticationError,
            AuthorizationError,
            NotFoundError,
            RateLimitError,
        ),
    ):
        logger.warning(
            "mcp_operation_failed",
            **log_context,
        )
    else:
        logger.exception(
            "mcp_operation_unexpected_failure",
            **log_context,
        )

    error_payload: dict[str, Any] = {
        "code": error_code,
        "message": safe_message,
        "status": status_code,
        "request_id": request_id,
        "retryable": is_retryable_error(exc),
    }

    if tool_name:
        error_payload["tool_name"] = tool_name

    if retry_after_seconds is not None:
        error_payload["retry_after_seconds"] = (
            retry_after_seconds
        )

    return {
        "error": error_payload,
    }


# ---------------------------------------------------------------------------
# Tool error boundary
# ---------------------------------------------------------------------------


def mcp_tool_error_boundary(
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """
    Decorator that converts tool failures into safe MCP errors.

    The decorator:
    - Measures tool duration.
    - Logs only non-sensitive execution metadata.
    - Sanitizes successful results.
    - Converts known application errors into stable MCP errors.
    - Prevents unexpected exceptions from leaking internal details.

    Args:
        func: Async MCP tool function.

    Returns:
        Wrapped async MCP tool function.
    """

    @wraps(func)
    async def wrapper(
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        started_at = time.perf_counter()

        try:
            result = await func(*args, **kwargs)

            duration_ms = (
                time.perf_counter() - started_at
            ) * 1_000

            logger.info(
                "mcp_tool_invocation_completed",
                tool_name=func.__name__,
                duration_ms=round(duration_ms, 3),
            )

            return sanitize_tool_result(result)

        except ValidationError as exc:
            logger.warning(
                "mcp_tool_validation_failed",
                tool_name=func.__name__,
                error_code=get_error_code(exc),
            )

            raise

        except AuthenticationError:
            logger.warning(
                "mcp_tool_authentication_failed",
                tool_name=func.__name__,
            )

            raise

        except AuthorizationError:
            logger.warning(
                "mcp_tool_authorization_failed",
                tool_name=func.__name__,
            )

            raise

        except NotFoundError:
            logger.warning(
                "mcp_tool_resource_not_found",
                tool_name=func.__name__,
            )

            raise

        except RateLimitError:
            logger.warning(
                "mcp_tool_rate_limited",
                tool_name=func.__name__,
            )

            raise

        except (ProviderTimeoutError, ExternalProviderError) as exc:
            duration_ms = (
                time.perf_counter() - started_at
            ) * 1_000

            logger.error(
                "mcp_tool_provider_failure",
                tool_name=func.__name__,
                error_code=get_error_code(exc),
                duration_ms=round(duration_ms, 3),
            )

            raise

        except (AppError, WorkflowStateError) as exc:
            duration_ms = (
                time.perf_counter() - started_at
            ) * 1_000

            logger.error(
                "mcp_tool_application_failed",
                tool_name=func.__name__,
                error_code=get_error_code(exc),
                duration_ms=round(duration_ms, 3),
            )

            raise

        except Exception as exc:
            duration_ms = (
                time.perf_counter() - started_at
            ) * 1_000

            logger.exception(
                "mcp_tool_unexpected_failure",
                tool_name=func.__name__,
                duration_ms=round(duration_ms, 3),
            )

            raise AppError(
                UNEXPECTED_ERROR_MESSAGE,
                code="internal_error",
            ) from exc

    return wrapper