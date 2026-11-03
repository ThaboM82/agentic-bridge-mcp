"""
API rate-limit dependencies and response helpers.

This module provides reusable FastAPI dependencies for per-principal,
per-tool, and per-tenant rate limiting. It also builds standards-aligned
rate-limit response headers and converts limiter failures into consistent
HTTP 429 responses.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Annotated, Any, Awaitable, Callable

from fastapi import (
    Depends,
    HTTPException,
    Request,
    Response,
    status,
)

from app.api.dependencies import (
    CurrentPrincipal,
    RateLimiter,
)


logger = logging.getLogger(
    "agentic_bridge.api.rate_limit"
)


# ---------------------------------------------------------------------------
# Rate-limit result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateLimitResult:
    """Outcome of a rate-limit check."""

    allowed: bool
    limit: int
    remaining: int
    reset_at: float | None = None
    retry_after_seconds: int | None = None

    @property
    def retry_after(self) -> int:
        """Return the number of seconds the client should wait."""
        if self.retry_after_seconds is not None:
            return max(
                self.retry_after_seconds,
                0,
            )

        if self.reset_at is None:
            return 60

        return max(
            int(
                self.reset_at
                - time.time()
            ),
            0,
        )


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------


def build_rate_limit_headers(
    *,
    limit: int = 60,
    remaining: int = 0,
    retry_after_seconds: int = 60,
    reset_at: float | None = None,
) -> dict[str, str]:
    """Build standard rate-limit response headers."""
    headers = {
        "Retry-After": str(
            max(
                retry_after_seconds,
                0,
            )
        ),
        "X-RateLimit-Limit": str(
            max(
                limit,
                0,
            )
        ),
        "X-RateLimit-Remaining": str(
            max(
                remaining,
                0,
            )
        ),
    }

    if reset_at is not None:
        headers[
            "X-RateLimit-Reset"
        ] = str(
            int(reset_at)
        )

    return headers


def apply_rate_limit_headers(
    response: Response,
    result: RateLimitResult,
) -> None:
    """Apply rate-limit headers to a successful response."""
    headers = build_rate_limit_headers(
        limit=result.limit,
        remaining=result.remaining,
        retry_after_seconds=result.retry_after,
        reset_at=result.reset_at,
    )

    for header_name, header_value in headers.items():
        response.headers[
            header_name
        ] = header_value


def raise_rate_limit_error(
    *,
    result: RateLimitResult,
    detail: str = "Too many requests.",
) -> HTTPException:
    """Create an HTTP 429 exception with rate-limit headers."""
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=detail,
        headers=build_rate_limit_headers(
            limit=result.limit,
            remaining=result.remaining,
            retry_after_seconds=result.retry_after,
            reset_at=result.reset_at,
        ),
    )


# ---------------------------------------------------------------------------
# Scope helpers
# ---------------------------------------------------------------------------


def _principal_identity(
    principal: Any,
) -> str:
    """Return a stable identity string for a principal."""
    user_id = getattr(
        principal,
        "user_id",
        None,
    )

    tenant_id = getattr(
        principal,
        "tenant_id",
        None,
    )

    return f"tenant:{tenant_id}:user:{user_id}"


def _normalize_tool_name(
    tool_name: str,
) -> str:
    """Normalize a tool name for limiter keys and logs."""
    return tool_name.strip().lower()


# ---------------------------------------------------------------------------
# Per-principal, per-tool limiter
# ---------------------------------------------------------------------------


def enforce_rate_limit(
    tool_name: str,
    *,
    limit: int | None = None,
    window_seconds: int | None = None,
) -> Callable[
    [CurrentPrincipal, RateLimiter],
    Awaitable[None],
]:
    """
    Return a dependency enforcing a per-principal, per-tool rate limit.

    Args:
        tool_name: Logical name of the protected operation.
        limit: Optional explicit request limit for the window.
        window_seconds: Optional explicit rolling-window length.
    """

    normalized_tool_name = (
        _normalize_tool_name(
            tool_name
        )
    )

    async def dependency(
        principal: CurrentPrincipal,
        rate_limiter: RateLimiter,
    ) -> None:
        identity = _principal_identity(
            principal
        )

        try:
            result = rate_limiter.check_rate_limit(
                principal=principal,
                tool_name=normalized_tool_name,
                limit=limit,
                window_seconds=window_seconds,
            )
        except TypeError:
            # Compatibility with limiters that do not accept overrides.
            result = rate_limiter.check_rate_limit(
                principal=principal,
                tool_name=normalized_tool_name,
            )
        except Exception as exc:
            logger.exception(
                "Rate limiter failure.",
                extra={
                    "identity": identity,
                    "tool_name": normalized_tool_name,
                },
            )

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rate limiting is temporarily unavailable.",
                headers={
                    "Retry-After": "5"
                },
            ) from exc

        if not result.allowed:
            logger.warning(
                "Rate limit exceeded.",
                extra={
                    "identity": identity,
                    "tool_name": normalized_tool_name,
                    "limit": result.limit,
                    "remaining": result.remaining,
                    "retry_after_seconds": result.retry_after,
                },
            )

            raise raise_rate_limit_error(
                result=result,
                detail=(
                    "Too many requests for this operation. "
                    "Please retry later."
                ),
            )

        logger.debug(
            "Rate limit check passed.",
            extra={
                "identity": identity,
                "tool_name": normalized_tool_name,
                "limit": result.limit,
                "remaining": result.remaining,
            },
        )

    return dependency


# ---------------------------------------------------------------------------
# Per-tenant limiter
# ---------------------------------------------------------------------------


def enforce_tenant_rate_limit(
    tool_name: str,
    *,
    limit: int | None = None,
    window_seconds: int | None = None,
) -> Callable[
    [CurrentPrincipal, RateLimiter],
    Awaitable[None],
]:
    """
    Return a dependency enforcing a tenant-wide rate limit.

    This is useful for expensive operations where the limit should be shared
    across all users in a tenant.
    """

    normalized_tool_name = (
        _normalize_tool_name(
            tool_name
        )
    )

    async def dependency(
        principal: CurrentPrincipal,
        rate_limiter: RateLimiter,
    ) -> None:
        tenant_id = getattr(
            principal,
            "tenant_id",
            None,
        )

        identity = (
            f"tenant:{tenant_id}"
        )

        try:
            result = rate_limiter.check_rate_limit(
                principal=principal,
                tool_name=f"tenant:{normalized_tool_name}",
                limit=limit,
                window_seconds=window_seconds,
            )
        except TypeError:
            result = rate_limiter.check_rate_limit(
                principal=principal,
                tool_name=f"tenant:{normalized_tool_name}",
            )
        except Exception as exc:
            logger.exception(
                "Tenant rate limiter failure.",
                extra={
                    "identity": identity,
                    "tool_name": normalized_tool_name,
                },
            )

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rate limiting is temporarily unavailable.",
                headers={
                    "Retry-After": "5"
                },
            ) from exc

        if not result.allowed:
            logger.warning(
                "Tenant rate limit exceeded.",
                extra={
                    "identity": identity,
                    "tool_name": normalized_tool_name,
                    "limit": result.limit,
                    "remaining": result.remaining,
                    "retry_after_seconds": result.retry_after,
                },
            )

            raise raise_rate_limit_error(
                result=result,
                detail=(
                    "Too many requests for this tenant. "
                    "Please retry later."
                ),
            )

        logger.debug(
            "Tenant rate limit check passed.",
            extra={
                "identity": identity,
                "tool_name": normalized_tool_name,
                "limit": result.limit,
                "remaining": result.remaining,
            },
        )

    return dependency


# ---------------------------------------------------------------------------
# Request-scoped limiter
# ---------------------------------------------------------------------------


async def enforce_request_rate_limit(
    request: Request,
    principal: CurrentPrincipal,
    rate_limiter: RateLimiter,
) -> None:
    """Apply a general per-principal API rate limit based on route path."""
    tool_name = _normalize_tool_name(
        request.url.path
    )

    identity = _principal_identity(
        principal
    )

    try:
        result = rate_limiter.check_rate_limit(
            principal=principal,
            tool_name=tool_name,
        )
    except Exception as exc:
        logger.exception(
            "Request rate limiter failure.",
            extra={
                "identity": identity,
                "tool_name": tool_name,
            },
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rate limiting is temporarily unavailable.",
            headers={
                "Retry-After": "5"
            },
        ) from exc

    if not result.allowed:
        logger.warning(
            "Request rate limit exceeded.",
            extra={
                "identity": identity,
                "tool_name": tool_name,
                "limit": result.limit,
                "remaining": result.remaining,
                "retry_after_seconds": result.retry_after,
            },
        )

        raise raise_rate_limit_error(
            result=result,
            detail="Too many requests. Please retry later.",
        )

    logger.debug(
        "Request rate limit check passed.",
        extra={
            "identity": identity,
            "tool_name": tool_name,
            "limit": result.limit,
            "remaining": result.remaining,
        },
    )


RequestRateLimit = Annotated[
    None,
    Depends(
        enforce_request_rate_limit
    ),
]