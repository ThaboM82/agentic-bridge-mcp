"""
Rate limiting.

Provides a process-local sliding-window limiter for local/single-process
deployments and an optional Redis-backed limiter for production deployments
with multiple workers, replicas, or containers.

Important:
- In-memory counters are per process and are reset on restart.
- Redis-backed counters are shared across all application replicas.
- Never use unvalidated client-supplied identifiers as the sole rate-limit key.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque

from fastapi import (
    HTTPException,
    Request,
    status,
)

from app.config import settings
from app.core.constants import (
    RATE_LIMIT_LIMIT_HEADER,
    RATE_LIMIT_REMAINING_HEADER,
    RATE_LIMIT_RESET_HEADER,
    REQUEST_ID_HEADER,
    RETRY_AFTER_HEADER,
)
from app.core.logging import get_logger


logger = get_logger(__name__)


MAX_KEY_LENGTH = 256
MAX_SCOPE_LENGTH = 64
DEFAULT_CLIENT_ID = "anonymous"


# ---------------------------------------------------------------------------
# Rate-limit result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    """Result of a rate-limit check."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int
    reset_at: int

    @property
    def headers(self) -> dict[str, str]:
        """Return standard rate-limit response headers."""
        return {
            RATE_LIMIT_LIMIT_HEADER: str(
                self.limit,
            ),
            RATE_LIMIT_REMAINING_HEADER: str(
                self.remaining,
            ),
            RATE_LIMIT_RESET_HEADER: str(
                self.reset_at,
            ),
            RETRY_AFTER_HEADER: str(
                self.retry_after_seconds,
            ),
        }


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """
    Compatibility rate-limit decision object.

    Some modules and tests use `RateLimitDecision`; new code should prefer
    `RateLimitResult`.
    """

    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int = 0
    reset_after_seconds: float = 0.0

    @property
    def headers(self) -> dict[str, str]:
        """Return standard rate-limit response headers."""
        return {
            RATE_LIMIT_LIMIT_HEADER: str(
                self.limit,
            ),
            RATE_LIMIT_REMAINING_HEADER: str(
                self.remaining,
            ),
            RATE_LIMIT_RESET_HEADER: str(
                int(time.time())
                + int(
                    self.reset_after_seconds,
                ),
            ),
            RETRY_AFTER_HEADER: str(
                self.retry_after_seconds,
            ),
        }


# ---------------------------------------------------------------------------
# Limiter interface
# ---------------------------------------------------------------------------


class BaseRateLimiter(ABC):
    """Interface for rate-limit backends."""

    @abstractmethod
    def check(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitResult:
        """Check whether an operation is allowed under the configured limit."""


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------


class InMemoryRateLimiter(
    BaseRateLimiter,
):
    """
    Process-local sliding-window rate limiter.

    Suitable for local development, tests, and single-worker deployments.
    Do not rely on this implementation when running multiple Uvicorn workers,
    containers, or horizontally scaled API replicas.
    """

    def __init__(
        self,
        *,
        max_tracked_keys: int = 10_000,
    ) -> None:
        self._requests: dict[
            str,
            Deque[float],
        ] = defaultdict(deque)

        self._max_tracked_keys = (
            max_tracked_keys
        )

    def check(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitResult:
        """Check whether a request is allowed using a sliding time window."""
        safe_limit = max(1, limit)
        safe_window = max(
            1,
            window_seconds,
        )

        now = time.monotonic()
        window_start = now - safe_window

        history = self._requests[key]

        while (
            history
            and history[0] <= window_start
        ):
            history.popleft()

        reset_at = int(
            time.time() + safe_window,
        )

        if len(history) >= safe_limit:
            oldest_request = history[0]

            retry_after = max(
                1,
                int(
                    safe_window
                    - (now - oldest_request),
                ),
            )

            return RateLimitResult(
                allowed=False,
                limit=safe_limit,
                remaining=0,
                retry_after_seconds=retry_after,
                reset_at=int(
                    time.time()
                    + retry_after,
                ),
            )

        history.append(now)

        self._evict_if_needed()

        return RateLimitResult(
            allowed=True,
            limit=safe_limit,
            remaining=safe_limit
            - len(history),
            retry_after_seconds=0,
            reset_at=reset_at,
        )

    def reset(
        self,
        key: str | None = None,
    ) -> None:
        """Clear one key or all tracked rate-limit windows."""
        if key is None:
            self._requests.clear()

            return

        self._requests.pop(
            key,
            None,
        )

    def _evict_if_needed(self) -> None:
        """
        Prevent unbounded memory growth.

        This is a coarse safeguard, not a replacement for Redis in distributed
        production deployments.
        """
        if (
            len(self._requests)
            <= self._max_tracked_keys
        ):
            return

        keys_to_remove = list(
            self._requests.keys(),
        )[
            : max(
                0,
                len(self._requests)
                - self._max_tracked_keys,
            )
        ]

        for key in keys_to_remove:
            self._requests.pop(
                key,
                None,
            )


# ---------------------------------------------------------------------------
# Redis implementation
# ---------------------------------------------------------------------------


class RedisRateLimiter(
    BaseRateLimiter,
):
    """
    Redis-backed fixed-window rate limiter.

    Uses an atomic Redis Lua script so multiple API replicas share the same
    counters without race conditions.
    """

    _LUA_SCRIPT = """
    local current_time = tonumber(ARGV[1])
    local window_seconds = tonumber(ARGV[2])
    local limit = tonumber(ARGV[3])

    local window_id = math.floor(current_time / window_seconds)
    local window_key = KEYS[1] .. ':' .. window_id

    local current_count = tonumber(redis.call('GET', window_key) or '0')

    if current_count >= limit then
        local ttl = redis.call('TTL', window_key)
        if ttl < 0 then
            ttl = window_seconds
        end

        return {0, limit, 0, current_time + ttl}
    end

    local new_count = redis.call('INCR', window_key)

    if new_count == 1 then
        redis.call('EXPIRE', window_key, window_seconds)
    end

    return {1, limit, limit - new_count, current_time + window_seconds}
    """

    def __init__(
        self,
        redis_client: Any,
    ) -> None:
        """
        Args:
            redis_client: An async Redis client, such as
                `redis.asyncio.Redis`.
        """
        self._redis = redis_client

        self._script = (
            self._redis.register_script(
                self._LUA_SCRIPT,
            )
        )

    async def check_async(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitResult:
        """Check a rate limit using Redis."""
        safe_key = _hash_key(key)
        safe_limit = max(1, limit)
        safe_window = max(
            1,
            window_seconds,
        )

        now = int(time.time())

        result = await self._script(
            keys=[
                f"rate_limit:{safe_key}",
            ],
            args=[
                now,
                safe_window,
                safe_limit,
            ],
        )

        (
            allowed,
            limit_value,
            remaining,
            reset_at,
        ) = result

        return RateLimitResult(
            allowed=bool(int(allowed)),
            limit=int(limit_value),
            remaining=int(remaining),
            retry_after_seconds=max(
                0,
                int(reset_at) - now,
            ),
            reset_at=int(reset_at),
        )

    def check(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitResult:
        """Synchronous compatibility method; prefer `check_async`."""
        raise RuntimeError(
            "RedisRateLimiter requires await "
            "rate_limiter.check_async().",
        )


# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------


def _hash_key(key: str) -> str:
    """Create a bounded, privacy-preserving rate-limit key."""
    return hashlib.sha256(
        key.encode("utf-8"),
    ).hexdigest()


def _validate_scope(
    scope: str,
) -> str:
    """Validate a rate-limit scope name."""
    normalized = scope.strip().lower()

    if not normalized:
        raise ValueError(
            "Rate-limit scope cannot be empty.",
        )

    if (
        len(normalized)
        > MAX_SCOPE_LENGTH
    ):
        raise ValueError(
            f"Rate-limit scope exceeds {MAX_SCOPE_LENGTH} characters.",
        )

    if not all(
        character.isalnum()
        or character
        in {"-", "_", ":"}
        for character in normalized
    ):
        raise ValueError(
            "Rate-limit scope may contain only letters, numbers, hyphens, "
            "underscores, and colons.",
        )

    return normalized


def get_client_identifier(
    request: Request,
) -> str:
    """
    Resolve a stable client identifier for rate limiting.

    Preference order:
    1. Authenticated user ID, when attached by authentication middleware.
    2. Tenant ID and client IP, when both are available.
    3. Client IP from trusted proxy headers or the socket peer.
    """
    user_id = getattr(
        request.state,
        "user_id",
        None,
    )

    if user_id:
        return f"user:{user_id}"

    tenant_id = getattr(
        request.state,
        "tenant_id",
        None,
    )

    client_ip = _get_client_ip(
        request,
    )

    if tenant_id:
        return (
            f"tenant:{tenant_id}:ip:{client_ip}"
        )

    return f"ip:{client_ip}"


def _get_client_ip(
    request: Request,
) -> str:
    """Resolve client IP from trusted proxy headers or the connection peer."""
    forwarded_for = request.headers.get(
        "X-Forwarded-For",
    )

    if forwarded_for:
        first_ip = forwarded_for.split(
            ",",
        )[0].strip()

        if first_ip:
            return first_ip

    real_ip = request.headers.get(
        "X-Real-IP",
    )

    if real_ip:
        return real_ip.strip()

    return (
        request.client.host
        if request.client
        else DEFAULT_CLIENT_ID
    )


def _build_rate_limit_key(
    scope: str,
    request: Request,
) -> str:
    """Build a bounded rate-limit key for the request and scope."""
    safe_scope = _validate_scope(
        scope,
    )

    client_identifier = (
        get_client_identifier(request)
    )

    key = (
        f"{safe_scope}:{client_identifier}"
    )

    if len(key) > MAX_KEY_LENGTH:
        key = _hash_key(key)

    return key


# ---------------------------------------------------------------------------
# Default limiter
# ---------------------------------------------------------------------------


rate_limiter = InMemoryRateLimiter()


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def rate_limit(
    limit: int,
    window_seconds: int,
    scope: str,
    *,
    enabled: bool | None = None,
):
    """
    Create a FastAPI dependency that applies route-level rate limiting.

    Example:
        @router.post(
            "/workflows",
            dependencies=[Depends(rate_limit(20, 60, "workflow:create"))],
        )
        async def create_workflow(...):
            ...
    """
    safe_scope = _validate_scope(scope)

    rate_limit_enabled = (
        settings.RATE_LIMIT_ENABLED
        if enabled is None
        else enabled
    )

    async def dependency(
        request: Request,
    ) -> None:
        if not rate_limit_enabled:
            return

        if (
            settings.APP_ENV
            == "development"
            and not settings.RATE_LIMIT_ENABLED
        ):
            return

        key = _build_rate_limit_key(
            safe_scope,
            request,
        )

        result = rate_limiter.check(
            key=key,
            limit=limit,
            window_seconds=window_seconds,
        )

        request.state.rate_limit_limit = (
            result.limit
        )

        request.state.rate_limit_remaining = (
            result.remaining
        )

        request.state.rate_limit_reset_at = (
            result.reset_at
        )

        if not result.allowed:
            logger.warning(
                "rate_limit_exceeded",
                extra={
                    "scope": safe_scope,
                    "client_ip": _get_client_ip(
                        request,
                    ),
                    "request_id": getattr(
                        request.state,
                        "request_id",
                        "",
                    ),
                    "limit": result.limit,
                    "retry_after_seconds": result.retry_after_seconds,
                },
            )

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please retry later.",
                headers=result.headers,
            )

        logger.debug(
            "rate_limit_checked",
            extra={
                "scope": safe_scope,
                "remaining": result.remaining,
                "limit": result.limit,
            },
        )

    return dependency


def authenticated_rate_limit(
    limit: int,
    window_seconds: int,
    scope: str,
):
    """
    Create a rate-limit dependency keyed by authenticated user ID.
    """
    return rate_limit(
        limit=limit,
        window_seconds=window_seconds,
        scope=f"auth:{scope}",
    )


def ip_rate_limit(
    limit: int,
    window_seconds: int,
    scope: str,
):
    """
    Create a rate-limit dependency keyed only by client IP.
    """
    return rate_limit(
        limit=limit,
        window_seconds=window_seconds,
        scope=f"ip:{scope}",
    )


__all__ = [
    "BaseRateLimiter",
    "DEFAULT_CLIENT_ID",
    "InMemoryRateLimiter",
    "RateLimitDecision",
    "RateLimitResult",
    "RedisRateLimiter",
    "authenticated_rate_limit",
    "get_client_identifier",
    "ip_rate_limit",
    "rate_limit",
    "rate_limiter",
]