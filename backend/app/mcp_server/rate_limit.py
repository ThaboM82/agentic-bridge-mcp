"""
In-process rate limiting for MCP tool invocations.

This limiter protects workflow capacity, external-provider quota, and LLM
spend from accidental or abusive MCP usage. It uses a per-principal,
per-tool sliding window and supports optional tenant-level limits.

For multi-instance production deployments, replace the in-memory store with a
shared Redis-backed limiter using the same public interface.

Security rules:
- Rate-limit by authenticated principal and tool name.
- Optionally enforce tenant-level limits.
- Never log tool arguments or results.
- Fail closed when the limiter cannot determine a principal.
- Do not expose internal limiter state, storage keys, or infrastructure
  details in client-facing errors.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.core.exceptions import RateLimitError
from app.mcp_server.auth import MCPPrincipal


logger = logging.getLogger("agentic_bridge.mcp_server.rate_limit")

DEFAULT_REQUESTS_PER_MINUTE = 60
DEFAULT_BURST = 20
DEFAULT_WINDOW_SECONDS = 60

MAX_REQUESTS_PER_MINUTE = 10_000
MAX_BURST = 1_000
MAX_WINDOW_SECONDS = 3_600

DEFAULT_RETRY_AFTER_SECONDS = 60


@dataclass(frozen=True)
class RateLimitRule:
    """Rate-limit rule for an MCP tool."""

    requests_per_minute: int
    burst: int
    window_seconds: int = DEFAULT_WINDOW_SECONDS


@dataclass(frozen=True)
class RateLimitDecision:
    """Result of a rate-limit evaluation."""

    allowed: bool
    limit: int
    remaining: int
    window_seconds: int
    reset_at: float | None = None
    retry_after_seconds: int | None = None


@dataclass
class TenantRateLimitRule:
    """Optional tenant-level rate-limit rule for an MCP tool."""

    burst: int
    window_seconds: int = DEFAULT_WINDOW_SECONDS


class MCPRateLimiter:
    """
    In-memory sliding-window rate limiter for MCP tools.

    User limits are enforced independently for each `(tenant_id, user_id,
    tool_name)` tuple. Optional tenant limits are enforced for each
    `(tenant_id, tool_name)` tuple.

    This prevents one user from consuming another user's quota while still
    allowing tenant-level capacity controls.
    """

    def __init__(
        self,
        *,
        default_requests_per_minute: int = (
            DEFAULT_REQUESTS_PER_MINUTE
        ),
        default_burst: int = DEFAULT_BURST,
        default_window_seconds: int = (
            DEFAULT_WINDOW_SECONDS
        ),
    ) -> None:
        """
        Initialize the MCP rate limiter.

        Args:
            default_requests_per_minute: Default sustained request rate.
            default_burst: Default immediate burst allowance.
            default_window_seconds: Default sliding-window duration.
        """
        self._default_rule = RateLimitRule(
            requests_per_minute=self._bounded_positive(
                default_requests_per_minute,
                maximum=MAX_REQUESTS_PER_MINUTE,
            ),
            burst=self._bounded_positive(
                default_burst,
                maximum=MAX_BURST,
            ),
            window_seconds=self._bounded_positive(
                default_window_seconds,
                maximum=MAX_WINDOW_SECONDS,
            ),
        )

        self._rules: dict[str, RateLimitRule] = {}
        self._tenant_rules: dict[
            str,
            TenantRateLimitRule,
        ] = {}

        self._user_events: dict[
            tuple[str, str, str],
            deque[float],
        ] = defaultdict(deque)

        self._tenant_events: dict[
            tuple[str, str],
            deque[float],
        ] = defaultdict(deque)

    @staticmethod
    def _bounded_positive(
        value: int,
        *,
        maximum: int,
    ) -> int:
        """Bound a positive integer configuration value."""
        return max(
            1,
            min(value, maximum),
        )

    def configure_tool(
        self,
        *,
        tool_name: str,
        requests_per_minute: int,
        burst: int,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ) -> None:
        """
        Configure a tool-specific user rate limit.

        Args:
            tool_name: MCP tool name.
            requests_per_minute: Allowed requests per minute.
            burst: Maximum immediate burst size.
            window_seconds: Sliding-window duration in seconds.

        Raises:
            ValueError: If the limits are invalid.
        """
        normalized_tool_name = self._normalize_tool_name(
            tool_name
        )

        self._rules[normalized_tool_name] = RateLimitRule(
            requests_per_minute=self._bounded_positive(
                requests_per_minute,
                maximum=MAX_REQUESTS_PER_MINUTE,
            ),
            burst=self._bounded_positive(
                burst,
                maximum=MAX_BURST,
            ),
            window_seconds=self._bounded_positive(
                window_seconds,
                maximum=MAX_WINDOW_SECONDS,
            ),
        )

    def configure_tenant_limit(
        self,
        *,
        tool_name: str,
        burst: int,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ) -> None:
        """
        Configure an optional tenant-level rate limit.

        Args:
            tool_name: MCP tool name.
            burst: Maximum tenant-wide burst size.
            window_seconds: Sliding-window duration in seconds.

        Raises:
            ValueError: If the limits are invalid.
        """
        normalized_tool_name = self._normalize_tool_name(
            tool_name
        )

        self._tenant_rules[normalized_tool_name] = (
            TenantRateLimitRule(
                burst=self._bounded_positive(
                    burst,
                    maximum=MAX_BURST,
                ),
                window_seconds=self._bounded_positive(
                    window_seconds,
                    maximum=MAX_WINDOW_SECONDS,
                ),
            )
        )

    def check_rate_limit(
        self,
        *,
        principal: MCPPrincipal,
        tool_name: str,
    ) -> dict[str, Any]:
        """
        Check and record a tool invocation.

        Args:
            principal: Authenticated MCP principal.
            tool_name: MCP tool name.

        Returns:
            Rate-limit metadata safe for audit logging.

        Raises:
            RateLimitError: If the caller or tenant exceeded the configured
                limit.
        """
        decision = self.evaluate_rate_limit(
            principal=principal,
            tool_name=tool_name,
        )

        if not decision.allowed:
            retry_after_seconds = (
                decision.retry_after_seconds
                or DEFAULT_RETRY_AFTER_SECONDS
            )

            logger.warning(
                "mcp_rate_limit_exceeded",
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                tool_name=tool_name,
                limit=decision.limit,
                window_seconds=decision.window_seconds,
                retry_after_seconds=retry_after_seconds,
            )

            raise RateLimitError(
                "MCP tool rate limit exceeded. Try again later.",
                retry_after_seconds=retry_after_seconds,
            )

        return {
            "tool_name": tool_name,
            "limit": decision.limit,
            "remaining": decision.remaining,
            "window_seconds": decision.window_seconds,
            "reset_at": decision.reset_at,
        }

    def evaluate_rate_limit(
        self,
        *,
        principal: MCPPrincipal,
        tool_name: str,
    ) -> RateLimitDecision:
        """
        Evaluate a rate limit without raising a rate-limit exception.

        Args:
            principal: Authenticated MCP principal.
            tool_name: MCP tool name.

        Returns:
            A rate-limit decision.

        Raises:
            RateLimitError: If the principal cannot be determined.
        """
        self._validate_principal(
            principal,
            tool_name=tool_name,
        )

        normalized_tool_name = self._normalize_tool_name(
            tool_name
        )

        rule = self._rules.get(
            normalized_tool_name,
            self._default_rule,
        )

        current_time = time.monotonic()
        user_window_start = (
            current_time - rule.window_seconds
        )

        user_key = (
            principal.tenant_id,
            principal.user_id,
            normalized_tool_name,
        )

        user_events = self._user_events[user_key]

        self._prune_events(
            user_events,
            window_start=user_window_start,
        )

        user_limit = rule.burst
        user_exceeded = (
            len(user_events) >= user_limit
        )

        tenant_rule = self._tenant_rules.get(
            normalized_tool_name
        )

        tenant_limit = (
            tenant_rule.burst
            if tenant_rule
            else None
        )

        tenant_events: deque[float] | None = None
        tenant_window_start: float | None = None
        tenant_exceeded = False

        if tenant_rule is not None:
            tenant_window_start = (
                current_time
                - tenant_rule.window_seconds
            )

            tenant_key = (
                principal.tenant_id,
                normalized_tool_name,
            )

            tenant_events = self._tenant_events[
                tenant_key
            ]

            self._prune_events(
                tenant_events,
                window_start=tenant_window_start,
            )

            tenant_exceeded = (
                len(tenant_events) >= tenant_limit
            )

        if user_exceeded or tenant_exceeded:
            if user_exceeded:
                blocked_events = user_events
                blocked_window_seconds = (
                    rule.window_seconds
                )
                blocked_limit = user_limit
            else:
                blocked_events = tenant_events
                blocked_window_seconds = (
                    tenant_rule.window_seconds
                )
                blocked_limit = tenant_limit

            return RateLimitDecision(
                allowed=False,
                limit=blocked_limit,
                remaining=0,
                window_seconds=blocked_window_seconds,
                reset_at=self._calculate_reset_at(
                    events=blocked_events,
                    window_seconds=blocked_window_seconds,
                ),
                retry_after_seconds=(
                    self._calculate_retry_after(
                        events=blocked_events,
                        window_seconds=blocked_window_seconds,
                    )
                ),
            )

        user_events.append(current_time)

        if tenant_events is not None:
            tenant_events.append(current_time)

        return RateLimitDecision(
            allowed=True,
            limit=user_limit,
            remaining=max(
                0,
                user_limit - len(user_events),
            ),
            window_seconds=rule.window_seconds,
            reset_at=self._calculate_reset_at(
                events=user_events,
                window_seconds=rule.window_seconds,
            ),
            retry_after_seconds=None,
        )

    def reset_principal(
        self,
        *,
        principal: MCPPrincipal,
        tool_name: str | None = None,
    ) -> None:
        """
        Reset rate-limit state for a principal.

        This is intended for tests and administrative recovery only.

        Args:
            principal: Authenticated MCP principal.
            tool_name: Optional tool name. If omitted, all user counters for
                the principal are cleared.
        """
        if principal is None:
            return

        if tool_name:
            normalized_tool_name = (
                self._normalize_tool_name(tool_name)
            )

            self._user_events.pop(
                (
                    principal.tenant_id,
                    principal.user_id,
                    normalized_tool_name,
                ),
                None,
            )

            return

        user_keys_to_remove = [
            key
            for key in self._user_events
            if key[0] == principal.tenant_id
            and key[1] == principal.user_id
        ]

        for key in user_keys_to_remove:
            self._user_events.pop(key, None)

    def reset_tenant(
        self,
        *,
        tenant_id: str | UUID,
        tool_name: str | None = None,
    ) -> None:
        """
        Reset tenant-level rate-limit state.

        This is intended for tests and administrative recovery only.

        Args:
            tenant_id: Tenant identifier.
            tool_name: Optional tool name. If omitted, all tenant counters
                for the tenant are cleared.
        """
        normalized_tenant_id = str(tenant_id)

        if tool_name:
            normalized_tool_name = (
                self._normalize_tool_name(tool_name)
            )

            self._tenant_events.pop(
                (
                    normalized_tenant_id,
                    normalized_tool_name,
                ),
                None,
            )

            return

        tenant_keys_to_remove = [
            key
            for key in self._tenant_events
            if key[0] == normalized_tenant_id
        ]

        for key in tenant_keys_to_remove:
            self._tenant_events.pop(key, None)

    def reset_all(self) -> None:
        """
        Clear all in-memory rate-limit state.

        This is intended for tests only.
        """
        self._user_events.clear()
        self._tenant_events.clear()

    @staticmethod
    def _normalize_tool_name(
        tool_name: str,
    ) -> str:
        """
        Validate and normalize a tool name.

        Args:
            tool_name: MCP tool name.

        Returns:
            A normalized tool name.

        Raises:
            ValueError: If the tool name is missing or malformed.
        """
        if not tool_name or not isinstance(
            tool_name,
            str,
        ):
            raise ValueError(
                "Tool name is required."
            )

        normalized_tool_name = tool_name.strip()

        if not normalized_tool_name:
            raise ValueError(
                "Tool name is required."
            )

        return normalized_tool_name

    @staticmethod
    def _validate_principal(
        principal: MCPPrincipal | None,
        *,
        tool_name: str,
    ) -> None:
        """
        Validate that a principal can be rate-limited.

        Args:
            principal: Authenticated MCP principal.
            tool_name: MCP tool name.

        Raises:
            RateLimitError: If the principal is missing or incomplete.
        """
        if principal is None:
            logger.warning(
                "mcp_rate_limit_principal_missing",
                tool_name=tool_name,
            )

            raise RateLimitError(
                "MCP caller authentication is required.",
                retry_after_seconds=0,
            )

        if not principal.user_id or not principal.tenant_id:
            logger.warning(
                "mcp_rate_limit_principal_incomplete",
                tool_name=tool_name,
            )

            raise RateLimitError(
                "MCP caller authentication is required.",
                retry_after_seconds=0,
            )

    @staticmethod
    def _prune_events(
        events: deque[float],
        *,
        window_start: float,
    ) -> None:
        """Remove expired events from a sliding window."""
        while events and events[0] < window_start:
            events.popleft()

    @staticmethod
    def _calculate_retry_after(
        *,
        events: deque[float],
        window_seconds: int,
    ) -> int:
        """
        Calculate a safe retry delay.

        Args:
            events: Current event timestamps.
            window_seconds: Sliding-window duration.

        Returns:
            Retry delay in seconds.
        """
        if not events:
            return DEFAULT_RETRY_AFTER_SECONDS

        oldest_event = events[0]
        elapsed = time.monotonic() - oldest_event

        remaining_window = (
            window_seconds - elapsed
        )

        if remaining_window <= 0:
            return 1

        return max(
            1,
            min(
                int(remaining_window) + 1,
                MAX_WINDOW_SECONDS,
            ),
        )

    @staticmethod
    def _calculate_reset_at(
        *,
        events: deque[float],
        window_seconds: int,
    ) -> float | None:
        """
        Calculate when the current window resets.

        Args:
            events: Current event timestamps.
            window_seconds: Sliding-window duration.

        Returns:
            Reset timestamp, or `None`.
        """
        if not events:
            return None

        return events[0] + window_seconds


# Configure conservative defaults for expensive or state-changing tools.
rate_limiter = MCPRateLimiter(
    default_requests_per_minute=60,
    default_burst=20,
)


# Workflow tools


rate_limiter.configure_tool(
    tool_name="start_generation_workflow",
    requests_per_minute=10,
    burst=5,
    window_seconds=60,
)

rate_limiter.configure_tool(
    tool_name="cancel_workflow",
    requests_per_minute=30,
    burst=10,
    window_seconds=60,
)

rate_limiter.configure_tool(
    tool_name="approve_workflow",
    requests_per_minute=30,
    burst=10,
    window_seconds=60,
)

rate_limiter.configure_tool(
    tool_name="reject_workflow",
    requests_per_minute=30,
    burst=10,
    window_seconds=60,
)

rate_limiter.configure_tool(
    tool_name="request_workflow_changes",
    requests_per_minute=30,
    burst=10,
    window_seconds=60,
)

rate_limiter.configure_tool(
    tool_name="retrieve_workflow_context",
    requests_per_minute=120,
    burst=30,
    window_seconds=60,
)


# Retrieval tools


rate_limiter.configure_tool(
    tool_name="search_design_knowledge",
    requests_per_minute=60,
    burst=20,
    window_seconds=60,
)

rate_limiter.configure_tool(
    tool_name="search_domain_knowledge",
    requests_per_minute=60,
    burst=20,
    window_seconds=60,
)


# CMS tools


rate_limiter.configure_tool(
    tool_name="search_cms_content",
    requests_per_minute=60,
    burst=20,
    window_seconds=60,
)

rate_limiter.configure_tool(
    tool_name="create_cms_draft",
    requests_per_minute=20,
    burst=10,
    window_seconds=60,
)

rate_limiter.configure_tool(
    tool_name="update_cms_draft",
    requests_per_minute=30,
    burst=10,
    window_seconds=60,
)

rate_limiter.configure_tool(
    tool_name="request_cms_publication",
    requests_per_minute=10,
    burst=5,
    window_seconds=60,
)


# Figma tools


rate_limiter.configure_tool(
    tool_name="get_figma_design_context",
    requests_per_minute=30,
    burst=10,
    window_seconds=60,
)

rate_limiter.configure_tool(
    tool_name="get_figma_frame_summary",
    requests_per_minute=60,
    burst=20,
    window_seconds=60,
)


# WordPress tools


rate_limiter.configure_tool(
    tool_name="search_wordpress_content",
    requests_per_minute=60,
    burst=20,
    window_seconds=60,
)

rate_limiter.configure_tool(
    tool_name="create_wordpress_draft",
    requests_per_minute=20,
    burst=10,
    window_seconds=60,
)

rate_limiter.configure_tool(
    tool_name="update_wordpress_draft",
    requests_per_minute=30,
    burst=10,
    window_seconds=60,
)

rate_limiter.configure_tool(
    tool_name="request_wordpress_publication",
    requests_per_minute=10,
    burst=5,
    window_seconds=60,
)


# Tenant-wide safeguards for the most expensive operations.


rate_limiter.configure_tenant_limit(
    tool_name="start_generation_workflow",
    burst=25,
    window_seconds=60,
)

rate_limiter.configure_tenant_limit(
    tool_name="get_figma_design_context",
    burst=100,
    window_seconds=60,
)

rate_limiter.configure_tenant_limit(
    tool_name="request_cms_publication",
    burst=20,
    window_seconds=60,
)

rate_limiter.configure_tenant_limit(
    tool_name="request_wordpress_publication",
    burst=20,
    window_seconds=60,
)