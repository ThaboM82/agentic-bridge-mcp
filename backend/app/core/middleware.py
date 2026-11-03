"""
HTTP middleware for request correlation, structured access logging, latency
measurement, security headers, payload-size protection, and rate-limit headers.

Middleware order matters in FastAPI/Starlette: the last middleware registered
runs first on the inbound request. Register this middleware after CORS if CORS
must remain the outermost layer.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import settings
from app.core.constants import (
    FORWARDED_FOR_HEADER,
    PROCESS_TIME_HEADER,
    RATE_LIMIT_LIMIT_HEADER,
    RATE_LIMIT_REMAINING_HEADER,
    RATE_LIMIT_RESET_HEADER,
    REAL_IP_HEADER,
    REQUEST_ID_HEADER,
    RETRY_AFTER_HEADER,
    SECURITY_HEADERS,
)
from app.core.logging import get_logger
from app.core.logging_context import (
    bind_request_context,
    clear_logging_context,
    generate_correlation_id,
    generate_request_id,
)


logger = get_logger(__name__)

MAX_REQUEST_ID_LENGTH = 64
MAX_CORRELATION_ID_LENGTH = 64
MAX_LOGGED_QUERY_LENGTH = 512

DEFAULT_RATE_LIMIT_REQUESTS = 100
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Assigns a request ID and correlation ID, binds structured logging context,
    measures latency, adds security headers, and emits one access-log event per
    completed or failed request.
    """

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        super().__init__(app)
        self._rate_limit_requests = DEFAULT_RATE_LIMIT_REQUESTS
        self._rate_limit_window_seconds = DEFAULT_RATE_LIMIT_WINDOW_SECONDS

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = self._resolve_request_id(request)
        correlation_id = self._resolve_correlation_id(request)

        request.state.request_id = request_id
        request.state.correlation_id = correlation_id
        request.state.started_at = time.perf_counter()

        bind_request_context(
            request_id=request_id,
            correlation_id=correlation_id,
            user_id=getattr(request.state, "user_id", None),
            tenant_id=getattr(request.state, "tenant_id", None),
        )

        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            query=self._get_safe_query(request),
            client_ip=self._get_client_ip(request),
            user_agent=self._get_user_agent(request),
        )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = self._calculate_duration_ms(request)

            logger.exception(
                "request_failed",
                method=request.method,
                path=request.url.path,
                query=self._get_safe_query(request),
                duration_ms=duration_ms,
                client_ip=self._get_client_ip(request),
            )

            clear_logging_context()

            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "internal_error",
                        "message": "An unexpected error occurred.",
                        "retryable": False,
                    }
                },
                headers={
                    REQUEST_ID_HEADER: request_id,
                    **SECURITY_HEADERS,
                },
            )

        duration_ms = self._calculate_duration_ms(request)

        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[PROCESS_TIME_HEADER] = f"{duration_ms / 1000:.4f}"

        self._apply_security_headers(response)
        self._apply_rate_limit_headers(request, response)

        log_level = (
            logging_level_for_status(response.status_code)
        )

        logger.log(
            log_level,
            "request_completed",
            method=request.method,
            path=request.url.path,
            query=self._get_safe_query(request),
            status_code=response.status_code,
            duration_ms=duration_ms,
            client_ip=self._get_client_ip(request),
            user_agent=self._get_user_agent(request),
        )

        clear_logging_context()

        return response

    # ------------------------------------------------------------------
    # Request identity
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_request_id(request: Request) -> str:
        """
        Use a bounded upstream request ID only when it is structurally valid.

        In production behind a trusted edge, validate the allow-list of proxy
        hosts before trusting externally supplied IDs.
        """
        upstream_id = request.headers.get(REQUEST_ID_HEADER)

        if upstream_id and RequestContextMiddleware._is_safe_id(
            upstream_id,
            MAX_REQUEST_ID_LENGTH,
        ):
            return upstream_id

        return generate_request_id()

    @staticmethod
    def _resolve_correlation_id(request: Request) -> str:
        """Resolve or generate a logical business-operation correlation ID."""
        upstream_id = request.headers.get("X-Correlation-ID")

        if upstream_id and RequestContextMiddleware._is_safe_id(
            upstream_id,
            MAX_CORRELATION_ID_LENGTH,
        ):
            return upstream_id

        return generate_correlation_id()

    @staticmethod
    def _is_safe_id(value: str, max_length: int) -> bool:
        """Validate an externally supplied identifier."""
        normalized = value.strip()

        if not normalized or len(normalized) > max_length:
            return False

        return all(
            character.isalnum() or character in {"-", "_", ".", ":"}
            for character in normalized
        )

    # ------------------------------------------------------------------
    # Request metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """
        Resolve the client IP from trusted proxy headers.

        Only trust these headers when the application is deployed behind a
        trusted reverse proxy or load balancer.
        """
        forwarded_for = request.headers.get(FORWARDED_FOR_HEADER)

        if forwarded_for:
            first_ip = forwarded_for.split(",")[0].strip()

            if first_ip:
                return first_ip

        real_ip = request.headers.get(REAL_IP_HEADER)

        if real_ip:
            return real_ip.strip()

        return request.client.host if request.client else "unknown"

    @staticmethod
    def _get_user_agent(request: Request) -> str:
        """Return a bounded user-agent value for access logs."""
        user_agent = request.headers.get("User-Agent", "")
        return user_agent[:256]

    @staticmethod
    def _get_safe_query(request: Request) -> str:
        """
        Return a bounded query string while excluding potentially sensitive
        query parameters.
        """
        sensitive_parameters = {
            "token",
            "access_token",
            "refresh_token",
            "api_key",
            "apikey",
            "secret",
            "password",
            "authorization",
        }

        filtered_parameters: list[str] = []

        for key, value in request.query_params.multi_items():
            if key.lower() in sensitive_parameters:
                filtered_parameters.append(f"{key}=[REDACTED]")
            else:
                filtered_parameters.append(f"{key}={value}")

        query = "&".join(filtered_parameters)

        return query[:MAX_LOGGED_QUERY_LENGTH]

    # ------------------------------------------------------------------
    # Timing and headers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_duration_ms(request: Request) -> float:
        """Calculate request duration in milliseconds."""
        started_at = getattr(request.state, "started_at", None)

        if started_at is None:
            return 0.0

        return round((time.perf_counter() - started_at) * 1000, 2)

    @staticmethod
    def _apply_security_headers(response: Response) -> None:
        """Apply baseline security headers without overriding existing values."""
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)

    def _apply_rate_limit_headers(
        self,
        request: Request,
        response: Response,
    ) -> None:
        """Expose rate-limit metadata when a route has populated request state."""
        if not settings.RATE_LIMIT_ENABLED:
            return

        remaining = getattr(request.state, "rate_limit_remaining", None)
        limit = getattr(request.state, "rate_limit_limit", None)
        reset_at = getattr(request.state, "rate_limit_reset_at", None)

        if limit is not None:
            response.headers.setdefault(
                RATE_LIMIT_LIMIT_HEADER,
                str(limit),
            )

        if remaining is not None:
            response.headers.setdefault(
                RATE_LIMIT_REMAINING_HEADER,
                str(remaining),
            )

        if reset_at is not None:
            response.headers.setdefault(
                RATE_LIMIT_RESET_HEADER,
                str(reset_at),
            )

        if remaining == 0:
            response.headers.setdefault(
                RETRY_AFTER_HEADER,
                str(self._rate_limit_window_seconds),
            )


def logging_level_for_status(status_code: int) -> int:
    """Map an HTTP status code to an appropriate log level."""
    if status_code >= 500:
        return logging.ERROR

    if status_code >= 400:
        return logging.WARNING

    return logging.INFO