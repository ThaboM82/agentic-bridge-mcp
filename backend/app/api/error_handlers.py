"""
Global FastAPI exception handlers for consistent error responses.

This module maps application, validation, HTTP, database, rate-limit, and
unexpected exceptions to a stable JSON error envelope. It also adds
request-correlation metadata and avoids leaking internal exception details to
clients.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import (
    DBAPIError,
    IntegrityError,
    OperationalError,
    SQLAlchemyError,
)
from starlette.exceptions import (
    HTTPException as StarletteHTTPException,
)

from app.core.exceptions import (
    AppError,
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    ValidationError,
)
from app.core.logging import (
    get_logger,
)


logger = get_logger(
    __name__
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_request_id(
    request: Request,
) -> str:
    """Return the incoming request ID or generate a correlation ID."""
    request_id = request.headers.get(
        "x-request-id"
    )

    if request_id:
        return request_id

    return str(
        uuid.uuid4()
    )


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    retryable: bool,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    """Build a consistent JSON error response."""
    error_payload: dict[str, Any] = {
        "code": code,
        "message": message,
        "retryable": retryable,
    }

    if details:
        error_payload[
            "details"
        ] = details

    if request_id:
        error_payload[
            "request_id"
        ] = request_id

    response_headers = dict(
        headers or {}
    )

    if request_id:
        response_headers.setdefault(
            "X-Request-ID",
            request_id,
        )

    return JSONResponse(
        status_code=status_code,
        content={
            "error": error_payload,
        },
        headers=response_headers,
    )


def _validation_error_details(
    exc: RequestValidationError,
) -> list[dict[str, Any]]:
    """Convert FastAPI validation errors into a client-safe structure."""
    errors: list[dict[str, Any]] = []

    for error in exc.errors():
        field_location = ".".join(
            str(part)
            for part in error.get(
                "loc",
                []
            )
        )

        errors.append(
            {
                "field": field_location,
                "message": error.get(
                    "msg",
                    "Invalid value.",
                ),
                "type": error.get(
                    "type",
                    "value_error",
                ),
            }
        )

    return errors


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------


def register_error_handlers(
    app: FastAPI,
) -> None:
    """Register application-wide exception handlers."""

    # -----------------------------------------------------------------------
    # Application errors
    # -----------------------------------------------------------------------

    @app.exception_handler(
        AppError
    )
    async def handle_app_error(
        request: Request,
        exc: AppError,
    ) -> JSONResponse:
        """Handle known application exceptions."""
        request_id = _get_request_id(
            request
        )

        logger.warning(
            "application_error",
            extra={
                "request_id": request_id,
                "error_code": exc.code,
                "status_code": exc.http_status,
                "path": request.url.path,
                "method": request.method,
                "details": exc.details,
            },
        )

        response_payload = exc.to_dict()

        if request_id:
            response_payload["error"][
                "request_id"
            ] = request_id

        return JSONResponse(
            status_code=exc.http_status,
            content=response_payload,
            headers=exc.headers or {},
        )

    @app.exception_handler(
        AuthenticationError
    )
    async def handle_authentication_error(
        request: Request,
        exc: AuthenticationError,
    ) -> JSONResponse:
        """Handle authentication failures."""
        request_id = _get_request_id(
            request
        )

        logger.warning(
            "authentication_error",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
            },
        )

        return _error_response(
            status_code=401,
            code="authentication_error",
            message=(
                exc.message
                or "Authentication is required."
            ),
            retryable=False,
            headers=exc.headers
            or {
                "WWW-Authenticate": "Bearer"
            },
            request_id=request_id,
        )

    @app.exception_handler(
        AuthorizationError
    )
    async def handle_authorization_error(
        request: Request,
        exc: AuthorizationError,
    ) -> JSONResponse:
        """Handle authorization and tenant-isolation failures."""
        request_id = _get_request_id(
            request
        )

        logger.warning(
            "authorization_error",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
            },
        )

        return _error_response(
            status_code=403,
            code="authorization_error",
            message=(
                exc.message
                or "You do not have permission to perform this action."
            ),
            retryable=False,
            headers=exc.headers,
            request_id=request_id,
        )

    @app.exception_handler(
        NotFoundError
    )
    async def handle_not_found_error(
        request: Request,
        exc: NotFoundError,
    ) -> JSONResponse:
        """Handle missing resources."""
        request_id = _get_request_id(
            request
        )

        logger.info(
            "resource_not_found",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
            },
        )

        return _error_response(
            status_code=404,
            code="not_found",
            message=(
                exc.message
                or "The requested resource was not found."
            ),
            retryable=False,
            headers=exc.headers,
            request_id=request_id,
        )

    @app.exception_handler(
        ConflictError
    )
    async def handle_conflict_error(
        request: Request,
        exc: ConflictError,
    ) -> JSONResponse:
        """Handle state or uniqueness conflicts."""
        request_id = _get_request_id(
            request
        )

        logger.warning(
            "conflict_error",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "details": exc.details,
            },
        )

        return _error_response(
            status_code=409,
            code="conflict",
            message=(
                exc.message
                or "The request conflicts with the current resource state."
            ),
            retryable=False,
            details=exc.details,
            headers=exc.headers,
            request_id=request_id,
        )

    @app.exception_handler(
        ValidationError
    )
    async def handle_domain_validation_error(
        request: Request,
        exc: ValidationError,
    ) -> JSONResponse:
        """Handle domain-level validation failures."""
        request_id = _get_request_id(
            request
        )

        logger.warning(
            "domain_validation_error",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "details": exc.details,
            },
        )

        return _error_response(
            status_code=422,
            code="validation_error",
            message=(
                exc.message
                or "The request contains invalid data."
            ),
            retryable=False,
            details=exc.details,
            headers=exc.headers,
            request_id=request_id,
        )

    @app.exception_handler(
        RateLimitError
    )
    async def handle_rate_limit_error(
        request: Request,
        exc: RateLimitError,
    ) -> JSONResponse:
        """Handle rate-limit rejections."""
        request_id = _get_request_id(
            request
        )

        retry_after = str(
            exc.retry_after_seconds
            or 60
        )

        logger.warning(
            "rate_limit_error",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "retry_after_seconds": exc.retry_after_seconds,
            },
        )

        return _error_response(
            status_code=429,
            code="rate_limit_exceeded",
            message=(
                exc.message
                or "Too many requests. Please retry later."
            ),
            retryable=True,
            headers=exc.headers
            or {
                "Retry-After": retry_after
            },
            request_id=request_id,
        )

    @app.exception_handler(
        ServiceUnavailableError
    )
    async def handle_service_unavailable_error(
        request: Request,
        exc: ServiceUnavailableError,
    ) -> JSONResponse:
        """Handle downstream-service failures."""
        request_id = _get_request_id(
            request
        )

        logger.error(
            "service_unavailable_error",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "details": exc.details,
            },
        )

        return _error_response(
            status_code=503,
            code="service_unavailable",
            message=(
                exc.message
                or "A required service is temporarily unavailable."
            ),
            retryable=True,
            details=exc.details,
            headers=exc.headers,
            request_id=request_id,
        )

    # -----------------------------------------------------------------------
    # Request validation errors
    # -----------------------------------------------------------------------

    @app.exception_handler(
        RequestValidationError
    )
    async def handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Handle malformed or invalid request payloads."""
        request_id = _get_request_id(
            request
        )
        errors = _validation_error_details(
            exc
        )

        logger.warning(
            "request_validation_error",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "error_count": len(
                    errors
                ),
            },
        )

        return _error_response(
            status_code=422,
            code="validation_error",
            message="The request contains invalid data.",
            retryable=False,
            details={
                "errors": errors,
            },
            request_id=request_id,
        )

    # -----------------------------------------------------------------------
    # HTTP errors
    # -----------------------------------------------------------------------

    @app.exception_handler(
        StarletteHTTPException
    )
    async def handle_http_exception(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        """Handle HTTP exceptions raised by FastAPI or Starlette."""
        request_id = _get_request_id(
            request
        )

        logger.warning(
            "http_error",
            extra={
                "request_id": request_id,
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method,
            },
        )

        return _error_response(
            status_code=exc.status_code,
            code="http_error",
            message=str(
                exc.detail
            ),
            retryable=exc.status_code
            >= 500,
            headers=getattr(
                exc,
                "headers",
                None,
            )
            or {},
            request_id=request_id,
        )

    # -----------------------------------------------------------------------
    # Database errors
    # -----------------------------------------------------------------------

    @app.exception_handler(
        IntegrityError
    )
    async def handle_integrity_error(
        request: Request,
        exc: IntegrityError,
    ) -> JSONResponse:
        """Handle database constraint and uniqueness violations."""
        request_id = _get_request_id(
            request
        )

        logger.warning(
            "database_integrity_error",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
            },
        )

        return _error_response(
            status_code=409,
            code="resource_conflict",
            message=(
                "The request violates a database constraint."
            ),
            retryable=False,
            request_id=request_id,
        )

    @app.exception_handler(
        OperationalError
    )
    async def handle_operational_error(
        request: Request,
        exc: OperationalError,
    ) -> JSONResponse:
        """Handle transient database connectivity or timeout failures."""
        request_id = _get_request_id(
            request
        )

        logger.error(
            "database_operational_error",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
            },
        )

        return _error_response(
            status_code=503,
            code="database_unavailable",
            message=(
                "The database is temporarily unavailable."
            ),
            retryable=True,
            headers={
                "Retry-After": "5"
            },
            request_id=request_id,
        )

    @app.exception_handler(
        DBAPIError
    )
    async def handle_dbapi_error(
        request: Request,
        exc: DBAPIError,
    ) -> JSONResponse:
        """Handle lower-level database driver failures."""
        request_id = _get_request_id(
            request
        )

        logger.error(
            "database_error",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
            },
        )

        return _error_response(
            status_code=500,
            code="database_error",
            message=(
                "A database error occurred while processing the request."
            ),
            retryable=False,
            request_id=request_id,
        )

    @app.exception_handler(
        SQLAlchemyError
    )
    async def handle_sqlalchemy_error(
        request: Request,
        exc: SQLAlchemyError,
    ) -> JSONResponse:
        """Handle unclassified SQLAlchemy failures."""
        request_id = _get_request_id(
            request
        )

        logger.exception(
            "sqlalchemy_error",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
            },
        )

        return _error_response(
            status_code=500,
            code="database_error",
            message=(
                "A database error occurred while processing the request."
            ),
            retryable=False,
            request_id=request_id,
        )

    # -----------------------------------------------------------------------
    # Unexpected errors
    # -----------------------------------------------------------------------

    @app.exception_handler(
        Exception
    )
    async def handle_unexpected_error(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle unexpected exceptions without leaking internals."""
        request_id = _get_request_id(
            request
        )
        started_at = getattr(
            request.state,
            "started_at",
            None,
        )

        duration_ms = (
            round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000,
                2,
            )
            if isinstance(
                started_at,
                float
            )
            else None
        )

        logger.exception(
            "unhandled_exception",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "duration_ms": duration_ms,
                "exception_type": type(
                    exc
                ).__name__,
            },
        )

        return _error_response(
            status_code=500,
            code="internal_error",
            message="An unexpected error occurred.",
            retryable=False,
            request_id=request_id,
        )