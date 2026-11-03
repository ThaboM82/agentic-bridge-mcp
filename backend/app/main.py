"""
AgenticBridge-MCP FastAPI application.

This module configures application lifecycle, database initialization, CORS,
security headers, request tracing, structured logging, exception handling,
API routing, and system health endpoints.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api import (
    approvals_router,
    artifacts_router,
    audit_router,
    briefs_router,
    documents_router,
    health_router,
    retrieval_router,
    usage_router,
    users_router,
    workflow_router,
)

# Importing app.models registers every SQLAlchemy model with Base.metadata.
from app.models import (  # noqa: F401
    ApiKey,
    Approval,
    ApprovalDelegation,
    ApprovalPolicy,
    Artifact,
    ArtifactDownloadToken,
    ArtifactVersion,
    AuditEvent,
    AuditRetentionPolicy,
    Brief,
    BriefTemplate,
    BriefVersion,
    Document,
    DocumentChunk,
    DocumentEmbedding,
    DocumentVersion,
    EmbeddingQueryLog,
    IngestionJob,
    IngestionJobItem,
    Integration,
    IntegrationCredential,
    IntegrationSyncState,
    Notification,
    NotificationDelivery,
    NotificationPreference,
    RetentionPolicy,
    RetentionJob,
    RetrievalExperiment,
    SavedSearch,
    SearchFeedback,
    Tenant,
    TenantInvitation,
    TenantSettings,
    UsageEvent,
    UsageQuota,
    UsageRateCard,
    User,
    UserSession,
    WebhookDelivery,
    WebhookEndpoint,
    WebhookEvent,
    WorkflowApprovalModel,
    WorkflowArtifactModel,
    WorkflowAuditEventModel,
    WorkflowExecutionModel,
    WorkflowStepModel,
)

from app.core.config import settings
from app.core.database import (
    Base,
    check_database_connection,
    dispose_engine,
    engine,
)

try:
    from app.services.llm_client import LLMProviderFactory
except ImportError:
    LLMProviderFactory = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(
        logging,
        settings.LOG_LEVEL.upper(),
        logging.INFO,
    ),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("agentic_bridge.main")

logging.getLogger("uvicorn.access").setLevel(
    logging.WARNING,
)

logging.getLogger("sqlalchemy.engine").setLevel(
    logging.WARNING
    if settings.is_production
    else logging.INFO,
)


API_V1_PREFIX = settings.API_V1_PREFIX
API_TITLE = "AgenticBridge-MCP API"
API_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.

    Startup:
    - Enables pgvector.
    - Creates database tables.
    - Logs LLM provider configuration.

    Shutdown:
    - Disposes database connections.
    """
    logger.info("Starting AgenticBridge-MCP backend...")

    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "CREATE EXTENSION IF NOT EXISTS vector;"
                )
            )

            await connection.run_sync(
                Base.metadata.create_all,
            )

        logger.info(
            "Database schema and pgvector extension initialized.",
        )

    except SQLAlchemyError as exc:
        logger.exception(
            "Database initialization failed.",
        )

        raise RuntimeError(
            "Database initialization failed.",
        ) from exc

    if LLMProviderFactory and hasattr(
        LLMProviderFactory,
        "get_client_metadata",
    ):
        try:
            metadata = (
                LLMProviderFactory.get_client_metadata()
            )

            logger.info(
                "LLM provider configuration: %s",
                metadata,
            )

        except Exception:
            logger.exception(
                "Unable to read LLM provider metadata.",
            )

    else:
        logger.info(
            "LLM provider factory not configured.",
        )

    yield

    logger.info(
        "Shutting down AgenticBridge-MCP backend...",
    )

    await dispose_engine()

    logger.info(
        "Database connections disposed.",
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


app = FastAPI(
    title=API_TITLE,
    description=(
        "Enterprise-grade autonomous orchestration platform for design-to-code "
        "and content-publishing workflows, powered by FastAPI, LangGraph, MCP, "
        "and PostgreSQL pgvector."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None,
    openapi_url=(
        "/openapi.json"
        if settings.ENABLE_DOCS
        else None
    ),
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
    ],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "X-Requested-With",
        "X-Request-ID",
        "X-CSRF-Token",
    ],
    expose_headers=[
        "X-Process-Time",
        "X-Request-ID",
    ],
    max_age=600,
)

app.add_middleware(
    GZipMiddleware,
    minimum_size=1_024,
)


@app.middleware("http")
async def request_context_middleware(
    request: Request,
    call_next,
):
    """
    Add request ID, processing latency, and baseline security headers.
    """
    request_id = (
        request.headers.get("X-Request-ID")
        or f"req_{uuid.uuid4().hex}"
    )

    request.state.request_id = request_id
    request.state.start_time = time.perf_counter()

    response = await call_next(request)

    process_time = (
        time.perf_counter()
        - request.state.start_time
    )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = (
        f"{process_time:.4f}"
    )
    response.headers["X-Content-Type-Options"] = (
        "nosniff"
    )
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = (
        "strict-origin-when-cross-origin"
    )
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )

    if settings.is_production:
        response.headers[
            "Strict-Transport-Security"
        ] = (
            "max-age=31536000; includeSubDomains"
        )

    if settings.LOG_REQUESTS:
        logger.info(
            "%s %s completed with %s in %.4fs [%s]",
            request.method,
            request.url.path,
            response.status_code,
            process_time,
            request_id,
        )

    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(SQLAlchemyError)
async def database_exception_handler(
    request: Request,
    exc: SQLAlchemyError,
):
    """
    Prevent database internals from leaking to API clients.
    """
    logger.exception(
        "Database error on %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=(
            status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        content={
            "error": "database_unavailable",
            "message": (
                "The service is temporarily unable "
                "to access its database."
            ),
            "request_id": (
                request.headers.get("X-Request-ID")
            ),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """
    Return a consistent response for request-validation failures.
    """
    logger.warning(
        "Validation error on %s %s: %s",
        request.method,
        request.url.path,
        exc.errors(),
    )

    return JSONResponse(
        status_code=(
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ),
        content={
            "error": "validation_error",
            "message": (
                "The request payload is invalid."
            ),
            "details": exc.errors(),
            "request_id": (
                request.headers.get("X-Request-ID")
            ),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):
    """
    Catch-all handler for unexpected server errors.
    """
    logger.exception(
        "Unhandled error on %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=(
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ),
        content={
            "error": "internal_server_error",
            "message": (
                "An unexpected error occurred."
            ),
            "request_id": (
                request.headers.get("X-Request-ID")
            ),
        },
    )


# ---------------------------------------------------------------------------
# API routers
# ---------------------------------------------------------------------------

app.include_router(
    approvals_router,
    prefix=API_V1_PREFIX,
)

app.include_router(
    artifacts_router,
    prefix=API_V1_PREFIX,
)

app.include_router(
    audit_router,
    prefix=API_V1_PREFIX,
)

app.include_router(
    briefs_router,
    prefix=API_V1_PREFIX,
)

app.include_router(
    documents_router,
    prefix=API_V1_PREFIX,
)

app.include_router(
    health_router,
    prefix=API_V1_PREFIX,
)

app.include_router(
    retrieval_router,
    prefix=API_V1_PREFIX,
)

app.include_router(
    usage_router,
    prefix=API_V1_PREFIX,
)

app.include_router(
    users_router,
    prefix=API_V1_PREFIX,
)

app.include_router(
    workflow_router,
    prefix=API_V1_PREFIX,
)


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/",
    tags=["System"],
)
async def root() -> dict[str, Any]:
    """Service overview."""
    return {
        "system": "AgenticBridge-MCP",
        "status": "operational",
        "version": app.version,
        "environment": settings.APP_ENV,
        "documentation": "/docs",
        "openapi": "/openapi.json",
        "api_prefix": API_V1_PREFIX,
        "workflow_endpoint": (
            f"{API_V1_PREFIX}/workflow"
        ),
        "documents_endpoint": (
            f"{API_V1_PREFIX}/documents"
        ),
    }


@app.get(
    "/health",
    tags=["System"],
)
async def liveness_check() -> dict[str, Any]:
    """
    Liveness probe.

    Returns success if the API process is running. It intentionally does not
    check external dependencies so orchestrators can distinguish a dead process
    from a dependency outage.
    """
    return {
        "status": "healthy",
        "service": "agentic-bridge-mcp",
        "version": app.version,
        "environment": settings.APP_ENV,
    }


@app.get(
    "/ready",
    tags=["System"],
)
async def readiness_check() -> dict[str, Any]:
    """
    Readiness probe.

    Verifies database connectivity. Add checks for Redis, vector store,
    Figma, CMS, and WordPress integrations as they become required for
    serving traffic.
    """
    checks: dict[str, Any] = {}
    ready = True

    database_ready = (
        await check_database_connection()
    )

    checks["database"] = {
        "status": (
            "healthy"
            if database_ready
            else "unhealthy"
        ),
    }

    if not database_ready:
        ready = False

    if LLMProviderFactory and hasattr(
        LLMProviderFactory,
        "get_client_metadata",
    ):
        try:
            metadata = (
                LLMProviderFactory.get_client_metadata()
            )

            checks["llm"] = {
                "status": "healthy",
                "providers": metadata,
            }

        except Exception:
            checks["llm"] = {
                "status": "degraded",
                "error": (
                    "LLM provider metadata "
                    "unavailable."
                ),
            }

    return {
        "status": (
            "ready"
            if ready
            else "not_ready"
        ),
        "service": "agentic-bridge-mcp",
        "version": app.version,
        "environment": settings.APP_ENV,
        "checks": checks,
    }


@app.get(
    "/metrics",
    tags=["System"],
    include_in_schema=False,
)
async def metrics() -> dict[str, Any]:
    """
    Basic application metrics endpoint.

    Replace or extend this with Prometheus-compatible metrics in production.
    """
    return {
        "service": "agentic-bridge-mcp",
        "version": app.version,
        "environment": settings.APP_ENV,
        "api_prefix": API_V1_PREFIX,
    }