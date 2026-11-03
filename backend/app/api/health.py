"""
Health and readiness API routes.

These routes expose liveness, readiness, dependency diagnostics, version
information, and uptime metadata. They are intentionally unauthenticated so
load balancers, Kubernetes probes, and monitoring systems can use them.
"""

from __future__ import annotations

import platform
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    Response,
    status,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    DatabaseSession,
)
from app.core.config import (
    settings,
)


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


SERVICE_STARTED_AT = (
    datetime.now(timezone.utc)
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _elapsed_ms(
    started_at: float,
) -> float:
    """Return elapsed time in milliseconds."""
    return round(
        (
            time.perf_counter()
            - started_at
        )
        * 1_000,
        2,
    )


def _overall_status(
    checks: dict[str, str],
) -> str:
    """Derive an overall health status from dependency checks."""
    if not checks:
        return "healthy"

    if all(
        check_status == "healthy"
        for check_status in checks.values()
    ):
        return "healthy"

    if any(
        check_status == "unhealthy"
        for check_status in checks.values()
    ):
        return "unhealthy"

    return "degraded"


def _service_uptime_seconds() -> int:
    """Return approximate service uptime in whole seconds."""
    return int(
        (
            datetime.now(timezone.utc)
            - SERVICE_STARTED_AT
        ).total_seconds()
    )


async def _check_database(
    db: AsyncSession,
) -> str:
    """Check database connectivity with a lightweight query."""
    try:
        await db.execute(
            text("SELECT 1")
        )
        return "healthy"
    except Exception:
        return "unhealthy"


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------


@router.get(
    "",
    summary="Liveness probe",
    description=(
        "Returns basic service liveness. This endpoint does not check "
        "external dependencies."
    ),
)
async def health_check() -> dict[
    str, str
]:
    """Return basic service liveness."""
    return {
        "status": "healthy",
    }


@router.get(
    "/live",
    summary="Liveness probe alias",
    description=(
        "Alias for the primary liveness endpoint, suitable for container "
        "orchestrators."
    ),
)
async def liveness_check() -> dict[
    str, str
]:
    """Return basic service liveness."""
    return {
        "status": "healthy",
    }


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


@router.get(
    "/ready",
    summary="Readiness probe",
    description=(
        "Verifies that critical dependencies are reachable before routing "
        "production traffic to this instance."
    ),
)
async def readiness_check(
    db: DatabaseSession,
    response: Response,
) -> dict[str, Any]:
    """Verify critical dependencies are reachable."""
    started_at = (
        time.perf_counter()
    )

    checks: dict[str, str] = {}

    checks[
        "database"
    ] = await _check_database(
        db
    )

    elapsed_ms = _elapsed_ms(
        started_at
    )

    overall_status = _overall_status(
        checks
    )

    if overall_status != "healthy":
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return {
        "status": overall_status,
        "checks": checks,
        "elapsed_ms": elapsed_ms,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ---------------------------------------------------------------------------
# Detailed diagnostics
# ---------------------------------------------------------------------------


@router.get(
    "/details",
    summary="Detailed health diagnostics",
    description=(
        "Returns detailed health diagnostics without exposing secrets or "
        "connection strings."
    ),
)
async def detailed_health_check(
    db: DatabaseSession,
    response: Response,
) -> dict[str, Any]:
    """Return detailed health diagnostics without secrets."""
    started_at = (
        time.perf_counter()
    )

    database_status = (
        await _check_database(
            db
        )
    )

    checks = {
        "api": "healthy",
        "database": database_status,
    }

    elapsed_ms = _elapsed_ms(
        started_at
    )

    overall_status = _overall_status(
        checks
    )

    if overall_status != "healthy":
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return {
        "status": overall_status,
        "checks": checks,
        "elapsed_ms": elapsed_ms,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "service": {
            "name": settings.PROJECT_NAME,
            "environment": settings.ENVIRONMENT,
            "version": settings.VERSION,
            "uptime_seconds": _service_uptime_seconds(),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.system(),
        },
    }


# ---------------------------------------------------------------------------
# Service metadata
# ---------------------------------------------------------------------------


@router.get(
    "/version",
    summary="Service version",
    description=(
        "Returns non-sensitive service version and environment metadata."
    ),
)
async def get_service_version() -> dict[
    str, str
]:
    """Return service version and environment metadata."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
    }


@router.get(
    "/uptime",
    summary="Service uptime",
    description=(
        "Returns approximate service uptime since process startup."
    ),
)
async def get_service_uptime() -> dict[
    str, Any
]:
    """Return approximate service uptime."""
    uptime_seconds = (
        _service_uptime_seconds()
    )

    return {
        "uptime_seconds": uptime_seconds,
        "started_at": SERVICE_STARTED_AT.isoformat(),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }