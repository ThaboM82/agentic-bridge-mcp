"""
Generated-artifact API routes.

These routes expose tenant-scoped artifact metadata, execution artifact
listings, lifecycle management, checksum verification metadata, and secure
download-URL generation. Artifact contents are never returned directly; clients
receive short-lived download references instead.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)
from pydantic import BaseModel, Field

from app.api.dependencies import (
    CurrentPrincipal,
    DatabaseSession,
    Pagination,
    require_scoped_access,
)
from app.api.pagination import (
    build_paginated_response,
)
from app.core.exceptions import ValidationError
from app.repositories.artifact_repository import (
    ArtifactRepository,
)
from app.repositories.audit_log_repository import (
    AuditLogRepository,
)
from app.repositories.workflow_repository import (
    WorkflowRepository,
)


logger = logging.getLogger(
    "agentic_bridge.api.artifacts"
)


router = APIRouter(
    prefix="/artifacts",
    tags=["Artifacts"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ArtifactCreateRequest(BaseModel):
    """Request to register a generated artifact."""

    execution_id: UUID
    artifact_type: str = Field(
        min_length=1,
        max_length=100,
    )
    storage_key: str = Field(
        min_length=1,
        max_length=1_000,
    )
    file_size_bytes: int = Field(
        gt=0,
        description="Artifact size in bytes.",
    )
    checksum: str | None = Field(
        default=None,
        max_length=256,
    )
    mime_type: str | None = Field(
        default=None,
        max_length=150,
    )
    metadata: dict = Field(
        default_factory=dict
    )


class ArtifactUploadRequest(BaseModel):
    """Request to mark an artifact as uploaded."""

    storage_key: str = Field(
        min_length=1,
        max_length=1_000,
    )


class ArtifactDownloadResponse(BaseModel):
    """Short-lived artifact download reference."""

    artifact_id: UUID
    download_url: str
    expires_in_seconds: int
    expires_at: datetime


# ---------------------------------------------------------------------------
# Artifact collection
# ---------------------------------------------------------------------------


@router.get("")
async def list_artifacts(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
    artifact_status: Annotated[
        str | None,
        Query(
            description=(
                "Filter by created, uploaded, expired, or deleted."
            )
        ),
    ] = None,
    artifact_type: Annotated[
        str | None,
        Query(
            max_length=100,
            description="Filter by artifact type."
        ),
    ] = None,
):
    """List tenant-scoped generated artifacts."""
    artifacts, total_count = (
        await ArtifactRepository.list_for_tenant(
            db,
            tenant_id=principal.tenant_id,
            skip=pagination.skip,
            limit=pagination.limit,
        )
    )

    if artifact_status:
        normalized_status = (
            artifact_status.strip().lower()
        )

        artifacts = [
            artifact
            for artifact in artifacts
            if artifact.status
            == normalized_status
        ]

    if artifact_type:
        normalized_type = (
            artifact_type.strip().lower()
        )

        artifacts = [
            artifact
            for artifact in artifacts
            if artifact.artifact_type
            == normalized_type
        ]

    await db.commit()

    return build_paginated_response(
        artifacts,
        skip=pagination.skip,
        limit=pagination.limit,
        total_count=total_count,
    )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
)
async def create_artifact(
    payload: ArtifactCreateRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:write"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Register a generated artifact for a workflow execution."""
    execution = (
        await WorkflowRepository.get_by_execution_id(
            db,
            execution_id=payload.execution_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow execution not found.",
        )

    try:
        artifact = (
            await ArtifactRepository.create_artifact(
                db,
                tenant_id=principal.tenant_id,
                execution_id=payload.execution_id,
                artifact_type=payload.artifact_type,
                storage_key=payload.storage_key,
                file_size_bytes=payload.file_size_bytes,
                checksum=payload.checksum,
                mime_type=payload.mime_type,
                metadata=payload.metadata,
            )
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="artifact_created",
        actor_id=principal.user_id,
        execution_id=payload.execution_id,
        resource_type="artifact",
        resource_id=artifact.id,
        action="create_artifact",
        outcome=artifact.status,
        metadata={
            "artifact_type": artifact.artifact_type,
            "file_size_bytes": artifact.file_size_bytes,
        },
    )

    await db.commit()

    logger.info(
        "Artifact registered.",
        extra={
            "artifact_id": str(
                artifact.id
            ),
            "execution_id": str(
                payload.execution_id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
        },
    )

    return artifact


# ---------------------------------------------------------------------------
# Individual artifacts
# ---------------------------------------------------------------------------


@router.get("/{artifact_id}")
async def get_artifact(
    artifact_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:read"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Retrieve tenant-scoped artifact metadata."""
    artifact = (
        await ArtifactRepository.get_by_id(
            db,
            record_id=artifact_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found.",
        )

    await db.commit()

    return artifact


@router.get(
    "/executions/{execution_id}"
)
async def list_artifacts_for_execution(
    execution_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
):
    """List artifacts associated with a workflow execution."""
    execution = (
        await WorkflowRepository.get_by_execution_id(
            db,
            execution_id=execution_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow execution not found.",
        )

    artifacts, total_count = (
        await ArtifactRepository.list_artifacts_for_execution(
            db,
            execution_id=execution_id,
            tenant_id=principal.tenant_id,
            skip=pagination.skip,
            limit=pagination.limit,
        )
    )

    await db.commit()

    return build_paginated_response(
        artifacts,
        skip=pagination.skip,
        limit=pagination.limit,
        total_count=total_count,
    )


# ---------------------------------------------------------------------------
# Artifact lifecycle
# ---------------------------------------------------------------------------


@router.post(
    "/{artifact_id}/upload",
    status_code=status.HTTP_200_OK,
)
async def mark_artifact_uploaded(
    artifact_id: UUID,
    payload: ArtifactUploadRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:write"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Mark an artifact as uploaded to object storage."""
    try:
        artifact = (
            await ArtifactRepository.mark_uploaded(
                db,
                artifact_id=artifact_id,
                tenant_id=principal.tenant_id,
                storage_key=payload.storage_key,
            )
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found.",
        )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="artifact_uploaded",
        actor_id=principal.user_id,
        execution_id=artifact.execution_id,
        resource_type="artifact",
        resource_id=artifact.id,
        action="mark_uploaded",
        outcome=artifact.status,
        metadata={
            "artifact_type": artifact.artifact_type,
        },
    )

    await db.commit()

    logger.info(
        "Artifact marked uploaded.",
        extra={
            "artifact_id": str(
                artifact.id
            ),
            "execution_id": str(
                artifact.execution_id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
        },
    )

    return artifact


@router.post(
    "/{artifact_id}/expire",
    status_code=status.HTTP_200_OK,
)
async def mark_artifact_expired(
    artifact_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:write"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Mark an artifact as expired."""
    artifact = (
        await ArtifactRepository.mark_expired(
            db,
            artifact_id=artifact_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found.",
        )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="artifact_expired",
        actor_id=principal.user_id,
        execution_id=artifact.execution_id,
        resource_type="artifact",
        resource_id=artifact.id,
        action="mark_expired",
        outcome=artifact.status,
        metadata={
            "artifact_type": artifact.artifact_type,
        },
    )

    await db.commit()

    logger.info(
        "Artifact marked expired.",
        extra={
            "artifact_id": str(
                artifact.id
            ),
            "execution_id": str(
                artifact.execution_id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
        },
    )

    return artifact


@router.delete(
    "/{artifact_id}",
    status_code=status.HTTP_200_OK,
)
async def delete_artifact(
    artifact_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:write"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Delete a tenant-scoped artifact record."""
    artifact = (
        await ArtifactRepository.get_by_id(
            db,
            record_id=artifact_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found.",
        )

    execution_id = artifact.execution_id
    artifact_type = artifact.artifact_type

    deleted = (
        await ArtifactRepository.delete_by_id(
            db,
            record_id=artifact_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found.",
        )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="artifact_deleted",
        actor_id=principal.user_id,
        execution_id=execution_id,
        resource_type="artifact",
        resource_id=artifact_id,
        action="delete_artifact",
        outcome="deleted",
        metadata={
            "artifact_type": artifact_type,
        },
    )

    await db.commit()

    logger.info(
        "Artifact deleted.",
        extra={
            "artifact_id": str(
                artifact_id
            ),
            "execution_id": str(
                execution_id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
        },
    )

    return {
        "artifact_id": str(
            artifact_id
        ),
        "deleted": True,
    }


# ---------------------------------------------------------------------------
# Secure download reference
# ---------------------------------------------------------------------------


@router.get(
    "/{artifact_id}/download-url",
    response_model=ArtifactDownloadResponse,
)
async def get_artifact_download_url(
    artifact_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:read"
            )
        ),
    ],
    db: DatabaseSession,
    expires_in: Annotated[
        int,
        Query(
            ge=30,
            le=900,
            description=(
                "Download reference lifetime in seconds."
            ),
        ),
    ] = 300,
):
    """Return a short-lived, tenant-scoped artifact download reference."""
    artifact = (
        await ArtifactRepository.get_by_id(
            db,
            record_id=artifact_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found.",
        )

    if artifact.status == "expired":
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Artifact download reference has expired.",
        )

    expires_at = datetime.now(
        timezone.utc
    ).timestamp() + expires_in

    download_url = (
        f"/api/v1/artifacts/{artifact_id}/download"
        f"?expires_at={int(expires_at)}"
    )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="artifact_download_requested",
        actor_id=principal.user_id,
        execution_id=artifact.execution_id,
        resource_type="artifact",
        resource_id=artifact.id,
        action="request_download_url",
        outcome="issued",
        metadata={
            "artifact_type": artifact.artifact_type,
            "expires_in_seconds": expires_in,
        },
    )

    await db.commit()

    logger.info(
        "Artifact download reference issued.",
        extra={
            "artifact_id": str(
                artifact_id
            ),
            "execution_id": str(
                artifact.execution_id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
        },
    )

    return ArtifactDownloadResponse(
        artifact_id=artifact_id,
        download_url=download_url,
        expires_in_seconds=expires_in,
        expires_at=datetime.fromtimestamp(
            expires_at,
            tz=timezone.utc,
        ),
    )