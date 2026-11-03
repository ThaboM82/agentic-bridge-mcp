"""
CRUD operations for ingestion jobs.

This module owns all direct persistence operations for asynchronous document
ingestion jobs. API routes, workflow services, and background workers should
call these functions rather than constructing SQLAlchemy queries directly.

Responsibilities:
- Create ingestion jobs.
- Retrieve jobs with tenant isolation.
- List and paginate jobs with filters.
- Transition job status safely.
- Track attempts, failures, completion, and cancellation.
- Map database failures to stable application exceptions.

Non-goals:
- Do not parse documents, generate embeddings, invoke LLMs, enqueue workers,
  or implement retry/backoff policy in this module.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import DocumentStatus, JobStatus
from app.core.exceptions import (
    DatabaseIntegrityError,
    NotFoundError,
    PersistenceError,
    ValidationError,
    WorkflowStateError,
)
from app.core.logging_context import bind_job_context
from app.core.pagination import Page, PaginationParams
from app.crud.base import (
    create_record,
    get_by_id,
    get_by_id_or_404,
    update_record,
)
from app.models.ingestion_job import IngestionJobModel


logger = logging.getLogger("agentic_bridge.crud.ingestion_job")

MAX_ERROR_MESSAGE_LENGTH = 2_000
MAX_RETRY_ATTEMPTS = 10

TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {
        JobStatus.COMPLETED,
        JobStatus.PARTIALLY_COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
)

ALLOWED_STATUS_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset(
        {
            JobStatus.PROCESSING,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
        }
    ),
    JobStatus.PROCESSING: frozenset(
        {
            JobStatus.COMPLETED,
            JobStatus.PARTIALLY_COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }
    ),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.PARTIALLY_COMPLETED: frozenset(
        {
            JobStatus.PROCESSING,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }
    ),
    JobStatus.FAILED: frozenset(
        {
            JobStatus.QUEUED,
            JobStatus.PROCESSING,
            JobStatus.CANCELLED,
        }
    ),
    JobStatus.CANCELLED: frozenset(),
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _normalize_job_id(job_id: str | UUID) -> UUID:
    """Validate and normalize an ingestion-job ID."""
    if isinstance(job_id, UUID):
        return job_id

    if not isinstance(job_id, str) or not job_id.strip():
        raise ValidationError("job_id must be a non-empty string.")

    try:
        return UUID(job_id.strip())
    except ValueError as exc:
        raise ValidationError(
            "job_id must be a valid UUID."
        ) from exc


def _normalize_job_status(
    status: str | JobStatus | None,
) -> JobStatus | None:
    """Validate and normalize an ingestion-job status."""
    if status is None:
        return None

    try:
        return JobStatus(status)
    except ValueError as exc:
        raise ValidationError(
            f"Unsupported ingestion job status: {status}."
        ) from exc


def _validate_status_transition(
    current_status: str | JobStatus,
    next_status: str | JobStatus,
) -> tuple[JobStatus, JobStatus]:
    """
    Validate a job-status transition.

    Args:
        current_status: Current persisted status.
        next_status: Requested status.

    Returns:
        Validated current and next status values.

    Raises:
        ValidationError: If either status is unsupported.
        WorkflowStateError: If the transition is not allowed.
    """
    current = _normalize_job_status(current_status)
    next = _normalize_job_status(next_status)

    if current is None or next is None:
        raise ValidationError(
            "Both current and next ingestion job statuses are required."
        )

    if current is next:
        return current, next

    allowed_transitions = ALLOWED_STATUS_TRANSITIONS.get(current, frozenset())

    if next not in allowed_transitions:
        raise WorkflowStateError(
            "Unsupported ingestion job status transition.",
            details={
                "current_status": current.value,
                "requested_status": next.value,
            },
        )

    return current, next


def _truncate_error_message(message: str | None) -> str | None:
    """Bound an error message before persistence."""
    if message is None:
        return None

    normalized = message.strip()

    if not normalized:
        return None

    return normalized[:MAX_ERROR_MESSAGE_LENGTH]


def _apply_tenant_filter(query, tenant_id: str | None):
    """Apply tenant isolation to an ingestion-job query."""
    if tenant_id:
        return query.where(IngestionJobModel.tenant_id == tenant_id)

    return query


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


async def create_ingestion_job(
    db: AsyncSession,
    *,
    document_id: str | UUID,
    tenant_id: str | None = None,
    created_by: str | None = None,
    job_type: str = "document_ingestion",
    priority: int = 5,
    max_attempts: int = 3,
    job_metadata: dict[str, Any] | None = None,
) -> IngestionJobModel:
    """
    Create and persist a new ingestion job.

    Args:
        db: Active async database session.
        document_id: Document to ingest.
        tenant_id: Owning tenant identifier.
        created_by: Authenticated user ID.
        job_type: Logical job type.
        priority: Job priority; lower values can be processed first.
        max_attempts: Maximum processing attempts.
        job_metadata: Non-sensitive job metadata.

    Returns:
        The persisted ingestion job.

    Raises:
        ValidationError: If values are invalid.
        DatabaseIntegrityError: If a constraint is violated.
        PersistenceError: If another database failure occurs.
    """
    document_uuid = _normalize_job_id(document_id)

    if max_attempts < 1 or max_attempts > MAX_RETRY_ATTEMPTS:
        raise ValidationError(
            f"max_attempts must be between 1 and {MAX_RETRY_ATTEMPTS}."
        )

    if priority < 0:
        raise ValidationError("priority must be non-negative.")

    job = IngestionJobModel(
        document_id=document_uuid,
        tenant_id=tenant_id,
        created_by=created_by,
        job_type=job_type,
        priority=priority,
        max_attempts=max_attempts,
        attempt_count=0,
        status=JobStatus.QUEUED.value,
        job_metadata=job_metadata or {},
    )

    job = await create_record(db, job)

    bind_job_context(
        str(job.id),
        workflow_id=None,
        tenant_id=tenant_id,
    )

    logger.info(
        "ingestion_job_created",
        job_id=str(job.id),
        document_id=str(document_uuid),
        tenant_id=tenant_id,
        job_type=job_type,
        priority=priority,
        max_attempts=max_attempts,
    )

    return job


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


async def get_ingestion_job(
    db: AsyncSession,
    job_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> IngestionJobModel | None:
    """
    Retrieve an ingestion job by ID.

    Args:
        db: Active async database session.
        job_id: Ingestion-job UUID.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The ingestion job, or `None` when it does not exist or belongs to
        another tenant.
    """
    job_uuid = _normalize_job_id(job_id)

    query = select(IngestionJobModel).where(
        IngestionJobModel.id == job_uuid
    )

    query = _apply_tenant_filter(query, tenant_id)

    result = await db.execute(query)

    return result.scalar_one_or_none()


async def get_ingestion_job_or_404(
    db: AsyncSession,
    job_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> IngestionJobModel:
    """
    Retrieve an ingestion job by ID or raise `NotFoundError`.

    Args:
        db: Active async database session.
        job_id: Ingestion-job UUID.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The matching ingestion job.

    Raises:
        NotFoundError: If the job does not exist or is outside the tenant.
    """
    job = await get_ingestion_job(
        db,
        job_id,
        tenant_id=tenant_id,
    )

    if job is None:
        raise NotFoundError(
            "Ingestion job was not found.",
            details={"job_id": str(job_id)},
        )

    return job


async def get_jobs_by_document(
    db: AsyncSession,
    document_id: str | UUID,
    *,
    tenant_id: str | None = None,
    status: str | JobStatus | None = None,
) -> Sequence[IngestionJobModel]:
    """
    Retrieve ingestion jobs associated with a document.

    Args:
        db: Active async database session.
        document_id: Document UUID.
        tenant_id: Optional tenant isolation filter.
        status: Optional status filter.

    Returns:
        A sequence of ingestion jobs, newest first.
    """
    document_uuid = _normalize_job_id(document_id)
    safe_status = _normalize_job_status(status)

    query = select(IngestionJobModel).where(
        IngestionJobModel.document_id == document_uuid
    )

    query = _apply_tenant_filter(query, tenant_id)

    if safe_status is not None:
        query = query.where(IngestionJobModel.status == safe_status.value)

    query = query.order_by(IngestionJobModel.created_at.desc())

    result = await db.execute(query)

    return result.scalars().all()


async def list_ingestion_jobs(
    db: AsyncSession,
    *,
    pagination: PaginationParams,
    tenant_id: str | None = None,
    status: str | JobStatus | None = None,
    document_id: str | UUID | None = None,
    created_by: str | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
) -> Page[IngestionJobModel]:
    """
    Retrieve a paginated, filtered list of ingestion jobs.

    Args:
        db: Active async database session.
        pagination: Validated pagination parameters.
        tenant_id: Optional tenant isolation filter.
        status: Optional status filter.
        document_id: Optional document filter.
        created_by: Optional creator filter.
        created_after: Optional inclusive lower created-at bound.
        created_before: Optional inclusive upper created-at bound.

    Returns:
        A `Page` containing ingestion jobs and pagination metadata.
    """
    safe_status = _normalize_job_status(status)

    filters = []

    if tenant_id:
        filters.append(IngestionJobModel.tenant_id == tenant_id)

    if safe_status is not None:
        filters.append(IngestionJobModel.status == safe_status.value)

    if document_id is not None:
        filters.append(
            IngestionJobModel.document_id
            == _normalize_job_id(document_id)
        )

    if created_by:
        filters.append(IngestionJobModel.created_by == created_by)

    if created_after is not None:
        filters.append(IngestionJobModel.created_at >= created_after)

    if created_before is not None:
        filters.append(IngestionJobModel.created_at <= created_before)

    count_query = select(func.count()).select_from(IngestionJobModel)

    if filters:
        count_query = count_query.where(and_(*filters))

    count_result = await db.execute(count_query)
    total = int(count_result.scalar_one())

    query = select(IngestionJobModel).order_by(
        IngestionJobModel.priority.asc(),
        IngestionJobModel.created_at.desc(),
    )

    if filters:
        query = query.where(and_(*filters))

    result = await db.execute(
        query.offset(pagination.skip).limit(pagination.limit)
    )

    jobs = list(result.scalars().all())

    return Page[IngestionJobModel].create(
        items=jobs,
        total=total,
        skip=pagination.skip,
        limit=pagination.limit,
    )


async def count_ingestion_jobs(
    db: AsyncSession,
    *,
    tenant_id: str | None = None,
    status: str | JobStatus | None = None,
) -> int:
    """
    Count ingestion jobs matching optional filters.

    Args:
        db: Active async database session.
        tenant_id: Optional tenant isolation filter.
        status: Optional status filter.

    Returns:
        Total matching job count.
    """
    safe_status = _normalize_job_status(status)

    filters = []

    if tenant_id:
        filters.append(IngestionJobModel.tenant_id == tenant_id)

    if safe_status is not None:
        filters.append(IngestionJobModel.status == safe_status.value)

    query = select(func.count()).select_from(IngestionJobModel)

    if filters:
        query = query.where(and_(*filters))

    result = await db.execute(query)

    return int(result.scalar_one())


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


async def update_ingestion_job_status(
    db: AsyncSession,
    job_id: str | UUID,
    status: str | JobStatus,
    *,
    tenant_id: str | None = None,
    error_message: str | None = None,
    result_metadata: dict[str, Any] | None = None,
) -> IngestionJobModel:
    """
    Transition an ingestion job to a new status.

    Args:
        db: Active async database session.
        job_id: Ingestion-job UUID.
        status: Requested status.
        tenant_id: Optional tenant isolation filter.
        error_message: Optional bounded failure message.
        result_metadata: Optional non-sensitive processing result metadata.

    Returns:
        The refreshed ingestion job.

    Raises:
        ValidationError: If the status is unsupported.
        NotFoundError: If the job does not exist.
        WorkflowStateError: If the transition is not allowed.
        PersistenceError: If the update cannot be completed.
    """
    job = await get_ingestion_job_or_404(
        db,
        job_id,
        tenant_id=tenant_id,
    )

    current_status, next_status = _validate_status_transition(
        job.status,
        status,
    )

    previous_status = job.status
    job.status = next_status.value

    now = _utc_now()

    if next_status is JobStatus.PROCESSING:
        job.started_at = job.started_at or now
        job.attempt_count = (job.attempt_count or 0) + 1

    if next_status in TERMINAL_STATUSES:
        job.completed_at = now

    if error_message is not None:
        job.error_message = _truncate_error_message(error_message)

    if result_metadata is not None:
        job.result_metadata = result_metadata

    job.updated_at = now

    job = await update_record(db, job)

    bind_job_context(
        str(job.id),
        tenant_id=job.tenant_id,
    )

    logger.info(
        "ingestion_job_status_updated",
        job_id=str(job.id),
        document_id=str(job.document_id),
        previous_status=previous_status,
        status=job.status,
        attempt_count=job.attempt_count,
    )

    return job


async def start_ingestion_job(
    db: AsyncSession,
    job_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> IngestionJobModel:
    """
    Mark an ingestion job as processing and increment its attempt count.

    Args:
        db: Active async database session.
        job_id: Ingestion-job UUID.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The refreshed ingestion job.
    """
    return await update_ingestion_job_status(
        db,
        job_id,
        JobStatus.PROCESSING,
        tenant_id=tenant_id,
    )


async def complete_ingestion_job(
    db: AsyncSession,
    job_id: str | UUID,
    *,
    tenant_id: str | None = None,
    result_metadata: dict[str, Any] | None = None,
) -> IngestionJobModel:
    """
    Mark an ingestion job as successfully completed.

    Args:
        db: Active async database session.
        job_id: Ingestion-job UUID.
        tenant_id: Optional tenant isolation filter.
        result_metadata: Optional processing result metadata.

    Returns:
        The refreshed ingestion job.
    """
    return await update_ingestion_job_status(
        db,
        job_id,
        JobStatus.COMPLETED,
        tenant_id=tenant_id,
        result_metadata=result_metadata,
    )


async def partially_complete_ingestion_job(
    db: AsyncSession,
    job_id: str | UUID,
    *,
    tenant_id: str | None = None,
    error_message: str | None = None,
    result_metadata: dict[str, Any] | None = None,
) -> IngestionJobModel:
    """
    Mark an ingestion job as partially completed.

    Use this when some documents or chunks were processed successfully but
    others failed.

    Args:
        db: Active async database session.
        job_id: Ingestion-job UUID.
        tenant_id: Optional tenant isolation filter.
        error_message: Optional bounded summary of remaining failures.
        result_metadata: Optional processing result metadata.

    Returns:
        The refreshed ingestion job.
    """
    return await update_ingestion_job_status(
        db,
        job_id,
        JobStatus.PARTIALLY_COMPLETED,
        tenant_id=tenant_id,
        error_message=error_message,
        result_metadata=result_metadata,
    )


async def fail_ingestion_job(
    db: AsyncSession,
    job_id: str | UUID,
    *,
    tenant_id: str | None = None,
    error_message: str | None = None,
) -> IngestionJobModel:
    """
    Mark an ingestion job as failed.

    Args:
        db: Active async database session.
        job_id: Ingestion-job UUID.
        tenant_id: Optional tenant isolation filter.
        error_message: Safe, bounded failure message.

    Returns:
        The refreshed ingestion job.
    """
    return await update_ingestion_job_status(
        db,
        job_id,
        JobStatus.FAILED,
        tenant_id=tenant_id,
        error_message=error_message,
    )


async def cancel_ingestion_job(
    db: AsyncSession,
    job_id: str | UUID,
    *,
    tenant_id: str | None = None,
    reason: str | None = None,
) -> IngestionJobModel:
    """
    Cancel an ingestion job.

    Args:
        db: Active async database session.
        job_id: Ingestion-job UUID.
        tenant_id: Optional tenant isolation filter.
        reason: Optional bounded cancellation reason.

    Returns:
        The refreshed ingestion job.
    """
    return await update_ingestion_job_status(
        db,
        job_id,
        JobStatus.CANCELLED,
        tenant_id=tenant_id,
        error_message=reason,
    )


async def retry_ingestion_job(
    db: AsyncSession,
    job_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> IngestionJobModel:
    """
    Queue a failed ingestion job for retry.

    Args:
        db: Active async database session.
        job_id: Ingestion-job UUID.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The refreshed ingestion job.

    Raises:
        ValidationError: If the job has exhausted its retry allowance.
        WorkflowStateError: If the current status cannot be retried.
    """
    job = await get_ingestion_job_or_404(
        db,
        job_id,
        tenant_id=tenant_id,
    )

    current_status = _normalize_job_status(job.status)

    if current_status is None:
        raise ValidationError("Ingestion job status is invalid.")

    if current_status not in {
        JobStatus.FAILED,
        JobStatus.PARTIALLY_COMPLETED,
    }:
        raise WorkflowStateError(
            "Only failed or partially completed ingestion jobs can be retried.",
            details={"current_status": current_status.value},
        )

    if (job.attempt_count or 0) >= (job.max_attempts or 0):
        raise ValidationError(
            "Ingestion job has exhausted its maximum retry attempts.",
            details={
                "attempt_count": job.attempt_count,
                "max_attempts": job.max_attempts,
            },
        )

    return await update_ingestion_job_status(
        db,
        job_id,
        JobStatus.QUEUED,
        tenant_id=tenant_id,
    )


# ---------------------------------------------------------------------------
# Queries for workers
# ---------------------------------------------------------------------------


async def get_next_queued_job(
    db: AsyncSession,
    *,
    tenant_id: str | None = None,
) -> IngestionJobModel | None:
    """
    Retrieve the next queued ingestion job for processing.

    Jobs are ordered by priority ascending, then creation time ascending.
    This query does not claim or lock the job; the worker service is
    responsible for claiming it safely in a distributed deployment.

    Args:
        db: Active async database session.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The next queued job, or `None` when no jobs are available.
    """
    query = select(IngestionJobModel).where(
        IngestionJobModel.status == JobStatus.QUEUED.value
    )

    query = _apply_tenant_filter(query, tenant_id)

    query = query.order_by(
        IngestionJobModel.priority.asc(),
        IngestionJobModel.created_at.asc(),
    ).limit(1)

    result = await db.execute(query)

    return result.scalar_one_or_none()


async def get_stale_processing_jobs(
    db: AsyncSession,
    *,
    older_than: datetime,
    tenant_id: str | None = None,
    limit: int = 100,
) -> Sequence[IngestionJobModel]:
    """
    Retrieve processing jobs that have not completed since a cutoff time.

    Use this from a maintenance task to identify jobs that may need recovery,
    cancellation, or retry.

    Args:
        db: Active async database session.
        older_than: Cutoff timestamp.
        tenant_id: Optional tenant isolation filter.
        limit: Maximum number of jobs to return.

    Returns:
        A sequence of potentially stale ingestion jobs.
    """
    query = select(IngestionJobModel).where(
        and_(
            IngestionJobModel.status == JobStatus.PROCESSING.value,
            IngestionJobModel.started_at.is_not(None),
            IngestionJobModel.started_at <= older_than,
        )
    )

    query = _apply_tenant_filter(query, tenant_id)

    query = query.order_by(
        IngestionJobModel.started_at.asc()
    ).limit(max(1, limit))

    result = await db.execute(query)

    return result.scalars().all()


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


async def delete_ingestion_job(
    db: AsyncSession,
    job_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> None:
    """
    Permanently delete an ingestion job by ID.

    Prefer cancelling active jobs instead of deleting them. Hard deletion should
    be limited to retention, data-subject, or administrative cleanup flows.

    Args:
        db: Active async database session.
        job_id: Ingestion-job UUID.
        tenant_id: Optional tenant isolation filter.

    Raises:
        NotFoundError: If the job does not exist.
        DatabaseIntegrityError: If related records prevent deletion.
        PersistenceError: If another database failure occurs.
    """
    job = await get_ingestion_job_or_404(
        db,
        job_id,
        tenant_id=tenant_id,
    )

    job_id_value = str(job.id)

    from app.crud.base import delete_record

    await delete_record(db, job)

    logger.warning(
        "ingestion_job_deleted",
        job_id=job_id_value,
        tenant_id=tenant_id,
    )