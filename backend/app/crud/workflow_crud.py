"""
CRUD operations for workflow executions.

This module owns all direct persistence operations for workflow execution
records. API routes and services should not construct SQLAlchemy queries
directly; they should call these functions so transaction boundaries, error
mapping, filtering, pagination, and auditing remain consistent.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import JobStatus, WorkflowStep
from app.core.exceptions import (
    DatabaseIntegrityError,
    NotFoundError,
    PersistenceError,
    ValidationError,
    WorkflowNotFoundError,
)
from app.core.logging_context import bind_workflow_context
from app.core.pagination import Page, PaginationParams
from app.models.workflow_model import WorkflowExecutionModel
from app.schemas.workflow_schema import (
    WorkflowExecutionCreate,
    WorkflowExecutionUpdate,
)


logger = logging.getLogger("agentic_bridge.crud.workflow")

ALLOWED_UPDATE_FIELDS: frozenset[str] = frozenset(
    {
        "workflow_name",
        "status",
        "current_step",
        "input_payload",
        "output_payload",
        "extra_metadata",
        "embedding",
        "current_error",
        "execution_logs",
        "started_at",
        "completed_at",
    }
)

TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {
        JobStatus.COMPLETED,
        JobStatus.PARTIALLY_COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_workflow_id(workflow_id: str | UUID) -> UUID:
    """Validate and convert a workflow ID into a UUID."""
    if isinstance(workflow_id, UUID):
        return workflow_id

    if not isinstance(workflow_id, str) or not workflow_id.strip():
        raise ValidationError("workflow_id must be a non-empty string.")

    try:
        return UUID(workflow_id)
    except ValueError as exc:
        raise ValidationError("workflow_id must be a valid UUID.") from exc


def _validate_status(status: str | JobStatus | None) -> JobStatus | None:
    """Validate and normalize a workflow status value."""
    if status is None:
        return None

    try:
        return JobStatus(status)
    except ValueError as exc:
        raise ValidationError(
            f"Unsupported workflow status: {status}."
        ) from exc


def _validate_workflow_step(
    workflow_step: str | WorkflowStep | None,
) -> WorkflowStep | None:
    """Validate and normalize a workflow step value."""
    if workflow_step is None:
        return None

    try:
        return WorkflowStep(workflow_step)
    except ValueError as exc:
        raise ValidationError(
            f"Unsupported workflow step: {workflow_step}."
        ) from exc


def _apply_update_fields(
    db_obj: WorkflowExecutionModel,
    update_data: dict[str, Any],
) -> None:
    """Apply only allow-listed fields to an existing model instance."""
    unknown_fields = set(update_data) - ALLOWED_UPDATE_FIELDS

    if unknown_fields:
        raise ValidationError(
            "Unsupported workflow execution update fields.",
            details={"fields": sorted(unknown_fields)},
        )

    for field, value in update_data.items():
        setattr(db_obj, field, value)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


async def create_workflow_execution(
    db: AsyncSession,
    obj_in: WorkflowExecutionCreate,
) -> WorkflowExecutionModel:
    """
    Create and persist a new workflow execution record.

    Args:
        db: Active async database session.
        obj_in: Validated workflow creation payload.

    Returns:
        The persisted workflow execution model.

    Raises:
        ValidationError: If required values are invalid.
        DatabaseIntegrityError: If a database constraint is violated.
        PersistenceError: If another database failure occurs.
    """
    status = _validate_status(getattr(obj_in, "status", None)) or JobStatus.QUEUED
    current_step = _validate_workflow_step(
        getattr(obj_in, "current_step", None)
    ) or WorkflowStep.INITIALIZED

    db_obj = WorkflowExecutionModel(
        workflow_name=obj_in.workflow_name,
        input_payload=obj_in.input_payload,
        extra_metadata=obj_in.extra_metadata,
        embedding=obj_in.embedding,
        status=status.value,
        current_step=current_step.value,
    )

    try:
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
    except IntegrityError as exc:
        await db.rollback()

        logger.exception(
            "workflow_execution_create_integrity_error",
            workflow_name=obj_in.workflow_name,
        )

        raise DatabaseIntegrityError(
            "Workflow execution could not be created because it violates a data constraint."
        ) from exc

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "workflow_execution_create_failed",
            workflow_name=obj_in.workflow_name,
        )

        raise PersistenceError(
            "The workflow execution could not be created."
        ) from exc

    bind_workflow_context(str(db_obj.id))

    logger.info(
        "workflow_execution_created",
        workflow_id=str(db_obj.id),
        workflow_name=db_obj.workflow_name,
        status=db_obj.status,
        current_step=db_obj.current_step,
    )

    return db_obj


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


async def get_workflow_execution(
    db: AsyncSession,
    workflow_id: str | UUID,
) -> WorkflowExecutionModel | None:
    """
    Retrieve a workflow execution by ID.

    Args:
        db: Active async database session.
        workflow_id: Workflow execution UUID.

    Returns:
        The workflow execution, or `None` when it does not exist.
    """
    workflow_uuid = _parse_workflow_id(workflow_id)

    result = await db.execute(
        select(WorkflowExecutionModel).where(
            WorkflowExecutionModel.id == workflow_uuid
        )
    )

    return result.scalar_one_or_none()


async def get_workflow_execution_or_404(
    db: AsyncSession,
    workflow_id: str | UUID,
) -> WorkflowExecutionModel:
    """
    Retrieve a workflow execution or raise `WorkflowNotFoundError`.

    Use this in API routes and services where a missing workflow is a client
    error rather than an expected empty result.
    """
    workflow = await get_workflow_execution(db, workflow_id)

    if workflow is None:
        raise WorkflowNotFoundError(
            "Workflow execution was not found.",
            details={"workflow_id": str(workflow_id)},
        )

    return workflow


async def list_workflow_executions(
    db: AsyncSession,
    *,
    skip: int = 0,
    limit: int = 100,
    status: str | JobStatus | None = None,
    current_step: str | WorkflowStep | None = None,
    workflow_name: str | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
) -> Sequence[WorkflowExecutionModel]:
    """
    Retrieve a filtered and paginated list of workflow executions.

    Args:
        db: Active async database session.
        skip: Number of records to skip.
        limit: Maximum number of records to return.
        status: Optional workflow status filter.
        current_step: Optional current workflow step filter.
        workflow_name: Optional exact workflow-name filter.
        created_after: Optional inclusive lower-created-at bound.
        created_before: Optional inclusive upper-created-at bound.

    Returns:
        A sequence of workflow execution models.
    """
    safe_status = _validate_status(status)
    safe_step = _validate_workflow_step(current_step)

    query = select(WorkflowExecutionModel)

    if safe_status is not None:
        query = query.where(WorkflowExecutionModel.status == safe_status.value)

    if safe_step is not None:
        query = query.where(
            WorkflowExecutionModel.current_step == safe_step.value
        )

    if workflow_name:
        query = query.where(
            WorkflowExecutionModel.workflow_name == workflow_name
        )

    if created_after is not None:
        query = query.where(WorkflowExecutionModel.created_at >= created_after)

    if created_before is not None:
        query = query.where(WorkflowExecutionModel.created_at <= created_before)

    query = (
        query.order_by(WorkflowExecutionModel.created_at.desc())
        .offset(max(0, skip))
        .limit(max(1, limit))
    )

    result = await db.execute(query)

    return result.scalars().all()


async def list_workflow_executions_page(
    db: AsyncSession,
    pagination: PaginationParams,
    *,
    status: str | JobStatus | None = None,
    current_step: str | WorkflowStep | None = None,
    workflow_name: str | None = None,
) -> Page[WorkflowExecutionModel]:
    """
    Retrieve a paginated workflow execution page with total count metadata.

    Args:
        db: Active async database session.
        pagination: Validated pagination parameters.
        status: Optional workflow status filter.
        current_step: Optional current workflow step filter.
        workflow_name: Optional exact workflow-name filter.

    Returns:
        A `Page` containing workflow executions and pagination metadata.
    """
    safe_status = _validate_status(status)
    safe_step = _validate_workflow_step(current_step)

    filters = []

    if safe_status is not None:
        filters.append(WorkflowExecutionModel.status == safe_status.value)

    if safe_step is not None:
        filters.append(
            WorkflowExecutionModel.current_step == safe_step.value
        )

    if workflow_name:
        filters.append(WorkflowExecutionModel.workflow_name == workflow_name)

    count_query = select(func.count()).select_from(WorkflowExecutionModel)

    if filters:
        count_query = count_query.where(*filters)

    count_result = await db.execute(count_query)
    total = int(count_result.scalar_one())

    items = await list_workflow_executions(
        db,
        skip=pagination.skip,
        limit=pagination.limit,
        status=safe_status,
        current_step=safe_step,
        workflow_name=workflow_name,
    )

    return Page[WorkflowExecutionModel].create(
        items=list(items),
        total=total,
        skip=pagination.skip,
        limit=pagination.limit,
    )


async def count_workflow_executions(
    db: AsyncSession,
    *,
    status: str | JobStatus | None = None,
) -> int:
    """Count workflow executions, optionally filtered by status."""
    safe_status = _validate_status(status)

    query = select(func.count()).select_from(WorkflowExecutionModel)

    if safe_status is not None:
        query = query.where(WorkflowExecutionModel.status == safe_status.value)

    result = await db.execute(query)

    return int(result.scalar_one())


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


async def update_workflow_execution(
    db: AsyncSession,
    db_obj: WorkflowExecutionModel,
    obj_in: WorkflowExecutionUpdate,
) -> WorkflowExecutionModel:
    """
    Update an existing workflow execution with validated changes.

    Args:
        db: Active async database session.
        db_obj: Existing workflow execution model.
        obj_in: Validated update payload.

    Returns:
        The refreshed workflow execution model.

    Raises:
        ValidationError: If an unsupported field or state transition is supplied.
        DatabaseIntegrityError: If a database constraint is violated.
        PersistenceError: If another database failure occurs.
    """
    update_data = obj_in.model_dump(exclude_unset=True)

    if "status" in update_data:
        update_data["status"] = _validate_status(update_data["status"]).value

    if "current_step" in update_data:
        update_data["current_step"] = _validate_workflow_step(
            update_data["current_step"]
        ).value

    _apply_update_fields(db_obj, update_data)

    try:
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
    except IntegrityError as exc:
        await db.rollback()

        logger.exception(
            "workflow_execution_update_integrity_error",
            workflow_id=str(db_obj.id),
        )

        raise DatabaseIntegrityError(
            "Workflow execution could not be updated because it violates a data constraint."
        ) from exc

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "workflow_execution_update_failed",
            workflow_id=str(db_obj.id),
        )

        raise PersistenceError(
            "The workflow execution could not be updated."
        ) from exc

    bind_workflow_context(str(db_obj.id))

    logger.info(
        "workflow_execution_updated",
        workflow_id=str(db_obj.id),
        status=db_obj.status,
        current_step=db_obj.current_step,
        updated_fields=sorted(update_data),
    )

    return db_obj


async def update_workflow_status(
    db: AsyncSession,
    workflow_id: str | UUID,
    *,
    status: str | JobStatus,
    current_step: str | WorkflowStep | None = None,
    current_error: str | None = None,
) -> WorkflowExecutionModel:
    """
    Update workflow status and optionally the current step/error.

    This is the preferred method for workflow nodes and background workers that
    need to record progress without loading and mutating a full ORM instance.
    """
    workflow_uuid = _parse_workflow_id(workflow_id)
    safe_status = _validate_status(status)

    if safe_status is None:
        raise ValidationError("Workflow status is required.")

    values: dict[str, Any] = {"status": safe_status.value}

    safe_step = _validate_workflow_step(current_step)

    if safe_step is not None:
        values["current_step"] = safe_step.value

    if current_error is not None:
        values["current_error"] = current_error

    if safe_status in TERMINAL_STATUSES:
        values["completed_at"] = datetime.utcnow()

    result = await db.execute(
        update(WorkflowExecutionModel)
        .where(WorkflowExecutionModel.id == workflow_uuid)
        .values(**values)
        .returning(WorkflowExecutionModel)
    )

    updated_workflow = result.scalar_one_or_none()

    if updated_workflow is None:
        await db.rollback()
        raise WorkflowNotFoundError(
            "Workflow execution was not found.",
            details={"workflow_id": str(workflow_uuid)},
        )

    await db.commit()
    await db.refresh(updated_workflow)

    bind_workflow_context(str(updated_workflow.id))

    logger.info(
        "workflow_status_updated",
        workflow_id=str(updated_workflow.id),
        status=updated_workflow.status,
        current_step=updated_workflow.current_step,
    )

    return updated_workflow


async def append_workflow_log(
    db: AsyncSession,
    workflow_id: str | UUID,
    message: str,
    *,
    level: str = "INFO",
) -> WorkflowExecutionModel:
    """
    Append a bounded execution-log entry to a workflow execution.

    Args:
        db: Active async database session.
        workflow_id: Workflow execution UUID.
        message: Human-readable log message.
        level: Log level label, such as `INFO`, `WARNING`, or `ERROR`.

    Returns:
        The refreshed workflow execution model.
    """
    workflow = await get_workflow_execution_or_404(db, workflow_id)

    if not message.strip():
        raise ValidationError("Workflow log message cannot be empty.")

    safe_level = level.strip().upper()

    if safe_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValidationError("Unsupported workflow log level.")

    logs = list(workflow.execution_logs or [])
    logs.append(f"{safe_level}: {message.strip()}")

    workflow.execution_logs = logs

    try:
        db.add(workflow)
        await db.commit()
        await db.refresh(workflow)
    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "workflow_log_append_failed",
            workflow_id=str(workflow.id),
        )

        raise PersistenceError(
            "The workflow execution log could not be updated."
        ) from exc

    return workflow


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


async def delete_workflow_execution(
    db: AsyncSession,
    db_obj: WorkflowExecutionModel,
) -> None:
    """
    Delete a workflow execution record.

    Args:
        db: Active async database session.
        db_obj: Existing workflow execution model to delete.

    Raises:
        PersistenceError: If the deletion cannot be completed.
    """
    workflow_id = str(db_obj.id)

    try:
        await db.delete(db_obj)
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()

        logger.exception(
            "workflow_execution_delete_integrity_error",
            workflow_id=workflow_id,
        )

        raise DatabaseIntegrityError(
            "Workflow execution could not be deleted because related records exist."
        ) from exc

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "workflow_execution_delete_failed",
            workflow_id=workflow_id,
        )

        raise PersistenceError(
            "The workflow execution could not be deleted."
        ) from exc

    logger.info(
        "workflow_execution_deleted",
        workflow_id=workflow_id,
    )


async def delete_workflow_execution_by_id(
    db: AsyncSession,
    workflow_id: str | UUID,
) -> None:
    """
    Delete a workflow execution by ID.

    Raises:
        WorkflowNotFoundError: If the workflow execution does not exist.
    """
    workflow = await get_workflow_execution_or_404(db, workflow_id)

    await delete_workflow_execution(db, workflow)