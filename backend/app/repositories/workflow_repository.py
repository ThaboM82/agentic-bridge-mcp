"""
Workflow execution repository.

This module provides an enterprise-grade asynchronous repository for workflow
execution records. It encapsulates creation, retrieval, pagination, state
updates, deletion, pgvector similarity search, analytics, tenant-aware
filtering, and safe state transitions.

Design principles:
- Repositories own persistence concerns only.
- All writes use the supplied AsyncSession and flush without committing.
- Tenant isolation is enforced for every tenant-scoped read or mutation.
- Inputs are validated before reaching SQLAlchemy or pgvector.
- State transitions are explicit, idempotent, and auditable.
- Raw provider data, secrets, prompts, and embeddings are never logged.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import JobStatus, WorkflowStep
from app.core.exceptions import ValidationError
from app.models.workflow_model import WorkflowExecutionModel


logger = logging.getLogger(
    "agentic_bridge.repositories.workflow"
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TERMINAL_WORKFLOW_STEPS = {
    WorkflowStep.COMPLETED.value,
    "completed",
    "END",
}

TERMINAL_WORKFLOW_STATUSES = {
    JobStatus.COMPLETED.value,
    JobStatus.FAILED.value,
    JobStatus.CANCELLED.value,
}

SUPPORTED_LIST_STATUSES = {
    JobStatus.QUEUED.value,
    JobStatus.PROCESSING.value,
    JobStatus.COMPLETED.value,
    JobStatus.FAILED.value,
    JobStatus.CANCELLED.value,
}

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MAX_SIMILARITY_RESULTS = 50
MAX_EMBEDDING_DIMENSIONS = 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_uuid(
    value: str | UUID,
    field_name: str,
) -> UUID:
    """Normalize and validate a UUID-like identifier."""
    if isinstance(value, UUID):
        return value

    if not isinstance(value, str):
        raise ValidationError(
            f"{field_name} must be a UUID string."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValidationError(
            f"{field_name} must not be empty."
        )

    try:
        return UUID(normalized_value)
    except ValueError as exc:
        raise ValidationError(
            f"{field_name} must be a valid UUID."
        ) from exc


def _normalize_optional_uuid(
    value: str | UUID | None,
    field_name: str,
) -> UUID | None:
    """Normalize an optional UUID-like identifier."""
    if value is None:
        return None

    return _normalize_uuid(
        value,
        field_name,
    )


def _normalize_pagination(
    skip: int,
    limit: int,
) -> tuple[int, int]:
    """Normalize pagination inputs and enforce safe bounds."""
    if not isinstance(skip, int) or isinstance(
        skip, bool
    ):
        raise ValidationError(
            "skip must be an integer."
        )

    if not isinstance(limit, int) or isinstance(
        limit, bool
    ):
        raise ValidationError(
            "limit must be an integer."
        )

    if skip < 0:
        raise ValidationError(
            "skip must be greater than or equal to zero."
        )

    if limit <= 0:
        raise ValidationError(
            "limit must be greater than zero."
        )

    return skip, min(limit, MAX_PAGE_SIZE)


def _normalize_status_filter(
    status_filter: str | JobStatus | None,
) -> str | None:
    """Normalize an optional workflow status filter."""
    if status_filter is None:
        return None

    if isinstance(status_filter, JobStatus):
        normalized_status = (
            status_filter.value
        )
    elif isinstance(status_filter, str):
        normalized_status = (
            status_filter.strip().lower()
        )
    else:
        raise ValidationError(
            "status_filter must be a string or JobStatus."
        )

    if not normalized_status:
        return None

    if (
        normalized_status
        not in SUPPORTED_LIST_STATUSES
    ):
        raise ValidationError(
            f"Unsupported workflow status filter: {status_filter!r}."
        )

    return normalized_status


def _normalize_embedding(
    embedding: Sequence[float] | None,
) -> list[float] | None:
    """Validate an optional embedding vector before persistence or search."""
    if embedding is None:
        return None

    if not isinstance(
        embedding, (list, tuple)
    ):
        raise ValidationError(
            "embedding must be a list or tuple of numeric values."
        )

    vector = list(embedding)

    if not vector:
        raise ValidationError(
            "embedding must not be empty."
        )

    if len(vector) > MAX_EMBEDDING_DIMENSIONS:
        raise ValidationError(
            "embedding exceeds the maximum supported dimensions."
        )

    normalized_vector: list[float] = []

    for value in vector:
        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise ValidationError(
                "embedding values must be numeric."
            )

        float_value = float(value)

        if not math.isfinite(float_value):
            raise ValidationError(
                "embedding values must be finite."
            )

        normalized_vector.append(
            float_value
        )

    return normalized_vector


def _normalize_state_updates(
    state_updates: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate and normalize a workflow state-update payload."""
    if not isinstance(
        state_updates, dict
    ):
        raise ValidationError(
            "state_updates must be a dictionary."
        )

    return dict(state_updates)


def _derive_status_from_state(
    current_record: WorkflowExecutionModel,
    state_updates: Dict[str, Any],
) -> str:
    """Derive a safe workflow status from updated graph state."""
    explicit_status = state_updates.get(
        "workflow_status"
    )

    if isinstance(
        explicit_status, JobStatus
    ):
        return explicit_status.value

    if isinstance(explicit_status, str):
        normalized_status = (
            explicit_status.strip().lower()
        )

        if (
            normalized_status
            in SUPPORTED_LIST_STATUSES
        ):
            return normalized_status

    current_error = state_updates.get(
        "current_error",
        current_record.current_error,
    )
    recoverable_error = state_updates.get(
        "recoverable_error",
        current_record.recoverable_error,
    )
    cancel_requested = state_updates.get(
        "cancel_requested",
        current_record.cancel_requested,
    )

    if cancel_requested:
        return JobStatus.CANCELLED.value

    if current_error:
        if recoverable_error:
            return JobStatus.FAILED.value

        return JobStatus.FAILED.value

    current_step = state_updates.get(
        "current_step",
        current_record.current_step,
    )

    if current_step in TERMINAL_WORKFLOW_STEPS:
        return JobStatus.COMPLETED.value

    return current_record.status


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class WorkflowRepository:
    """
    Enterprise-grade repository encapsulating asynchronous database operations,
    pagination, vector similarity searches, and telemetry analytics for
    workflow executions.
    """

    @staticmethod
    async def create_execution(
        db: AsyncSession,
        execution_id: str,
        brief_id: str,
        raw_brief_text: str,
        tenant_id: str,
        user_id: str,
        embedding: Optional[List[float]] = None,
        approval_required: bool = True,
        max_retries: int = 3,
    ) -> WorkflowExecutionModel:
        """
        Create and persist a new workflow execution record with initial state.
        """
        normalized_execution_id = _normalize_uuid(
            execution_id,
            "execution_id",
        )
        normalized_brief_id = _normalize_uuid(
            brief_id,
            "brief_id",
        )
        normalized_tenant_id = _normalize_uuid(
            tenant_id,
            "tenant_id",
        )
        normalized_user_id = _normalize_uuid(
            user_id,
            "user_id",
        )
        normalized_embedding = (
            _normalize_embedding(embedding)
        )

        if not isinstance(
            raw_brief_text, str
        ) or not raw_brief_text.strip():
            raise ValidationError(
                "raw_brief_text must be a non-empty string."
            )

        if not isinstance(
            approval_required, bool
        ):
            raise ValidationError(
                "approval_required must be a boolean."
            )

        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or max_retries < 0
        ):
            raise ValidationError(
                "max_retries must be a non-negative integer."
            )

        logger.info(
            "Persisting new workflow execution record.",
            extra={
                "execution_id": str(
                    normalized_execution_id
                ),
                "tenant_id": str(
                    normalized_tenant_id
                ),
                "brief_id": str(
                    normalized_brief_id
                ),
            },
        )

        db_record = WorkflowExecutionModel(
            execution_id=str(
                normalized_execution_id
            ),
            brief_id=str(normalized_brief_id),
            tenant_id=str(normalized_tenant_id),
            user_id=str(normalized_user_id),
            raw_brief_text=raw_brief_text.strip(),
            current_step=WorkflowStep.INITIALIZED.value,
            status=JobStatus.QUEUED.value,
            approval_required=approval_required,
            retry_count=0,
            max_retries=max_retries,
            execution_logs=[
                "INFO: Initialized execution record "
                f"{normalized_execution_id}"
            ],
            brief_embedding=normalized_embedding,
        )

        db.add(db_record)
        await db.flush()
        await db.refresh(db_record)

        return db_record

    @staticmethod
    async def get_by_execution_id(
        db: AsyncSession,
        execution_id: str,
        tenant_id: str | UUID | None = None,
    ) -> Optional[WorkflowExecutionModel]:
        """
        Retrieve a workflow execution record by its unique execution ID.

        When tenant_id is supplied, records belonging to another tenant are
        treated as nonexistent.
        """
        normalized_execution_id = _normalize_uuid(
            execution_id,
            "execution_id",
        )
        normalized_tenant_id = (
            _normalize_optional_uuid(
                tenant_id,
                "tenant_id",
            )
        )

        stmt = select(
            WorkflowExecutionModel
        ).where(
            WorkflowExecutionModel.execution_id
            == str(normalized_execution_id)
        )

        if normalized_tenant_id:
            stmt = stmt.where(
                WorkflowExecutionModel.tenant_id
                == str(normalized_tenant_id)
            )

        result = await db.execute(stmt)

        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_brief_id(
        db: AsyncSession,
        brief_id: str,
        tenant_id: str | UUID,
    ) -> List[WorkflowExecutionModel]:
        """Retrieve all workflow executions associated with a brief."""
        normalized_brief_id = _normalize_uuid(
            brief_id,
            "brief_id",
        )
        normalized_tenant_id = _normalize_uuid(
            tenant_id,
            "tenant_id",
        )

        stmt = (
            select(WorkflowExecutionModel)
            .where(
                WorkflowExecutionModel.brief_id
                == str(normalized_brief_id)
            )
            .where(
                WorkflowExecutionModel.tenant_id
                == str(normalized_tenant_id)
            )
            .order_by(
                WorkflowExecutionModel.created_at.desc()
            )
        )

        result = await db.execute(stmt)

        return list(
            result.scalars().all()
        )

    @staticmethod
    async def list_executions(
        db: AsyncSession,
        tenant_id: str | UUID,
        skip: int = 0,
        limit: int = DEFAULT_PAGE_SIZE,
        status_filter: Optional[
            str | JobStatus
        ] = None,
        current_step_filter: Optional[
            str | WorkflowStep
        ] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
    ) -> Tuple[
        List[WorkflowExecutionModel],
        int,
    ]:
        """
        Retrieve a paginated, tenant-scoped list of workflow execution records
        with optional status, step, and creation-date filters.
        """
        normalized_tenant_id = _normalize_uuid(
            tenant_id,
            "tenant_id",
        )
        normalized_skip, normalized_limit = (
            _normalize_pagination(
                skip, limit
            )
        )
        normalized_status = (
            _normalize_status_filter(
                status_filter
            )
        )

        if current_step_filter is not None:
            if isinstance(
                current_step_filter,
                WorkflowStep,
            ):
                normalized_step = (
                    current_step_filter.value
                )
            elif isinstance(
                current_step_filter, str
            ):
                normalized_step = (
                    current_step_filter.strip().lower()
                )
            else:
                raise ValidationError(
                    "current_step_filter must be a string or WorkflowStep."
                )

            if not normalized_step:
                normalized_step = None
        else:
            normalized_step = None

        if created_after is not None and not isinstance(
            created_after, datetime
        ):
            raise ValidationError(
                "created_after must be a datetime."
            )

        if created_before is not None and not isinstance(
            created_before, datetime
        ):
            raise ValidationError(
                "created_before must be a datetime."
            )

        filters = [
            WorkflowExecutionModel.tenant_id
            == str(normalized_tenant_id)
        ]

        if normalized_status:
            filters.append(
                WorkflowExecutionModel.status
                == normalized_status
            )

        if normalized_step:
            filters.append(
                WorkflowExecutionModel.current_step
                == normalized_step
            )

        if created_after:
            filters.append(
                WorkflowExecutionModel.created_at
                >= created_after
            )

        if created_before:
            filters.append(
                WorkflowExecutionModel.created_at
                <= created_before
            )

        count_query = select(
            func.count(
                WorkflowExecutionModel.id
            )
        ).where(*filters)

        total_count_result = await db.execute(
            count_query
        )
        total_count = (
            total_count_result.scalar_one()
        )

        paginated_stmt = (
            select(WorkflowExecutionModel)
            .where(*filters)
            .order_by(
                WorkflowExecutionModel.created_at.desc(),
                WorkflowExecutionModel.id.desc(),
            )
            .offset(normalized_skip)
            .limit(normalized_limit)
        )

        result = await db.execute(
            paginated_stmt
        )
        records = list(
            result.scalars().all()
        )

        return records, total_count

    @staticmethod
    async def update_execution_state(
        db: AsyncSession,
        execution_id: str,
        tenant_id: str | UUID,
        state_updates: Dict[str, Any],
    ) -> Optional[WorkflowExecutionModel]:
        """
        Update an existing tenant-scoped workflow execution record with new
        state data emitted by the workflow graph.
        """
        normalized_execution_id = _normalize_uuid(
            execution_id,
            "execution_id",
        )
        normalized_tenant_id = _normalize_uuid(
            tenant_id,
            "tenant_id",
        )
        normalized_updates = (
            _normalize_state_updates(
                state_updates
            )
        )

        logger.debug(
            "Updating database state for workflow execution.",
            extra={
                "execution_id": str(
                    normalized_execution_id
                ),
                "tenant_id": str(
                    normalized_tenant_id
                ),
                "update_keys": sorted(
                    normalized_updates.keys()
                ),
            },
        )

        db_record = (
            await WorkflowRepository.get_by_execution_id(
                db,
                execution_id=str(
                    normalized_execution_id
                ),
                tenant_id=normalized_tenant_id,
            )
        )

        if not db_record:
            logger.warning(
                "Workflow execution not found for state update.",
                extra={
                    "execution_id": str(
                        normalized_execution_id
                    ),
                    "tenant_id": str(
                        normalized_tenant_id
                    ),
                },
            )
            return None

        allowed_fields = {
            "current_step",
            "previous_step",
            "workflow_status",
            "extracted_requirements",
            "architectural_specification",
            "design_tokens",
            "generated_code_files",
            "execution_logs",
            "current_error",
            "error_step",
            "error_code",
            "recoverable_error",
            "retry_count",
            "max_retries",
            "approval_required",
            "approval_status",
            "review_status",
            "cancel_requested",
            "interrupt_history",
            "current_interrupt",
            "provider_usage",
            "artifacts",
            "citations",
        }

        for field_name, field_value in normalized_updates.items():
            if (
                field_name
                not in allowed_fields
            ):
                continue

            setattr(
                db_record,
                field_name,
                field_value,
            )

        if (
            "current_error"
            in normalized_updates
        ):
            current_error = normalized_updates[
                "current_error"
            ]

            if current_error:
                db_record.status = (
                    JobStatus.FAILED.value
                )

        if normalized_updates.get(
            "cancel_requested"
        ):
            db_record.status = (
                JobStatus.CANCELLED.value
            )

        db_record.status = (
            _derive_status_from_state(
                db_record,
                normalized_updates,
            )
        )

        db_record.updated_at = datetime.now(
            timezone.utc
        )

        await db.flush()
        await db.refresh(db_record)

        return db_record

    @staticmethod
    async def set_workflow_status(
        db: AsyncSession,
        execution_id: str,
        tenant_id: str | UUID,
        status: str | JobStatus,
    ) -> Optional[WorkflowExecutionModel]:
        """Explicitly set the status of a tenant-scoped workflow execution."""
        normalized_execution_id = _normalize_uuid(
            execution_id,
            "execution_id",
        )
        normalized_tenant_id = _normalize_uuid(
            tenant_id,
            "tenant_id",
        )

        if isinstance(status, JobStatus):
            normalized_status = status.value
        elif isinstance(status, str):
            normalized_status = (
                status.strip().lower()
            )
        else:
            raise ValidationError(
                "status must be a string or JobStatus."
            )

        if (
            normalized_status
            not in SUPPORTED_LIST_STATUSES
        ):
            raise ValidationError(
                f"Unsupported workflow status: {status!r}."
            )

        db_record = (
            await WorkflowRepository.get_by_execution_id(
                db,
                execution_id=str(
                    normalized_execution_id
                ),
                tenant_id=normalized_tenant_id,
            )
        )

        if not db_record:
            return None

        db_record.status = normalized_status
        db_record.updated_at = datetime.now(
            timezone.utc
        )

        await db.flush()
        await db.refresh(db_record)

        return db_record

    @staticmethod
    async def append_execution_log(
        db: AsyncSession,
        execution_id: str,
        tenant_id: str | UUID,
        message: str,
        level: str = "INFO",
    ) -> Optional[WorkflowExecutionModel]:
        """Append a bounded, sanitized log entry to a workflow execution."""
        normalized_execution_id = _normalize_uuid(
            execution_id,
            "execution_id",
        )
        normalized_tenant_id = _normalize_uuid(
            tenant_id,
            "tenant_id",
        )

        if not isinstance(message, str):
            raise ValidationError(
                "message must be a string."
            )

        normalized_message = (
            message.strip()
        )

        if not normalized_message:
            raise ValidationError(
                "message must not be empty."
            )

        if not isinstance(level, str):
            raise ValidationError(
                "level must be a string."
            )

        normalized_level = (
            level.strip().upper()
        )

        if normalized_level not in {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        }:
            raise ValidationError(
                "Unsupported log level."
            )

        db_record = (
            await WorkflowRepository.get_by_execution_id(
                db,
                execution_id=str(
                    normalized_execution_id
                ),
                tenant_id=normalized_tenant_id,
            )
        )

        if not db_record:
            return None

        existing_logs = list(
            db_record.execution_logs or []
        )
        existing_logs.append(
            f"{normalized_level}: {normalized_message}"
        )

        db_record.execution_logs = existing_logs[
            -100:
        ]
        db_record.updated_at = datetime.now(
            timezone.utc
        )

        await db.flush()
        await db.refresh(db_record)

        return db_record

    @staticmethod
    async def delete_execution(
        db: AsyncSession,
        execution_id: str,
        tenant_id: str | UUID,
    ) -> bool:
        """Delete a tenant-scoped workflow execution record."""
        normalized_execution_id = _normalize_uuid(
            execution_id,
            "execution_id",
        )
        normalized_tenant_id = _normalize_uuid(
            tenant_id,
            "tenant_id",
        )

        logger.info(
            "Deleting workflow execution record.",
            extra={
                "execution_id": str(
                    normalized_execution_id
                ),
                "tenant_id": str(
                    normalized_tenant_id
                ),
            },
        )

        db_record = (
            await WorkflowRepository.get_by_execution_id(
                db,
                execution_id=str(
                    normalized_execution_id
                ),
                tenant_id=normalized_tenant_id,
            )
        )

        if not db_record:
            return False

        await db.delete(db_record)
        await db.flush()

        return True

    @staticmethod
    async def delete_executions_for_brief(
        db: AsyncSession,
        brief_id: str,
        tenant_id: str | UUID,
    ) -> int:
        """Delete every workflow execution associated with a tenant brief."""
        normalized_brief_id = _normalize_uuid(
            brief_id,
            "brief_id",
        )
        normalized_tenant_id = _normalize_uuid(
            tenant_id,
            "tenant_id",
        )

        stmt = (
            delete(WorkflowExecutionModel)
            .where(
                WorkflowExecutionModel.brief_id
                == str(normalized_brief_id)
            )
            .where(
                WorkflowExecutionModel.tenant_id
                == str(normalized_tenant_id)
            )
        )

        result = await db.execute(stmt)
        await db.flush()

        return result.rowcount or 0

    @staticmethod
    async def search_similar_briefs(
        db: AsyncSession,
        tenant_id: str | UUID,
        query_embedding: Sequence[float],
        limit: int = 5,
        exclude_execution_id: Optional[
            str | UUID
        ] = None,
    ) -> List[WorkflowExecutionModel]:
        """
        Perform a tenant-scoped pgvector cosine-distance similarity search
        against stored brief embeddings.
        """
        normalized_tenant_id = _normalize_uuid(
            tenant_id,
            "tenant_id",
        )
        normalized_embedding = (
            _normalize_embedding(
                query_embedding
            )
        )

        if normalized_embedding is None:
            raise ValidationError(
                "query_embedding must not be empty."
            )

        _, normalized_limit = (
            _normalize_pagination(
                0, limit
            )
        )
        normalized_limit = min(
            normalized_limit,
            MAX_SIMILARITY_RESULTS,
        )

        normalized_exclude_id = (
            _normalize_optional_uuid(
                exclude_execution_id,
                "exclude_execution_id",
            )
        )

        logger.info(
            "Executing pgvector similarity search for workflow briefs.",
            extra={
                "tenant_id": str(
                    normalized_tenant_id
                ),
                "result_limit": normalized_limit,
            },
        )

        stmt = (
            select(WorkflowExecutionModel)
            .where(
                WorkflowExecutionModel.tenant_id
                == str(normalized_tenant_id)
            )
            .where(
                WorkflowExecutionModel.brief_embedding.is_not(
                    None
                )
            )
        )

        if normalized_exclude_id:
            stmt = stmt.where(
                WorkflowExecutionModel.execution_id
                != str(normalized_exclude_id)
            )

        stmt = (
            stmt.order_by(
                WorkflowExecutionModel.brief_embedding.cosine_distance(
                    normalized_embedding
                )
            )
            .limit(normalized_limit)
        )

        result = await db.execute(stmt)

        return list(
            result.scalars().all()
        )

    @staticmethod
    async def get_execution_analytics(
        db: AsyncSession,
        tenant_id: str | UUID,
    ) -> Dict[str, Any]:
        """Aggregate tenant-scoped workflow telemetry and status statistics."""
        normalized_tenant_id = _normalize_uuid(
            tenant_id,
            "tenant_id",
        )

        tenant_filter = (
            WorkflowExecutionModel.tenant_id
            == str(normalized_tenant_id)
        )

        total_stmt = select(
            func.count(
                WorkflowExecutionModel.id
            )
        ).where(tenant_filter)

        completed_stmt = (
            select(
                func.count(
                    WorkflowExecutionModel.id
                )
            )
            .where(tenant_filter)
            .where(
                WorkflowExecutionModel.status
                == JobStatus.COMPLETED.value
            )
        )

        failed_stmt = (
            select(
                func.count(
                    WorkflowExecutionModel.id
                )
            )
            .where(tenant_filter)
            .where(
                WorkflowExecutionModel.status
                == JobStatus.FAILED.value
            )
        )

        running_stmt = (
            select(
                func.count(
                    WorkflowExecutionModel.id
                )
            )
            .where(tenant_filter)
            .where(
                WorkflowExecutionModel.status
                == JobStatus.PROCESSING.value
            )
        )

        queued_stmt = (
            select(
                func.count(
                    WorkflowExecutionModel.id
                )
            )
            .where(tenant_filter)
            .where(
                WorkflowExecutionModel.status
                == JobStatus.QUEUED.value
            )
        )

        cancelled_stmt = (
            select(
                func.count(
                    WorkflowExecutionModel.id
                )
            )
            .where(tenant_filter)
            .where(
                WorkflowExecutionModel.status
                == JobStatus.CANCELLED.value
            )
        )

        total = (
            await db.execute(total_stmt)
        ).scalar_one()
        completed = (
            await db.execute(
                completed_stmt
            )
        ).scalar_one()
        failed = (
            await db.execute(failed_stmt)
        ).scalar_one()
        running = (
            await db.execute(
                running_stmt
            )
        ).scalar_one()
        queued = (
            await db.execute(queued_stmt)
        ).scalar_one()
        cancelled = (
            await db.execute(
                cancelled_stmt
            )
        ).scalar_one()

        success_rate = (
            round(
                completed / total * 100,
                2,
            )
            if total > 0
            else 0.0
        )

        return {
            "tenant_id": str(
                normalized_tenant_id
            ),
            "total_executions": total,
            "completed_executions": completed,
            "failed_executions": failed,
            "running_executions": running,
            "queued_executions": queued,
            "cancelled_executions": cancelled,
            "success_rate_percentage": success_rate,
        }

    @staticmethod
    async def count_executions(
        db: AsyncSession,
        tenant_id: str | UUID,
        status_filter: Optional[
            str | JobStatus
        ] = None,
    ) -> int:
        """Count tenant-scoped workflow executions, optionally by status."""
        normalized_tenant_id = _normalize_uuid(
            tenant_id,
            "tenant_id",
        )
        normalized_status = (
            _normalize_status_filter(
                status_filter
            )
        )

        stmt = (
            select(
                func.count(
                    WorkflowExecutionModel.id
                )
            ).where(
                WorkflowExecutionModel.tenant_id
                == str(normalized_tenant_id)
            )
        )

        if normalized_status:
            stmt = stmt.where(
                WorkflowExecutionModel.status
                == normalized_status
            )

        result = await db.execute(stmt)

        return result.scalar_one()

    @staticmethod
    async def mark_cancel_requested(
        db: AsyncSession,
        execution_id: str,
        tenant_id: str | UUID,
    ) -> Optional[WorkflowExecutionModel]:
        """Mark a tenant-scoped workflow execution as cancellation requested."""
        normalized_execution_id = _normalize_uuid(
            execution_id,
            "execution_id",
        )
        normalized_tenant_id = _normalize_uuid(
            tenant_id,
            "tenant_id",
        )

        db_record = (
            await WorkflowRepository.get_by_execution_id(
                db,
                execution_id=str(
                    normalized_execution_id
                ),
                tenant_id=normalized_tenant_id,
            )
        )

        if not db_record:
            return None

        if (
            db_record.status
            in TERMINAL_WORKFLOW_STATUSES
        ):
            return db_record

        db_record.cancel_requested = True
        db_record.updated_at = datetime.now(
            timezone.utc
        )

        await db.flush()
        await db.refresh(db_record)

        return db_record