"""
Shared async CRUD and repository helpers.

This module centralizes repeatable SQLAlchemy persistence patterns so each
domain CRUD module remains focused on its own queries, filters, and model
rules.

Responsibilities:
- Retrieve records by ID.
- Apply pagination and counting.
- Create, update, and delete records.
- Map low-level database failures to stable application exceptions.
- Support bulk inserts and optimistic concurrency where appropriate.

Non-goals:
- Do not place workflow rules, approval policy, LLM calls, embedding generation,
  caching, or HTTP concerns in this module.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Generic, TypeVar
from uuid import UUID

from sqlalchemy import Delete, Insert, Select, Update, delete, func, insert, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    DatabaseIntegrityError,
    NotFoundError,
    OptimisticLockError,
    PersistenceError,
    ValidationError,
)


logger = logging.getLogger("agentic_bridge.crud.base")


ModelT = TypeVar("ModelT")


# ---------------------------------------------------------------------------
# ID and query validation
# ---------------------------------------------------------------------------


def normalize_uuid(value: UUID | str, field_name: str = "id") -> UUID:
    """
    Validate and normalize a UUID value.

    Args:
        value: UUID instance or UUID-formatted string.
        field_name: Field name used in validation errors.

    Returns:
        A validated `UUID`.

    Raises:
        ValidationError: If the value is empty or not a valid UUID.
    """
    if isinstance(value, UUID):
        return value

    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} must be a non-empty string.")

    try:
        return UUID(value.strip())
    except ValueError as exc:
        raise ValidationError(
            f"{field_name} must be a valid UUID."
        ) from exc


def apply_filters(query: Select, *filters: Any) -> Select:
    """Apply one or more SQLAlchemy filters to a select query."""
    if filters:
        query = query.where(*filters)

    return query


def apply_pagination(
    query: Select,
    *,
    skip: int = 0,
    limit: int = 100,
) -> Select:
    """
    Apply safe offset/limit pagination to a select query.

    Args:
        query: SQLAlchemy select query.
        skip: Number of records to skip.
        limit: Maximum number of records to return.

    Returns:
        A paginated query.
    """
    if skip < 0:
        raise ValidationError("skip must be greater than or equal to zero.")

    if limit <= 0:
        raise ValidationError("limit must be greater than zero.")

    return query.offset(skip).limit(limit)


def build_count_query(
    model: type[ModelT],
    *filters: Any,
) -> Select:
    """Build a count query for a model with optional filters."""
    query = select(func.count()).select_from(model)

    if filters:
        query = query.where(*filters)

    return query


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


async def get_by_id(
    db: AsyncSession,
    model: type[ModelT],
    record_id: UUID | str,
) -> ModelT | None:
    """
    Retrieve a record by primary key.

    Args:
        db: Active async database session.
        model: SQLAlchemy model class.
        record_id: Record UUID.

    Returns:
        The matching record, or `None` when it does not exist.
    """
    record_uuid = normalize_uuid(record_id)

    result = await db.execute(
        select(model).where(model.id == record_uuid)
    )

    return result.scalar_one_or_none()


async def get_by_id_or_404(
    db: AsyncSession,
    model: type[ModelT],
    record_id: UUID | str,
) -> ModelT:
    """
    Retrieve a record by primary key or raise `NotFoundError`.

    Args:
        db: Active async database session.
        model: SQLAlchemy model class.
        record_id: Record UUID.

    Returns:
        The matching record.

    Raises:
        NotFoundError: If no matching record exists.
    """
    record = await get_by_id(db, model, record_id)

    if record is None:
        raise NotFoundError(
            "The requested resource was not found.",
            details={"id": str(record_id)},
        )

    return record


async def get_one(
    db: AsyncSession,
    query: Select,
) -> ModelT | None:
    """
    Execute a query and return at most one record.

    Raises:
        PersistenceError: If the query returns more than one record.
    """
    result = await db.execute(query)

    try:
        return result.scalar_one_or_none()
    except SQLAlchemyError as exc:
        logger.exception("crud_get_one_failed")
        raise PersistenceError(
            "The database query could not be completed."
        ) from exc


async def get_one_or_404(
    db: AsyncSession,
    query: Select,
    *,
    resource_name: str = "resource",
    resource_id: str | None = None,
) -> ModelT:
    """
    Execute a query and return one record or raise `NotFoundError`.

    Args:
        db: Active async database session.
        query: SQLAlchemy select query.
        resource_name: Human-readable resource name for error messages.
        resource_id: Optional resource identifier for error details.

    Returns:
        The matching record.

    Raises:
        NotFoundError: If no matching record exists.
    """
    record = await get_one(db, query)

    if record is None:
        details = {"resource": resource_name}

        if resource_id:
            details["id"] = resource_id

        raise NotFoundError(
            f"The requested {resource_name} was not found.",
            details=details,
        )

    return record


async def list_records(
    db: AsyncSession,
    query: Select,
    *,
    skip: int = 0,
    limit: int = 100,
) -> Sequence[ModelT]:
    """
    Execute a query with pagination and return model records.

    Args:
        db: Active async database session.
        query: SQLAlchemy select query.
        skip: Number of records to skip.
        limit: Maximum number of records to return.

    Returns:
        A sequence of model records.
    """
    paginated_query = apply_pagination(
        query,
        skip=skip,
        limit=limit,
    )

    result = await db.execute(paginated_query)

    return result.scalars().all()


async def count_records(
    db: AsyncSession,
    model: type[ModelT],
    *filters: Any,
) -> int:
    """
    Count records matching optional filters.

    Args:
        db: Active async database session.
        model: SQLAlchemy model class.
        filters: Optional SQLAlchemy filter expressions.

    Returns:
        Total matching record count.
    """
    query = build_count_query(model, *filters)

    result = await db.execute(query)

    return int(result.scalar_one())


async def exists(
    db: AsyncSession,
    model: type[ModelT],
    *filters: Any,
) -> bool:
    """
    Check whether at least one record matches the supplied filters.

    Args:
        db: Active async database session.
        model: SQLAlchemy model class.
        filters: Optional SQLAlchemy filter expressions.

    Returns:
        `True` when at least one matching record exists.
    """
    query = select(func.count()).select_from(model)

    if filters:
        query = query.where(*filters)

    result = await db.execute(query)

    return int(result.scalar_one()) > 0


# ---------------------------------------------------------------------------
# Create operations
# ---------------------------------------------------------------------------


async def create_record(
    db: AsyncSession,
    record: ModelT,
) -> ModelT:
    """
    Persist a new record and refresh it from the database.

    Args:
        db: Active async database session.
        record: SQLAlchemy model instance.

    Returns:
        The persisted and refreshed record.

    Raises:
        DatabaseIntegrityError: If a uniqueness or foreign-key constraint fails.
        PersistenceError: If another database error occurs.
    """
    try:
        db.add(record)
        await db.commit()
        await db.refresh(record)
    except IntegrityError as exc:
        await db.rollback()

        logger.exception(
            "crud_create_integrity_error",
            model=type(record).__name__,
        )

        raise DatabaseIntegrityError(
            "The record could not be created because it violates a data constraint."
        ) from exc

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "crud_create_failed",
            model=type(record).__name__,
        )

        raise PersistenceError(
            "The record could not be created."
        ) from exc

    logger.debug(
        "crud_record_created",
        model=type(record).__name__,
        record_id=str(getattr(record, "id", "")),
    )

    return record


async def create_records_bulk(
    db: AsyncSession,
    records: Sequence[ModelT],
) -> Sequence[ModelT]:
    """
    Persist multiple records in one transaction.

    Args:
        db: Active async database session.
        records: Model instances to insert.

    Returns:
        The persisted records.

    Raises:
        ValidationError: If no records are supplied.
        DatabaseIntegrityError: If any record violates a constraint.
        PersistenceError: If another database error occurs.
    """
    if not records:
        raise ValidationError("At least one record must be supplied.")

    try:
        db.add_all(records)
        await db.commit()

        for record in records:
            await db.refresh(record)

    except IntegrityError as exc:
        await db.rollback()

        logger.exception(
            "crud_bulk_create_integrity_error",
            model=type(records[0]).__name__,
            record_count=len(records),
        )

        raise DatabaseIntegrityError(
            "One or more records could not be created because they violate data constraints."
        ) from exc

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "crud_bulk_create_failed",
            model=type(records[0]).__name__,
            record_count=len(records),
        )

        raise PersistenceError(
            "One or more records could not be created."
        ) from exc

    logger.info(
        "crud_records_created",
        model=type(records[0]).__name__,
        record_count=len(records),
    )

    return records


# ---------------------------------------------------------------------------
# Update operations
# ---------------------------------------------------------------------------


async def update_record(
    db: AsyncSession,
    record: ModelT,
) -> ModelT:
    """
    Persist changes to an existing record.

    Args:
        db: Active async database session.
        record: SQLAlchemy model instance with changes applied.

    Returns:
        The refreshed record.

    Raises:
        DatabaseIntegrityError: If a constraint is violated.
        PersistenceError: If another database error occurs.
    """
    try:
        db.add(record)
        await db.commit()
        await db.refresh(record)
    except IntegrityError as exc:
        await db.rollback()

        logger.exception(
            "crud_update_integrity_error",
            model=type(record).__name__,
            record_id=str(getattr(record, "id", "")),
        )

        raise DatabaseIntegrityError(
            "The record could not be updated because it violates a data constraint."
        ) from exc

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "crud_update_failed",
            model=type(record).__name__,
            record_id=str(getattr(record, "id", "")),
        )

        raise PersistenceError(
            "The record could not be updated."
        ) from exc

    logger.debug(
        "crud_record_updated",
        model=type(record).__name__,
        record_id=str(getattr(record, "id", "")),
    )

    return record


async def update_fields_by_id(
    db: AsyncSession,
    model: type[ModelT],
    record_id: UUID | str,
    values: dict[str, Any],
) -> ModelT:
    """
    Update selected fields on a record by ID.

    Args:
        db: Active async database session.
        model: SQLAlchemy model class.
        record_id: Record UUID.
        values: Field values to update.

    Returns:
        The refreshed updated record.

    Raises:
        ValidationError: If no fields are supplied.
        NotFoundError: If the record does not exist.
        OptimisticLockError: If the record was modified concurrently.
        PersistenceError: If another database error occurs.
    """
    if not values:
        raise ValidationError("At least one field must be supplied.")

    record_uuid = normalize_uuid(record_id)

    values = {
        **values,
        "updated_at": datetime.utcnow(),
    }

    result = await db.execute(
        update(model)
        .where(model.id == record_uuid)
        .values(**values)
        .returning(model)
    )

    updated_record = result.scalar_one_or_none()

    if updated_record is None:
        await db.rollback()

        raise NotFoundError(
            "The requested resource was not found.",
            details={"id": str(record_uuid)},
        )

    try:
        await db.commit()
        await db.refresh(updated_record)
    except IntegrityError as exc:
        await db.rollback()

        logger.exception(
            "crud_update_fields_integrity_error",
            model=model.__name__,
            record_id=str(record_uuid),
        )

        raise DatabaseIntegrityError(
            "The record could not be updated because it violates a data constraint."
        ) from exc

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "crud_update_fields_failed",
            model=model.__name__,
            record_id=str(record_uuid),
        )

        raise PersistenceError(
            "The record could not be updated."
        ) from exc

    logger.info(
        "crud_record_fields_updated",
        model=model.__name__,
        record_id=str(record_uuid),
        updated_fields=sorted(values),
    )

    return updated_record


async def update_with_version(
    db: AsyncSession,
    model: type[ModelT],
    record_id: UUID | str,
    *,
    expected_version: int,
    values: dict[str, Any],
) -> ModelT:
    """
    Update a record using optimistic concurrency control.

    The model must contain an integer `version` column.

    Args:
        db: Active async database session.
        model: SQLAlchemy model class.
        record_id: Record UUID.
        expected_version: Version value read by the caller.
        values: Field values to update.

    Returns:
        The refreshed updated record.

    Raises:
        ValidationError: If no fields are supplied or version is invalid.
        OptimisticLockError: If the version does not match.
        PersistenceError: If another database error occurs.
    """
    if not values:
        raise ValidationError("At least one field must be supplied.")

    if expected_version < 0:
        raise ValidationError("expected_version must be non-negative.")

    record_uuid = normalize_uuid(record_id)

    next_version = expected_version + 1

    result = await db.execute(
        update(model)
        .where(
            model.id == record_uuid,
            model.version == expected_version,
        )
        .values(
            **values,
            version=next_version,
            updated_at=datetime.utcnow(),
        )
        .returning(model)
    )

    updated_record = result.scalar_one_or_none()

    if updated_record is None:
        await db.rollback()

        raise OptimisticLockError(
            "The record was modified by another request. Please refresh and retry."
        )

    try:
        await db.commit()
        await db.refresh(updated_record)
    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "crud_optimistic_update_failed",
            model=model.__name__,
            record_id=str(record_uuid),
        )

        raise PersistenceError(
            "The record could not be updated."
        ) from exc

    logger.info(
        "crud_record_updated_with_version",
        model=model.__name__,
        record_id=str(record_uuid),
        previous_version=expected_version,
        new_version=next_version,
    )

    return updated_record


# ---------------------------------------------------------------------------
# Delete operations
# ---------------------------------------------------------------------------


async def delete_record(
    db: AsyncSession,
    record: ModelT,
) -> None:
    """
    Delete an existing record.

    Args:
        db: Active async database session.
        record: SQLAlchemy model instance to delete.

    Raises:
        DatabaseIntegrityError: If related records prevent deletion.
        PersistenceError: If another database error occurs.
    """
    record_id = str(getattr(record, "id", ""))

    try:
        await db.delete(record)
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()

        logger.exception(
            "crud_delete_integrity_error",
            model=type(record).__name__,
            record_id=record_id,
        )

        raise DatabaseIntegrityError(
            "The record could not be deleted because related records exist."
        ) from exc

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "crud_delete_failed",
            model=type(record).__name__,
            record_id=record_id,
        )

        raise PersistenceError(
            "The record could not be deleted."
        ) from exc

    logger.info(
        "crud_record_deleted",
        model=type(record).__name__,
        record_id=record_id,
    )


async def delete_by_id(
    db: AsyncSession,
    model: type[ModelT],
    record_id: UUID | str,
) -> None:
    """
    Delete a record by primary key.

    Args:
        db: Active async database session.
        model: SQLAlchemy model class.
        record_id: Record UUID.

    Raises:
        NotFoundError: If the record does not exist.
        DatabaseIntegrityError: If related records prevent deletion.
        PersistenceError: If another database error occurs.
    """
    record = await get_by_id_or_404(db, model, record_id)

    await delete_record(db, record)


async def delete_many(
    db: AsyncSession,
    model: type[ModelT],
    *filters: Any,
) -> int:
    """
    Delete all records matching filters.

    Args:
        db: Active async database session.
        model: SQLAlchemy model class.
        filters: Required SQLAlchemy filter expressions.

    Returns:
        Number of deleted records.

    Raises:
        ValidationError: If no filters are supplied.
        DatabaseIntegrityError: If constraints prevent deletion.
        PersistenceError: If another database error occurs.
    """
    if not filters:
        raise ValidationError(
            "At least one filter is required for bulk deletion."
        )

    query = delete(model).where(*filters)

    try:
        result = await db.execute(query)
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()

        logger.exception(
            "crud_bulk_delete_integrity_error",
            model=model.__name__,
        )

        raise DatabaseIntegrityError(
            "Records could not be deleted because related records exist."
        ) from exc

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception(
            "crud_bulk_delete_failed",
            model=model.__name__,
        )

        raise PersistenceError(
            "Records could not be deleted."
        ) from exc

    deleted_count = int(result.rowcount or 0)

    logger.warning(
        "crud_records_bulk_deleted",
        model=model.__name__,
        deleted_count=deleted_count,
    )

    return deleted_count


# ---------------------------------------------------------------------------
# Transactional helpers
# ---------------------------------------------------------------------------


async def execute_query(
    db: AsyncSession,
    query: Select | Insert | Update | Delete,
) -> Any:
    """
    Execute a SQLAlchemy query without committing.

    Use this only for specialized queries. Prefer the typed helpers above for
    standard CRUD operations.
    """
    try:
        return await db.execute(query)
    except SQLAlchemyError as exc:
        logger.exception(
            "crud_query_execution_failed",
            query_type=type(query).__name__,
        )

        raise PersistenceError(
            "The database query could not be completed."
        ) from exc


async def commit_session(db: AsyncSession) -> None:
    """Commit the current database transaction."""
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()

        raise DatabaseIntegrityError(
            "The transaction could not be committed because it violates a data constraint."
        ) from exc

    except SQLAlchemyError as exc:
        await db.rollback()

        logger.exception("crud_commit_failed")

        raise PersistenceError(
            "The database transaction could not be committed."
        ) from exc


async def rollback_session(db: AsyncSession) -> None:
    """Roll back the current database transaction."""
    try:
        await db.rollback()
    except SQLAlchemyError:
        logger.exception("crud_rollback_failed")