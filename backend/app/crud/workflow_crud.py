from typing import (
    List,
    Optional,
)
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.workflow_model import WorkflowExecutionModel
from app.schemas.workflow_schema import (
    WorkflowExecutionCreate,
    WorkflowExecutionUpdate,
)

logger = logging.getLogger("agentic_bridge.crud.workflow")

async def create_workflow_execution(
    db: AsyncSession,
    obj_in: WorkflowExecutionCreate,
) -> WorkflowExecutionModel:
    """Create and persist a new workflow execution record."""
    try:
        db_obj = WorkflowExecutionModel(
            workflow_name=obj_in.workflow_name,
            input_payload=obj_in.input_payload,
            extra_metadata=obj_in.extra_metadata,
            embedding=obj_in.embedding,
            status="pending",
        )
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        logger.info(f"Created workflow execution ID: {db_obj.id} for workflow: {db_obj.workflow_name}")
        return db_obj
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to create workflow execution: {e}", exc_info=True)
        raise

async def get_workflow_execution(
    db: AsyncSession,
    workflow_id: str,
) -> Optional[WorkflowExecutionModel]:
    """Retrieve a single workflow execution record by its primary UUID string."""
    result = await db.execute(
        select(WorkflowExecutionModel).where(WorkflowExecutionModel.id == workflow_id)
    )
    return result.scalars().first()

async def list_workflow_executions(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
) -> List[WorkflowExecutionModel]:
    """Retrieve a paginated list of workflow executions, optionally filtered by status."""
    query = select(WorkflowExecutionModel)
    if status:
        query = query.where(WorkflowExecutionModel.status == status)
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()

async def update_workflow_execution(
    db: AsyncSession,
    db_obj: WorkflowExecutionModel,
    obj_in: WorkflowExecutionUpdate,
) -> WorkflowExecutionModel:
    """Update an existing workflow execution record with new state or telemetry."""
    try:
        update_data = obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
            
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        logger.info(f"Updated workflow execution ID: {db_obj.id} to status: {db_obj.status}")
        return db_obj
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to update workflow execution ID {db_obj.id}: {e}", exc_info=True)
        raise

async def delete_workflow_execution(
    db: AsyncSession,
    db_obj: WorkflowExecutionModel,
) -> None:
    """Delete a workflow execution record."""
    try:
        await db.delete(db_obj)
        await db.commit()
        logger.info(f"Deleted workflow execution ID: {db_obj.id}")
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to delete workflow execution ID {db_obj.id}: {e}", exc_info=True)
        raise