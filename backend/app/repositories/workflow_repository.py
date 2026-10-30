import logging
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from app.models.workflow_model import WorkflowExecutionModel

logger = logging.getLogger("agentic_bridge.repositories.workflow")

class WorkflowRepository:
    """
    Enterprise-grade repository class encapsulating asynchronous database operations,
    pagination, vector similarity searches, and telemetry analytics for workflow executions.
    """

    @staticmethod
    async def create_execution(
        db: AsyncSession,
        execution_id: str,
        brief_id: str,
        raw_brief_text: str,
        embedding: Optional[List[float]] = None
    ) -> WorkflowExecutionModel:
        """Creates and persists a new workflow execution record with initial state."""
        logger.info(f"Persisting new workflow execution record: {execution_id}")
        
        db_record = WorkflowExecutionModel(
            execution_id=execution_id,
            brief_id=brief_id,
            current_step="initialized",
            status="running",
            execution_logs=[f"INFO: Initialized execution record {execution_id}"],
            brief_embedding=embedding
        )
        
        db.add(db_record)
        await db.flush()
        await db.refresh(db_record)
        return db_record

    @staticmethod
    async def get_by_execution_id(db: AsyncSession, execution_id: str) -> Optional[WorkflowExecutionModel]:
        """Retrieves a workflow execution record by its unique execution ID."""
        stmt = select(WorkflowExecutionModel).where(WorkflowExecutionModel.execution_id == execution_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def list_executions(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 20,
        status_filter: Optional[str] = None
    ) -> Tuple[List[WorkflowExecutionModel], int]:
        """
        Retrieves a paginated list of workflow execution records with optional status filtering,
        returning both the records and the total matching count.
        """
        query = select(WorkflowExecutionModel)
        count_query = select(func.count(WorkflowExecutionModel.id))

        if status_filter:
            query = query.where(WorkflowExecutionModel.status == status_filter)
            count_query = count_query.where(WorkflowExecutionModel.status == status_filter)

        # Execute count query
        total_count_result = await db.execute(count_query)
        total_count = total_count_result.scalar_one()

        # Execute paginated query ordered by creation date descending
        paginated_stmt = query.order_by(WorkflowExecutionModel.created_at.desc()).offset(skip).limit(limit)
        result = await db.execute(paginated_stmt)
        records = list(result.scalars().all())

        return records, total_count

    @staticmethod
    async def update_execution_state(
        db: AsyncSession,
        execution_id: str,
        state_updates: Dict[str, Any]
    ) -> Optional[WorkflowExecutionModel]:
        """Updates an existing workflow execution record with new state data from LangGraph."""
        logger.debug(f"Updating database state for execution: {execution_id}")
        
        db_record = await WorkflowRepository.get_by_execution_id(db, execution_id)
        if not db_record:
            logger.warning(f"Execution ID '{execution_id}' not found for state update.")
            return None

        # Map state keys to ORM model attributes safely
        if "current_step" in state_updates:
            db_record.current_step = state_updates["current_step"]
        if "extracted_requirements" in state_updates:
            db_record.extracted_requirements = state_updates["extracted_requirements"]
        if "architectural_specification" in state_updates:
            db_record.architectural_specification = state_updates["architectural_specification"]
        if "design_tokens" in state_updates:
            db_record.design_tokens = state_updates["design_tokens"]
        if "execution_logs" in state_updates:
            db_record.execution_logs = state_updates["execution_logs"]
        if "current_error" in state_updates:
            error_val = state_updates["current_error"]
            db_record.current_error = error_val
            if error_val:
                db_record.status = "error"
        
        if db_record.current_step in ["completed", "END"] and not db_record.current_error:
            db_record.status = "completed"

        await db.flush()
        await db.refresh(db_record)
        return db_record

    @staticmethod
    async def delete_execution(db: AsyncSession, execution_id: str) -> bool:
        """Deletes a workflow execution record by its unique execution ID."""
        logger.info(f"Deleting workflow execution record: {execution_id}")
        db_record = await WorkflowRepository.get_by_execution_id(db, execution_id)
        if not db_record:
            return False

        await db.delete(db_record)
        await db.flush()
        return True

    @staticmethod
    async def search_similar_briefs(
        db: AsyncSession,
        query_embedding: List[float],
        limit: int = 5
    ) -> List[WorkflowExecutionModel]:
        """Performs a pgvector cosine distance similarity search against stored brief embeddings."""
        logger.info("Executing pgvector similarity search on workflow executions.")
        
        stmt = (
            select(WorkflowExecutionModel)
            .where(WorkflowExecutionModel.brief_embedding.is_not(None))
            .order_by(WorkflowExecutionModel.brief_embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_execution_analytics(db: AsyncSession) -> Dict[str, Any]:
        """Aggregates telemetry and status statistics across all workflow executions."""
        total_stmt = select(func.count(WorkflowExecutionModel.id))
        completed_stmt = select(func.count(WorkflowExecutionModel.id)).where(WorkflowExecutionModel.status == "completed")
        error_stmt = select(func.count(WorkflowExecutionModel.id)).where(WorkflowExecutionModel.status == "error")
        running_stmt = select(func.count(WorkflowExecutionModel.id)).where(WorkflowExecutionModel.status == "running")

        total = (await db.execute(total_stmt)).scalar_one()
        completed = (await db.execute(completed_stmt)).scalar_one()
        errors = (await db.execute(error_stmt)).scalar_one()
        running = (await db.execute(running_stmt)).scalar_one()

        success_rate = round((completed / total * 100), 2) if total > 0 else 0.0

        return {
            "total_executions": total,
            "completed_executions": completed,
            "error_executions": errors,
            "running_executions": running,
            "success_rate_percentage": success_rate
        }