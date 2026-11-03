import asyncio
import logging
import uuid
from typing import Any, Dict

from app.graph.workflow import app_workflow
from app.schemas.workflow_schema import WorkflowRequestSchema


logger = logging.getLogger(__name__)


class WorkflowNotFoundError(Exception):
    """Raised when a workflow execution cannot be found."""


class WorkflowExecutionError(Exception):
    """Raised when a workflow fails during execution."""


class InMemoryWorkflowRepository:
    """
    Development-grade execution state repository.

    Replace with PostgreSQL/Redis-backed repository for production.
    This store is process-local and will not work with multiple workers.
    """

    def __init__(self) -> None:
        self._executions: Dict[str, Dict[str, Any]] = {}

    async def create(self, execution_id: str, state: Dict[str, Any]) -> None:
        self._executions[execution_id] = state.copy()

    async def update(self, execution_id: str, state: Dict[str, Any]) -> None:
        self._executions[execution_id] = state.copy()

    async def get(self, execution_id: str) -> Dict[str, Any]:
        if execution_id not in self._executions:
            raise WorkflowNotFoundError(
                f"Workflow execution '{execution_id}' was not found."
            )
        return self._executions[execution_id]

    async def count(self) -> int:
        return len(self._executions)


class WorkflowService:
    """Orchestrates AgenticBridge LangGraph workflow executions."""

    def __init__(self, workflow=app_workflow, repository=None) -> None:
        self.workflow = workflow
        self.repository = repository or InMemoryWorkflowRepository()

    def _build_initial_state(
        self,
        execution_id: str,
        payload: WorkflowRequestSchema,
        current_step: str,
    ) -> Dict[str, Any]:
        return {
            "execution_id": execution_id,
            "brief_id": payload.brief_id,
            "raw_brief_text": payload.raw_brief_text,
            "extracted_requirements": [],
            "architectural_specification": None,
            "design_tokens": {},
            "execution_logs": [
                f"INFO: Initialized workflow execution {execution_id}"
            ],
            "current_error": None,
            "current_step": current_step,
        }

    def _generate_ids(self, payload: WorkflowRequestSchema) -> tuple[str, str]:
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        brief_id = payload.brief_id or f"brief_{uuid.uuid4().hex[:8]}"
        return execution_id, brief_id

    @staticmethod
    def _to_response(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "execution_id": state["execution_id"],
            "brief_id": state.get("brief_id", "unknown"),
            "current_step": state.get("current_step", "unknown"),
            "extracted_requirements": state.get("extracted_requirements", []),
            "architectural_specification": state.get(
                "architectural_specification"
            ),
            "design_tokens": state.get("design_tokens", {}),
            "execution_logs": state.get("execution_logs", []),
            "current_error": state.get("current_error"),
        }

    async def execute_sync(
        self,
        payload: WorkflowRequestSchema,
    ) -> Dict[str, Any]:
        """Run the workflow synchronously and return its final state."""
        execution_id, brief_id = self._generate_ids(payload)
        payload.brief_id = brief_id

        initial_state = self._build_initial_state(
            execution_id=execution_id,
            payload=payload,
            current_step="initialized",
        )

        await self.repository.create(execution_id, initial_state)

        logger.info(
            "Starting synchronous workflow execution_id=%s brief_id=%s",
            execution_id,
            brief_id,
        )

        try:
            final_state = await self.workflow.ainvoke(initial_state)
            final_state["execution_id"] = execution_id

            await self.repository.update(execution_id, final_state)

            logger.info(
                "Completed workflow execution_id=%s step=%s",
                execution_id,
                final_state.get("current_step"),
            )

            return self._to_response(final_state)

        except Exception as exc:
            logger.exception(
                "Workflow execution failed execution_id=%s", execution_id
            )

            failed_state = {
                **initial_state,
                "current_step": "error",
                "current_error": str(exc),
                "execution_logs": initial_state["execution_logs"]
                + [f"ERROR: Workflow execution failed: {exc}"],
            }

            await self.repository.update(execution_id, failed_state)

            raise WorkflowExecutionError(
                "Workflow execution failed."
            ) from exc

    async def start_async(
        self,
        payload: WorkflowRequestSchema,
    ) -> Dict[str, Any]:
        """
        Persist a queued workflow and schedule background execution.

        For production, replace BackgroundTasks with a durable queue/worker.
        """
        execution_id, brief_id = self._generate_ids(payload)
        payload.brief_id = brief_id

        initial_state = self._build_initial_state(
            execution_id=execution_id,
            payload=payload,
            current_step="queued",
        )

        await self.repository.create(execution_id, initial_state)

        asyncio.create_task(
            self._execute_background(execution_id, initial_state)
        )

        logger.info(
            "Queued async workflow execution_id=%s brief_id=%s",
            execution_id,
            brief_id,
        )

        return {
            "execution_id": execution_id,
            "brief_id": brief_id,
            "status": "queued",
            "poll_endpoint": f"/api/v1/workflow/status/{execution_id}",
            "stream_endpoint": f"/api/v1/workflow/stream/{execution_id}",
        }

    async def _execute_background(
        self,
        execution_id: str,
        state: Dict[str, Any],
    ) -> None:
        """Execute a queued workflow outside the request/response cycle."""
        try:
            logger.info(
                "Background workflow started execution_id=%s", execution_id
            )

            final_state = await self.workflow.ainvoke(state)
            final_state["execution_id"] = execution_id

            await self.repository.update(execution_id, final_state)

            logger.info(
                "Background workflow completed execution_id=%s", execution_id
            )

        except Exception as exc:
            logger.exception(
                "Background workflow failed execution_id=%s", execution_id
            )

            failed_state = {
                **state,
                "current_step": "error",
                "current_error": str(exc),
                "execution_logs": state.get("execution_logs", [])
                + [f"ERROR: Background workflow failed: {exc}"],
            }

            await self.repository.update(execution_id, failed_state)

    async def get_execution(self, execution_id: str) -> Dict[str, Any]:
        """Return the latest known state for an execution."""
        state = await self.repository.get(execution_id)
        return self._to_response(state)

    async def get_latest_log(self, execution_id: str) -> Dict[str, Any]:
        """Return a compact progress event for SSE consumers."""
        state = await self.repository.get(execution_id)

        logs = state.get("execution_logs", [])
        latest_log = logs[-1] if logs else "Processing..."

        return {
            "execution_id": execution_id,
            "step": state.get("current_step", "processing"),
            "latest_log": latest_log,
        }

    async def count_active_executions(self) -> int:
        """Return number of executions currently held by the repository."""
        return await self.repository.count()


workflow_service = WorkflowService()