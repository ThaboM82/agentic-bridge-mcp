import logging
import uuid
import json
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.graph.workflow import app_workflow

logger = logging.getLogger("agentic_bridge.routers.workflow")

router = APIRouter(prefix="/workflow", tags=["Workflow Orchestration"])

# In-memory execution state cache for async tracking (replaceable with PostgreSQL table)
_EXECUTION_STORE: Dict[str, Dict[str, Any]] = {}

class WorkflowRequestSchema(BaseModel):
    """Pydantic request schema for triggering the LangGraph agent pipeline."""
    brief_id: Optional[str] = Field(default=None, description="Optional unique identifier for the project brief.")
    raw_brief_text: str = Field(description="The unstructured product brief text to be ingested and analyzed.")

class WorkflowResponseSchema(BaseModel):
    """Pydantic response schema returning execution results and architectural specs."""
    execution_id: str
    brief_id: str
    current_step: str
    extracted_requirements: List[str] = []
    architectural_specification: Optional[Dict[str, Any]] = None
    design_tokens: Dict[str, Any] = {}
    execution_logs: List[str] = []
    current_error: Optional[str] = None

@router.post("/execute", response_model=WorkflowResponseSchema)
async def execute_workflow(payload: WorkflowRequestSchema) -> Dict[str, Any]:
    """
    Triggers the enterprise LangGraph pipeline synchronously, executing ingestion and processing nodes.
    """
    execution_id = f"exec_{uuid.uuid4().hex[:8]}"
    brief_id = payload.brief_id or f"brief_{uuid.uuid4().hex[:6]}"
    
    logger.info(f"[{execution_id}] Received synchronous execution request for brief ID: {brief_id}")

    initial_state = {
        "execution_id": execution_id,
        "brief_id": brief_id,
        "raw_brief_text": payload.raw_brief_text,
        "extracted_requirements": [],
        "architectural_specification": None,
        "design_tokens": {},
        "execution_logs": [f"INFO: Initialized synchronous execution {execution_id}"],
        "current_error": None,
        "current_step": "initialized"
    }

    try:
        final_state = app_workflow.invoke(initial_state)
        _EXECUTION_STORE[execution_id] = final_state

        return {
            "execution_id": execution_id,
            "brief_id": brief_id,
            "current_step": final_state.get("current_step", "completed"),
            "extracted_requirements": final_state.get("extracted_requirements", []),
            "architectural_specification": final_state.get("architectural_specification"),
            "design_tokens": final_state.get("design_tokens", {}),
            "execution_logs": final_state.get("execution_logs", []),
            "current_error": final_state.get("current_error")
        }

    except Exception as e:
        error_msg = f"Critical failure executing LangGraph workflow: {str(e)}"
        logger.exception(f"[{execution_id}] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


async def _run_workflow_background(execution_id: str, state: Dict[str, Any]) -> None:
    """Background task runner for asynchronous graph execution."""
    try:
        logger.info(f"[{execution_id}] Background workflow execution started.")
        final_state = app_workflow.invoke(state)
        _EXECUTION_STORE[execution_id] = final_state
        logger.info(f"[{execution_id}] Background workflow execution completed successfully.")
    except Exception as e:
        logger.exception(f"[{execution_id}] Background execution failed: {str(e)}")
        _EXECUTION_STORE[execution_id] = {**state, "current_error": str(e), "current_step": "error"}


@router.post("/execute/async", status_code=status.HTTP_202_ACCEPTED)
async def execute_workflow_async(payload: WorkflowRequestSchema, background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Triggers the LangGraph pipeline asynchronously in the background, returning an execution handle immediately.
    """
    execution_id = f"exec_async_{uuid.uuid4().hex[:8]}"
    brief_id = payload.brief_id or f"brief_{uuid.uuid4().hex[:6]}"
    
    initial_state = {
        "execution_id": execution_id,
        "brief_id": brief_id,
        "raw_brief_text": payload.raw_brief_text,
        "extracted_requirements": [],
        "architectural_specification": None,
        "design_tokens": {},
        "execution_logs": [f"INFO: Initialized asynchronous execution {execution_id}"],
        "current_error": None,
        "current_step": "queued"
    }

    _EXECUTION_STORE[execution_id] = initial_state
    background_tasks.add_task(_run_workflow_background, execution_id, initial_state)

    logger.info(f"[{execution_id}] Queued async workflow execution for brief ID: {brief_id}")
    return {
        "execution_id": execution_id,
        "status": "queued",
        "poll_endpoint": f"/api/v1/workflow/status/{execution_id}"
    }


@router.get("/status/{execution_id}", response_model=WorkflowResponseSchema)
async def get_execution_status(execution_id: str) -> Dict[str, Any]:
    """Retrieves the current execution state and results for an asynchronous workflow run."""
    if execution_id not in _EXECUTION_STORE:
        raise HTTPException(status_code=404, detail=f"Execution ID '{execution_id}' not found.")
    
    state = _EXECUTION_STORE[execution_id]
    return {
        "execution_id": execution_id,
        "brief_id": state.get("brief_id", "unknown"),
        "current_step": state.get("current_step", "unknown"),
        "extracted_requirements": state.get("extracted_requirements", []),
        "architectural_specification": state.get("architectural_specification"),
        "design_tokens": state.get("design_tokens", {}),
        "execution_logs": state.get("execution_logs", []),
        "current_error": state.get("current_error")
    }


@router.get("/stream/{execution_id}")
async def stream_execution_progress(execution_id: str) -> StreamingResponse:
    """
    Streams live progress updates and execution logs via Server-Sent Events (SSE).
    """
    async def event_generator():
        yield f"data: {json.dumps({'event': 'connected', 'execution_id': execution_id})}\n\n"
        
        while True:
            state = _EXECUTION_STORE.get(execution_id)
            if not state:
                yield f"data: {json.dumps({'error': 'Execution not found'})}\n\n"
                break
                
            step = state.get("current_step", "processing")
            logs = state.get("execution_logs", [])
            latest_log = logs[-1] if logs else "Processing..."
            
            yield f"data: {json.dumps({'step': step, 'latest_log': latest_log})}\n\n"
            
            if step in ["completed", "error", "END"]:
                break
                
            await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/health")
async def workflow_router_health() -> Dict[str, Any]:
    """Audits router health and active execution cache size."""
    return {
        "status": "healthy",
        "active_cached_executions": len(_EXECUTION_STORE),
        "router": "workflow_router"
    }