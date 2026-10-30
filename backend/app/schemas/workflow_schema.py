from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

class WorkflowExecutionBase(BaseModel):
    """Shared base schema for workflow execution properties."""
    workflow_name: str = Field(..., description="Name or identifier of the agentic workflow")
    parent_execution_id: Optional[str] = Field(default=None, description="UUID of parent workflow for nested multi-agent sub-graphs")
    agent_version: Optional[str] = Field(default="1.0.0", description="Version string of the executing agent")
    model_name: Optional[str] = Field(default=None, description="LLM model identifier (e.g., gpt-4o, claude-3-5-sonnet)")
    input_payload: Optional[Dict[str, Any]] = Field(default=None, description="Input parameters and context payload provided to the agent")
    extra_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata tags, tracing IDs, or configuration parameters")

class WorkflowExecutionCreate(WorkflowExecutionBase):
    """Schema for initiating a new workflow execution request."""
    embedding: Optional[List[float]] = Field(
        default=None, 
        description="Optional semantic vector embedding (e.g. 1536 dimensions) for vector search"
    )

class WorkflowExecutionUpdate(BaseModel):
    """Schema for updating an active or completed workflow execution state."""
    status: Optional[str] = Field(
        default=None, 
        description="Execution status: pending, running, completed, failed, cancelled, or paused"
    )
    output_result: Optional[Dict[str, Any]] = Field(default=None, description="Final output data or results produced by the agent")
    execution_steps: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="Chronological step-by-step telemetry logs and intermediate agent reasoning traces"
    )
    error_message: Optional[str] = Field(default=None, description="Detailed stack trace or error message if execution failed")
    prompt_tokens: Optional[int] = Field(default=None, description="Number of prompt tokens consumed")
    completion_tokens: Optional[int] = Field(default=None, description="Number of completion tokens generated")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens consumed")
    completed_at: Optional[datetime] = Field(default=None, description="Timestamp when the execution completed or failed")

class WorkflowExecutionResponse(WorkflowExecutionBase):
    """Schema for serializing workflow execution details in API responses."""
    id: str = Field(..., description="Unique UUID string identifier for the execution record")
    status: str = Field(..., description="Current status of the workflow execution")
    output_result: Optional[Dict[str, Any]] = Field(default=None)
    execution_steps: Optional[List[Dict[str, Any]]] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    prompt_tokens: Optional[int] = Field(default=0)
    completion_tokens: Optional[int] = Field(default=0)
    total_tokens: Optional[int] = Field(default=0)
    embedding: Optional[List[float]] = Field(default=None, description="Stored vector embedding from pgvector")
    created_at: datetime = Field(..., description="Timestamp when the execution record was created")
    updated_at: datetime = Field(..., description="Timestamp when the record was last updated")
    completed_at: Optional[datetime] = Field(default=None)

    # Pydantic v2 configuration to enable automatic ORM model serialization
    model_config = ConfigDict(from_attributes=True)