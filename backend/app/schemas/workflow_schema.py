from typing import Any, Optional

from pydantic import BaseModel, Field


class WorkflowRequestSchema(BaseModel):
    """Request schema for triggering an AgenticBridge workflow."""

    brief_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional stable identifier for the source brief.",
    )
    raw_brief_text: str = Field(
        ...,
        min_length=10,
        max_length=100_000,
        description="Unstructured product brief text to ingest and analyze.",
    )


class WorkflowResponseSchema(BaseModel):
    """Response schema for a workflow execution."""

    execution_id: str
    brief_id: str
    current_step: str
    extracted_requirements: list[str] = []
    architectural_specification: Optional[dict[str, Any]] = None
    design_tokens: dict[str, Any] = {}
    execution_logs: list[str] = []
    current_error: Optional[str] = None