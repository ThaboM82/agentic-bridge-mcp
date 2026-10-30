import uuid
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
)
from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Integer,
    JSON,
    Index,
    func,
)
from pgvector.sqlalchemy import Vector
from app.database import Base

class WorkflowExecutionModel(Base):
    """
    Enterprise-grade ORM model for tracking autonomous agentic workflow executions,
    hierarchical sub-graph tracing, step telemetry logs, LLM token utilization, 
    and semantic vector embeddings.
    """
    __tablename__ = "workflow_executions"

    # Primary Identifiers & Status
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    parent_execution_id = Column(String(36), nullable=True, index=True)  # For nested/hierarchical multi-agent sub-graphs
    workflow_name = Column(String(255), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="pending", index=True)  # pending, running, completed, failed, cancelled, paused
    
    # Agent & Model Telemetry
    agent_version = Column(String(50), nullable=True, default="1.0.0")
    model_name = Column(String(100), nullable=True)  # e.g., gpt-4o, claude-3-5-sonnet
    
    # Execution Payloads & Results
    input_payload = Column(JSON, nullable=True)
    output_result = Column(JSON, nullable=True)
    execution_steps = Column(JSON, nullable=True)  # Chronological telemetry logs and agent reasoning traces
    error_message = Column(Text, nullable=True)
    
    # Token Usage & Cost Auditing
    prompt_tokens = Column(Integer, nullable=True, default=0)
    completion_tokens = Column(Integer, nullable=True, default=0)
    total_tokens = Column(Integer, nullable=True, default=0)

    # Vector Embedding for Semantic RAG / Agent Memory Retrieval (e.g., 1536 dimensions)
    embedding = Column(Vector(1536), nullable=True)

    # Additional Metadata & Tracing Tags
    extra_metadata = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Composite Indexes for High-Performance Queries & Vector Similarity Filtering
    __table_args__ = (
        Index("idx_workflow_name_status", "workflow_name", "status"),
        Index("idx_workflow_created_at", "created_at"),
        Index("idx_parent_execution", "parent_execution_id"),
    )

    def __repr__(self) -> str:
        return f"<WorkflowExecutionModel(id='{self.id}', name='{self.workflow_name}', status='{self.status}')>"