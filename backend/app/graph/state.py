from typing import TypedDict, List, Dict, Any, Optional
from operator import add
from typing_extensions import Annotated

class AgenticState(TypedDict):
    """
    Defines the centralized state schema passed across nodes 
    in the LangGraph workflow execution pipeline for AgenticBridge-MCP.
    """
    # Unique identifiers and tracking
    execution_id: str
    brief_id: str
    
    # Ingestion layer outputs
    raw_brief_text: str
    extracted_requirements: List[str]
    
    # Design & Context layer outputs
    design_tokens: Dict[str, Any]
    vector_context_matches: List[str]
    
    # Code generation layer outputs
    generated_code_files: Dict[str, str]  # Mapping of filename to source code content
    
    # Governance & Human-in-the-Loop (HITL)
    review_status: str  # e.g., "pending_approval", "approved", "rejected"
    approval_comment: Optional[str]
    
    # Execution log history with reducer to accumulate status updates across nodes
    execution_logs: Annotated[List[str], add]
    
    # Error handling and resilience
    current_error: Optional[str]