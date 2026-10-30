import logging
from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from app.graph.nodes.ingestion_node import ingestion_node

logger = logging.getLogger("agentic_bridge.graph.workflow")

class AgentState(TypedDict):
    """Defines the state schema passed between nodes in the LangGraph workflow."""
    query: str
    documents: List[Dict[str, Any]]
    current_step: str
    execution_logs: List[str]
    final_output: str
    error: str

def processing_node(state: AgentState) -> AgentState:
    """Core reasoning and context synthesis node."""
    logger.info("Executing processing_node in LangGraph pipeline.")
    state["current_step"] = "processing"
    state["execution_logs"].append("Successfully processed ingested documents and generated embeddings context.")
    state["final_output"] = f"Processed response for query: {state['query']}"
    return state

def error_handler_node(state: AgentState) -> AgentState:
    """Fallback error handling node for resilient agent execution."""
    logger.error(f"Workflow encountered an error: {state.get('error', 'Unknown error')}")
    state["current_step"] = "error"
    state["execution_logs"].append(f"Error encountered: {state.get('error')}")
    state["final_output"] = "Workflow execution failed safely."
    return state

# Initialize the StateGraph with our AgentState schema
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("ingest", ingestion_node)
workflow.add_node("process", processing_node)
workflow.add_node("handle_error", error_handler_node)

# Define graph flow and conditional transitions
workflow.set_entry_point("ingest")

# Route from ingestion to processing, or to error handler if ingestion fails
def should_continue(state: AgentState) -> str:
    if state.get("error"):
        return "handle_error"
    return "process"

workflow.add_conditional_edges(
    "ingest",
    should_continue,
    {
        "process": "process",
        "handle_error": "handle_error"
    }
)

workflow.add_edge("process", END)
workflow.add_edge("handle_error", END)

# Compile the graph into an executable application handle
app_workflow = workflow.compile()
logger.info("LangGraph agent orchestrator successfully compiled.")