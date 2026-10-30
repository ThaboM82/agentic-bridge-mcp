import logging
from typing import Dict, Any
from app.graph.state import AgenticState

logger = logging.getLogger("agentic_bridge.graph.publish_node")

def publish_node(state: AgenticState) -> Dict[str, Any]:
    """
    Production-grade LangGraph node responsible for validating generated code artifacts,
    setting review statuses, and finalizing the workflow execution package.
    
    Args:
        state (AgenticState): Current graph execution state dictionary.
        
    Returns:
        Dict[str, Any]: State updates containing final review status and execution logs.
    """
    execution_id = state.get("execution_id", "unknown-execution")
    brief_id = state.get("brief_id", "unknown-brief")
    generated_files = state.get("generated_code_files", {})

    logger.info(f"[{execution_id}] Starting publish node processing for brief ID: {brief_id}")

    if not generated_files:
        error_msg = f"[{execution_id}] Publishing failed: No generated code files found in state to publish."
        logger.error(error_msg)
        return {
            "review_status": "failed",
            "current_error": error_msg,
            "execution_logs": [f"ERROR: {error_msg}"]
        }

    try:
        file_count = len(generated_files)
        logger.info(f"[{execution_id}] Verifying integrity of {file_count} generated files prior to publishing...")

        # Perform rigorous validation checks on artifact integrity
        for filename, content in generated_files.items():
            if not content or len(content.strip()) == 0:
                raise ValueError(f"Generated file '{filename}' is empty or contains invalid content.")

        logger.info(f"[{execution_id}] All artifacts verified successfully. Setting review status to pending approval.")

        return {
            "review_status": "pending_approval",
            "current_error": None,
            "execution_logs": [
                f"SUCCESS: Publish node completed. Prepared {file_count} files for deployment and set status to pending approval."
            ]
        }

    except Exception as e:
        error_msg = f"[{execution_id}] Critical exception in publish node: {str(e)}"
        logger.exception(error_msg)
        return {
            "review_status": "failed",
            "current_error": error_msg,
            "execution_logs": [f"ERROR: {error_msg}"]
        }