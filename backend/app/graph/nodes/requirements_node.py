import logging
import time
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.graph.state import AgenticState
from app.services.llm_client import LLMProviderFactory

logger = logging.getLogger("agentic_bridge.graph.processing_node")

class ArchitecturalSpecificationSchema(BaseModel):
    """
    Pydantic schema enforcing strict enterprise structure for generated 
    system architecture specifications and deployment topologies.
    """
    system_overview: str = Field(description="High-level architectural overview and design patterns selected.")
    microservices: List[Dict[str, str]] = Field(description="List of required microservices with their primary responsibilities and communication protocols.")
    database_entities: List[str] = Field(description="Key database tables, models, and vector storage requirements (e.g., PostgreSQL with pgvector).")
    api_endpoints: List[str] = Field(description="Core FastAPI REST and MCP endpoints required to support the features.")
    security_and_compliance: List[str] = Field(description="Security hardening, authentication, and compliance measures.")
    deployment_recommendations: List[str] = Field(description="Containerization, orchestration (Docker Compose/K8s), and cloud target recommendations.")

def processing_node(state: AgenticState) -> Dict[str, Any]:
    """
    Production-grade LangGraph node responsible for synthesizing extracted requirements 
    into a complete enterprise architectural specification with telemetry and retry hardening.
    
    Args:
        state (AgenticState): Current graph execution state dictionary.
        
    Returns:
        Dict[str, Any]: State updates containing architectural specifications, telemetry, and execution logs.
    """
    execution_id = state.get("execution_id", "unknown-execution")
    brief_id = state.get("brief_id", "unknown-brief")
    extracted_requirements = state.get("extracted_requirements", [])
    design_tokens = state.get("design_tokens", {})

    logger.info(f"[{execution_id}] Starting architectural processing node for brief ID: {brief_id}")

    if not extracted_requirements:
        error_msg = f"[{execution_id}] Processing failed: No extracted requirements found in graph state."
        logger.error(error_msg)
        return {
            "current_error": error_msg,
            "execution_logs": [f"ERROR: {error_msg}"]
        }

    start_time = time.time()
    max_retries = 2
    attempt = 0
    parsed_spec = None
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    try:
        while attempt <= max_retries:
            try:
                attempt += 1
                # Instantiate LLM via multi-provider factory
                llm = LLMProviderFactory.get_llm(temperature=0.2)
                
                # Setup Pydantic output parser
                parser = PydanticOutputParser(pydantic_object=ArchitecturalSpecificationSchema)
                
                # Construct prompt template for systems architecture synthesis
                prompt_template = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        "You are a Principal Enterprise Systems Architect and Chief Technology Officer. "
                        "Design a robust, containerized, production-ready system architecture based on the provided "
                        "extracted requirements and design tokens. Define microservices, data models, API endpoints, "
                        "security measures, and deployment targets.\n{format_instructions}"
                    ),
                    (
                        "human",
                        "Extracted Requirements:\n{requirements}\n\nDesign Tokens / Constraints:\n{tokens}"
                    )
                ])
                
                prompt = prompt_template.format_messages(
                    format_instructions=parser.get_format_instructions(),
                    requirements="\n".join([f"- {req}" for req in extracted_requirements]),
                    tokens=str(design_tokens)
                )

                logger.debug(f"[{execution_id}] Dispatching architecture synthesis prompt (Attempt {attempt}/{max_retries + 1})...")
                
                response = llm.invoke(prompt)
                
                # Extract token usage telemetry if available in response metadata
                response_metadata = getattr(response, "response_metadata", {})
                usage = response_metadata.get("token_usage", {})
                token_usage = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                }

                parsed_spec = parser.parse(response.content)
                break  # Success, exit retry loop

            except Exception as inner_e:
                logger.warning(f"[{execution_id}] Processing attempt {attempt} failed: {str(inner_e)}")
                if attempt > max_retries:
                    raise inner_e
                time.sleep(1.5 * attempt)  # Exponential backoff

        duration = round(time.time() - start_time, 2)
        logger.info(f"[{execution_id}] Successfully synthesized architectural spec in {duration}s using {token_usage['total_tokens']} tokens.")

        return {
            "architectural_specification": {
                "system_overview": parsed_spec.system_overview,
                "microservices": parsed_spec.microservices,
                "database_entities": parsed_spec.database_entities,
                "api_endpoints": parsed_spec.api_endpoints,
                "security_and_compliance": parsed_spec.security_and_compliance,
                "deployment_recommendations": parsed_spec.deployment_recommendations
            },
            "design_tokens": {
                **design_tokens,
                "processing_duration_seconds": duration,
                "token_usage": token_usage
            },
            "current_error": None,
            "execution_logs": [
                f"SUCCESS: Processing node generated architectural specification in {duration}s ({token_usage['total_tokens']} tokens)."
            ]
        }

    except Exception as e:
        error_msg = f"[{execution_id}] Critical exception in processing node: {str(e)}"
        logger.exception(error_msg)
        return {
            "current_error": error_msg,
            "execution_logs": [f"ERROR: {error_msg}"]
        }