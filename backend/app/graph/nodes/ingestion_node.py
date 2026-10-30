import logging
import time
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.graph.state import AgenticState
from app.services.llm_client import LLMProviderFactory

logger = logging.getLogger("agentic_bridge.graph.ingestion_node")

class ExtractedRequirementsSchema(BaseModel):
    """
    Pydantic schema enforcing strict structure for requirements 
    extracted from unstructured product briefs.
    """
    summary: str = Field(description="A concise executive summary of the project brief.")
    core_features: List[str] = Field(description="List of primary functional requirements and user stories.")
    technical_constraints: List[str] = Field(description="Identified tech stack requirements, performance metrics, or security constraints.")
    risk_assessment: str = Field(description="Potential architectural or implementation risks identified in the brief.")

def ingestion_node(state: AgenticState) -> Dict[str, Any]:
    """
    Production-grade LangGraph node responsible for ingesting, parsing, 
    and structuring raw product briefs using enterprise LLMs with retry hardening and telemetry.
    
    Args:
        state (AgenticState): Current graph execution state dictionary.
        
    Returns:
        Dict[str, Any]: State updates containing extracted requirements, design tokens, and execution logs.
    """
    execution_id = state.get("execution_id", "unknown-execution")
    brief_id = state.get("brief_id", "unknown-brief")
    raw_brief_text = state.get("raw_brief_text", "")

    logger.info(f"[{execution_id}] Starting ingestion node processing for brief ID: {brief_id}")

    if not raw_brief_text or not raw_brief_text.strip():
        error_msg = f"[{execution_id}] Ingestion failed: Raw brief text is empty or missing for brief ID {brief_id}."
        logger.error(error_msg)
        return {
            "current_error": error_msg,
            "execution_logs": [f"ERROR: {error_msg}"]
        }

    start_time = time.time()
    max_retries = 2
    attempt = 0
    parsed_output = None

    try:
        while attempt <= max_retries:
            try:
                attempt += 1
                # Instantiate LLM via our multi-provider factory pattern
                llm = LLMProviderFactory.get_llm(temperature=0.1)
                
                # Setup Pydantic output parser for reliable JSON schema validation
                parser = PydanticOutputParser(pydantic_object=ExtractedRequirementsSchema)
                
                # Construct enterprise prompt template
                prompt_template = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        "You are a Principal Enterprise Systems Architect and Senior Technical Product Manager. "
                        "Analyze the provided product brief and extract structured requirements, core features, "
                        "technical constraints, and architectural risks.\n{format_instructions}"
                    ),
                    (
                        "human",
                        "Here is the product brief to analyze:\n\n{raw_brief}"
                    )
                ])
                
                # Format prompt with parser instructions
                prompt = prompt_template.format_messages(
                    format_instructions=parser.get_format_instructions(),
                    raw_brief=raw_brief_text
                )

                logger.debug(f"[{execution_id}] Dispatching prompt to LLM provider (Attempt {attempt}/{max_retries + 1})...")
                
                # Execute model call
                response = llm.invoke(prompt)
                
                # Parse and validate structured output against Pydantic schema
                parsed_output = parser.parse(response.content)
                break  # Success, exit retry loop

            except Exception as inner_e:
                logger.warning(f"[{execution_id}] Attempt {attempt} failed: {str(inner_e)}")
                if attempt > max_retries:
                    raise inner_e
                time.sleep(1.5 * attempt)  # Exponential backoff delay

        duration = round(time.time() - start_time, 2)
        logger.info(f"[{execution_id}] Successfully extracted {len(parsed_output.core_features)} core requirements in {duration}s.")

        # Compile comprehensive state update
        return {
            "extracted_requirements": parsed_output.core_features,
            "design_tokens": {
                "summary": parsed_output.summary,
                "technical_constraints": parsed_output.technical_constraints,
                "risk_assessment": parsed_output.risk_assessment,
                "ingestion_duration_seconds": duration
            },
            "current_error": None,
            "execution_logs": [
                f"SUCCESS: Ingestion node completed for brief {brief_id} in {duration}s. Extracted {len(parsed_output.core_features)} requirements."
            ]
        }

    except Exception as e:
        error_msg = f"[{execution_id}] Critical exception in ingestion node: {str(e)}"
        logger.exception(error_msg)
        return {
            "current_error": error_msg,
            "execution_logs": [f"ERROR: {error_msg}"]
        }