import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.graph.state import AgenticState
from app.services.llm_client import LLMProviderFactory

logger = logging.getLogger("agentic_bridge.graph.code_gen_node")

class CodeFileModel(BaseModel):
    filename: str = Field(description="Relative path and filename, e.g., 'frontend/src/components/Dashboard.jsx' or 'backend/app/routers/feature.py'.")
    content: str = Field(description="Complete, production-ready source code content with zero placeholders or TODO comments.")

class MultiFileCodeGenerationSchema(BaseModel):
    files: List[CodeFileModel] = Field(description="List of fully implemented code files generated based on requirements and design tokens.")

def code_generation_node(state: AgenticState) -> Dict[str, Any]:
    """
    Production-grade LangGraph node responsible for multi-file code generation
    (React frontends and FastAPI backends) based on structured requirements and design tokens.
    
    Args:
        state (AgenticState): Current graph execution state dictionary.
        
    Returns:
        Dict[str, Any]: State updates containing generated code files and execution logs.
    """
    execution_id = state.get("execution_id", "unknown-execution")
    brief_id = state.get("brief_id", "unknown-brief")
    extracted_requirements = state.get("extracted_requirements", [])
    design_tokens = state.get("design_tokens", {})

    logger.info(f"[{execution_id}] Starting code generation node processing for brief ID: {brief_id}")

    if not extracted_requirements:
        error_msg = f"[{execution_id}] Code generation failed: No extracted requirements available in state."
        logger.error(error_msg)
        return {
            "current_error": error_msg,
            "execution_logs": [f"ERROR: {error_msg}"]
        }

    try:
        # Instantiate enterprise LLM via Provider Factory
        llm = LLMProviderFactory.get_llm(temperature=0.2)
        
        # Setup Pydantic parser for structured multi-file output
        parser = PydanticOutputParser(pydantic_object=MultiFileCodeGenerationSchema)

        # Construct enterprise prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a Principal Full-Stack Software Engineer and Systems Architect. "
                "Generate complete, production-grade, enterprise-standard source code files "
                "(React/Tailwind components and FastAPI backend endpoints) fulfilling the exact requirements "
                "and design tokens provided. Never use placeholders, 'TODO' comments, or shortened logic. "
                "Every file must be fully written out.\n{format_instructions}"
            ),
            (
                "human",
                "Project Requirements:\n{requirements}\n\n"
                "Design Tokens & Architecture Specifications:\n{design_tokens}"
            )
        ])

        prompt = prompt_template.format_messages(
            format_instructions=parser.get_format_instructions(),
            requirements="\n".join([f"- {req}" for req in extracted_requirements]),
            design_tokens=str(design_tokens)
        )

        logger.debug(f"[{execution_id}] Dispatching multi-file code generation prompt to LLM...")
        
        response = llm.invoke(prompt)
        parsed_code: MultiFileCodeGenerationSchema = parser.parse(response.content)

        # Map generated files into a dictionary format
        generated_files_map: Dict[str, str] = {}
        for file_item in parsed_code.files:
            generated_files_map[file_item.filename] = file_item.content

        logger.info(f"[{execution_id}] Successfully generated {len(generated_files_map)} source code files.")

        return {
            "generated_code_files": generated_files_map,
            "current_error": None,
            "execution_logs": [
                f"SUCCESS: Code generation node completed. Generated {len(generated_files_map)} production files."
            ]
        }

    except Exception as e:
        error_msg = f"[{execution_id}] Critical exception in code generation node: {str(e)}"
        logger.exception(error_msg)
        return {
            "current_error": error_msg,
            "execution_logs": [f"ERROR: {error_msg}"]
        }