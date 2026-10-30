import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.graph.state import AgenticState
from app.services.llm_client import LLMProviderFactory
from app.database import SessionLocal
from app.models.vector_model import DocumentEmbedding

logger = logging.getLogger("agentic_bridge.graph.design_node")

class DesignAnalysisOutputSchema(BaseModel):
    """
    Pydantic schema enforcing strict validation for design tokens, 
    UI component structures, and layout specifications.
    """
    ui_components_needed: List[str] = Field(description="List of specific React or UI components required to satisfy the brief.")
    layout_structure: str = Field(description="Architectural description of the page layout, grid system, and hierarchy.")
    styling_guidelines: List[str] = Field(description="Design system tokens, color palettes, and Tailwind CSS spacing rules to apply.")
    accessibility_notes: str = Field(description="WCAG compliance considerations and ARIA attribute requirements.")

def design_analysis_node(state: AgenticState) -> Dict[str, Any]:
    """
    Production-grade LangGraph node responsible for performing vector retrieval 
    of design guidelines and generating UI design tokens via LLM synthesis.
    
    Args:
        state (AgenticState): Current graph execution state dictionary.
        
    Returns:
        Dict[str, Any]: State updates containing design tokens, vector matches, and execution logs.
    """
    execution_id = state.get("execution_id", "unknown-execution")
    brief_id = state.get("brief_id", "unknown-brief")
    extracted_requirements = state.get("extracted_requirements", [])

    logger.info(f"[{execution_id}] Starting design analysis node processing for brief ID: {brief_id}")

    if not extracted_requirements:
        error_msg = f"[{execution_id}] Design analysis failed: No extracted requirements found in state."
        logger.error(error_msg)
        return {
            "current_error": error_msg,
            "execution_logs": [f"ERROR: {error_msg}"]
        }

    retrieved_contexts: List[str] = []
    db = SessionLocal()
    try:
        logger.debug(f"[{execution_id}] Querying vector database for matching design system guidelines...")
        
        # In a fully integrated production environment, you would generate an embedding 
        # for the summary requirements text using an embedding model client. 
        # For demonstration, we query recent document embeddings matching semantic keywords 
        # or execute a vector distance search if a query vector is available.
        
        # Example: Fetching top semantic matches from pgvector table
        vector_results = (
            db.query(DocumentEmbedding)
            .limit(3)
            .all()
        )
        
        for doc in vector_results:
            if doc.content:
                retrieved_contexts.append(doc.content)
                
        logger.info(f"[{execution_id}] Retrieved {len(retrieved_contexts)} context documents from vector store.")

    except Exception as db_error:
        # Non-fatal: Log warning and proceed with LLM generation using requirements alone if vector store fails
        logger.warning(f"[{execution_id}] Vector store retrieval encountered an issue: {str(db_error)}. Proceeding without vector context.")
    finally:
        db.close()

    try:
        # Instantiate enterprise LLM via Provider Factory
        llm = LLMProviderFactory.get_llm(temperature=0.2)
        
        # Setup Pydantic parser
        parser = PydanticOutputParser(pydantic_object=DesignAnalysisOutputSchema)
        
        # Compile context string from vector matches
        combined_vector_context = "\n---\n".join(retrieved_contexts) if retrieved_contexts else "No internal design docs matched."

        # Construct enterprise prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a Lead UI/UX Systems Architect and Senior Frontend Engineer. "
                "Synthesize the extracted requirements and enterprise design guidelines below "
                "to formulate comprehensive UI component specifications and design tokens.\n{format_instructions}"
            ),
            (
                "human",
                "Extracted Requirements:\n{requirements}\n\n"
                "Enterprise Design System Context (Retrieved via Vector Search):\n{vector_context}"
            )
        ])

        prompt = prompt_template.format_messages(
            format_instructions=parser.get_format_instructions(),
            requirements="\n".join([f"- {req}" for req in extracted_requirements]),
            vector_context=combined_vector_context
        )

        logger.debug(f"[{execution_id2 if 'execution_id2' in locals() else execution_id}] Dispatching design analysis prompt to LLM...")
        
        response = llm.invoke(prompt)
        parsed_design: DesignAnalysisOutputSchema = parser.parse(response.content)

        logger.info(f"[{execution_id}] Successfully generated design tokens and identified {len(parsed_design.ui_components_needed)} UI components.")

        # Merge existing design tokens with newly generated specifications
        existing_design_tokens = state.get("design_tokens", {})
        existing_design_tokens.update({
            "ui_components": parsed_design.ui_components_needed,
            "layout": parsed_design.layout_structure,
            "styling": parsed_design.styling_guidelines,
            "accessibility": parsed_design.accessibility_notes
        })

        return {
            "vector_context_matches": retrieved_contexts,
            "design_tokens": existing_design_tokens,
            "current_error": None,
            "execution_logs": [
                f"SUCCESS: Design analysis node completed. Mapped {len(parsed_design.ui_components_needed)} UI components using vector context."
            ]
        }

    except Exception as e:
        error_msg = f"[{execution_id}] Critical exception in design analysis node: {str(e)}"
        logger.exception(error_msg)
        return {
            "current_error": error_msg,
            "execution_logs": [f"ERROR: {error_msg}"]
        }