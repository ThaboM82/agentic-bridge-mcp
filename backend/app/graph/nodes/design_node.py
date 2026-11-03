"""
LangGraph design-analysis node.

Retrieves relevant design-system context, synthesizes UI component and
design-token specifications through an LLM, validates the output, and returns
bounded state updates for downstream workflow nodes.

Responsibilities:
- Retrieve design context from the vector store.
- Build a versioned, sanitized design-analysis prompt.
- Invoke the configured language-model provider.
- Validate structured model output.
- Record design tokens, citations, prompt version, and provider telemetry.
- Map provider and validation failures to recoverable workflow errors.

Non-goals:
- Do not generate source code, publish artifacts, enforce approval policy,
  or own retrieval-ranking strategy in this node.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser

from app.core.enums import WorkflowStep
from app.core.exceptions import (
    ExternalProviderError,
    ProviderTimeoutError,
    ValidationError,
    WorkflowStateError,
)
from app.core.logging_context import bind_workflow_context
from app.core.utils import truncate_text
from app.graph.prompts.design import (
    DESIGN_ANALYSIS_PROMPT_VERSION,
    DesignAnalysisOutputSchema,
    build_design_analysis_messages,
    estimate_prompt_tokens,
    get_design_prompt_metadata,
)
from app.graph.state import AgenticState
from app.graph.tools.vector_search_tools import retrieve_design_context
from app.services.llm_client import LLMProviderFactory


logger = logging.getLogger("agentic_bridge.graph.nodes.design")

MAX_RETRIEVED_DOCUMENTS = 5
MAX_CONTEXT_DOCUMENT_CHARS = 4_000
MAX_REQUIREMENTS = 50
MAX_REQUIREMENT_CHARS = 500
MAX_TOTAL_CONTEXT_CHARS = 30_000


# ---------------------------------------------------------------------------
# Prompt input helpers
# ---------------------------------------------------------------------------


def _build_requirements_text(
    requirements: list[str],
) -> str:
    """Create a bounded, readable requirements block for the prompt."""
    bounded_requirements = requirements[:MAX_REQUIREMENTS]

    return "\n".join(
        f"- {truncate_text(str(requirement), MAX_REQUIREMENT_CHARS)}"
        for requirement in bounded_requirements
    )


def _build_combined_context(
    context_matches: list[dict[str, Any]] | list[str],
) -> str:
    """
    Build a bounded combined context block from retrieval matches.

    Supports both:
    - Structured matches returned by `retrieve_design_context`
    - Plain string context matches for backward compatibility
    """
    if not context_matches:
        return "No internal design documentation matched."

    context_parts: list[str] = []
    total_characters = 0

    for index, match in enumerate(context_matches, start=1):
        if isinstance(match, str):
            content = match
            document_id = None
            similarity = None
        else:
            content = str(match.get("content") or "")
            document_id = match.get("document_id")
            similarity = match.get("similarity")

        if not content:
            continue

        bounded_content = truncate_text(
            content,
            MAX_CONTEXT_DOCUMENT_CHARS,
        )

        header = f"Context {index}"

        if document_id:
            header += f" | document_id={document_id}"

        if similarity is not None:
            header += f" | similarity={similarity:.3f}"

        context_block = f"{header}\n{bounded_content}"

        if total_characters + len(context_block) > MAX_TOTAL_CONTEXT_CHARS:
            break

        context_parts.append(context_block)
        total_characters += len(context_block)

    if not context_parts:
        return "No usable internal design documentation matched."

    return "\n---\n".join(context_parts)


# ---------------------------------------------------------------------------
# Retrieval helper
# ---------------------------------------------------------------------------


async def _retrieve_design_context(
    db: Any | None,
    requirements: list[str],
    *,
    tenant_id: str | None = None,
    document_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Retrieve design-system context for the supplied requirements.

    Retrieval failures are non-fatal: the workflow can still generate a design
    from the extracted requirements.

    Args:
        db: Optional async database session.
        requirements: Extracted workflow requirements.
        tenant_id: Optional tenant isolation filter.
        document_ids: Optional document restriction.

    Returns:
        A tuple of `(structured_context_matches, query_embedding)`.

        `query_embedding` is currently `None` until the embedding service is
        integrated. When available, generate an embedding for the requirements
        summary before calling `retrieve_design_context`.
    """
    if db is None:
        logger.info(
            "design_context_retrieval_skipped",
            reason="database_session_unavailable",
        )

        return [], None

    requirements_summary = _build_requirements_text(requirements)

    try:
        # TODO(service integration): Replace with:
        # query_embedding = await embedding_service.create_embedding(
        #     requirements_summary
        # )
        query_embedding = None

        if query_embedding is None:
            logger.info(
                "design_context_retrieval_skipped",
                reason="embedding_service_not_configured",
            )

            return [], None

        matches = await retrieve_design_context(
            db,
            query_embedding,
            tenant_id=tenant_id,
            document_ids=document_ids,
            limit=MAX_RETRIEVED_DOCUMENTS,
            similarity_threshold=0.0,
        )

        logger.info(
            "design_context_retrieved",
            match_count=len(matches),
            tenant_id=tenant_id,
        )

        return matches, query_embedding

    except Exception:
        logger.warning(
            "design_context_retrieval_failed",
            exc_info=True,
        )

        return [], None


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def design_analysis_node(
    state: AgenticState,
    db: Any | None = None,
) -> AgenticState:
    """
    Generate validated UI design tokens and component specifications.

    Args:
        state: Current LangGraph workflow state.
        db: Optional async database session used for vector retrieval.

    Returns:
        State updates containing design tokens, context matches, citations,
        prompt metadata, provider telemetry, logs, and error information.

    Raises:
        WorkflowStateError: If required workflow inputs are missing.
        ExternalProviderError: If the language-model provider fails.
    """
    execution_id = str(state.get("execution_id") or "unknown-execution")
    brief_id = str(state.get("brief_id") or "unknown-brief")
    extracted_requirements = state.get("extracted_requirements") or []
    tenant_id = state.get("tenant_id")
    document_ids = state.get("document_ids") or []

    bind_workflow_context(
        execution_id,
        workflow_step=WorkflowStep.DESIGN,
        tenant_id=tenant_id,
        user_id=state.get("user_id"),
    )

    logger.info(
        "design_analysis_started",
        brief_id=brief_id,
        requirement_count=len(extracted_requirements),
    )

    if not extracted_requirements:
        raise WorkflowStateError(
            "Design analysis cannot continue without extracted requirements."
        )

    context_matches, query_embedding = await _retrieve_design_context(
        db,
        extracted_requirements,
        tenant_id=tenant_id,
        document_ids=document_ids,
    )

    combined_context = _build_combined_context(context_matches)
    requirements_text = _build_requirements_text(extracted_requirements)

    try:
        llm = LLMProviderFactory.get_llm(temperature=0.2)

        parser = PydanticOutputParser(
            pydantic_object=DesignAnalysisOutputSchema
        )

        format_instructions = parser.get_format_instructions()

        messages = build_design_analysis_messages(
            requirements=requirements_text,
            design_context=combined_context,
            format_instructions=format_instructions,
        )

        estimated_prompt_tokens = estimate_prompt_tokens(
            requirements=requirements_text,
            design_context=combined_context,
            format_instructions=format_instructions,
        )

        logger.debug(
            "design_analysis_llm_dispatched",
            brief_id=brief_id,
            context_match_count=len(context_matches),
            estimated_prompt_tokens=estimated_prompt_tokens,
            prompt_version=DESIGN_ANALYSIS_PROMPT_VERSION,
        )

        response = await llm.ainvoke(messages)

        parsed_design = parser.parse(response.content)

        existing_design_tokens = dict(state.get("design_tokens") or {})

        existing_design_tokens.update(
            {
                "ui_components": parsed_design.ui_components_needed,
                "layout": parsed_design.layout_structure,
                "styling": parsed_design.styling_guidelines,
                "accessibility": parsed_design.accessibility_notes,
                "assumptions": parsed_design.assumptions,
                "confidence": parsed_design.confidence,
            }
        )

        citations = [
            {
                "document_id": match.get("document_id"),
                "embedding_id": match.get("embedding_id"),
                "title": match.get("title"),
                "chunk_index": match.get("chunk_index"),
                "similarity": match.get("similarity"),
            }
            for match in context_matches
            if isinstance(match, dict)
        ]

        logger.info(
            "design_analysis_completed",
            brief_id=brief_id,
            component_count=len(parsed_design.ui_components_needed),
            context_match_count=len(context_matches),
            confidence=parsed_design.confidence,
            prompt_version=DESIGN_ANALYSIS_PROMPT_VERSION,
        )

        return {
            "current_step": WorkflowStep.DESIGN,
            "design_tokens": existing_design_tokens,
            "vector_context_matches": context_matches,
            "citations": citations,
            "design_prompt_version": DESIGN_ANALYSIS_PROMPT_VERSION,
            "provider_name": state.get("provider_name"),
            "provider_model": state.get("provider_model"),
            "provider_token_usage": {
                "estimated_prompt_tokens": estimated_prompt_tokens,
            },
            "execution_logs": [
                (
                    "SUCCESS: Design analysis completed. Identified "
                    f"{len(parsed_design.ui_components_needed)} UI components "
                    f"using {len(context_matches)} context matches."
                )
            ],
            "current_error": None,
            "error_step": None,
            "error_code": None,
        }

    except (ProviderTimeoutError, ExternalProviderError):
        logger.exception(
            "design_analysis_provider_failure",
            brief_id=brief_id,
        )

        return {
            "current_step": WorkflowStep.DESIGN,
            "current_error": (
                "Design analysis failed because the language-model provider "
                "was unavailable or timed out."
            ),
            "error_step": WorkflowStep.DESIGN,
            "error_code": "provider_timeout",
            "recoverable_error": True,
            "execution_logs": [
                "ERROR: Design analysis failed due to a provider timeout."
            ],
        }

    except ValidationError as exc:
        logger.warning(
            "design_analysis_validation_failed",
            brief_id=brief_id,
            error=str(exc),
        )

        return {
            "current_step": WorkflowStep.DESIGN,
            "current_error": (
                "Design analysis produced invalid or incomplete output."
            ),
            "error_step": WorkflowStep.DESIGN,
            "error_code": "design_validation_error",
            "recoverable_error": True,
            "execution_logs": [
                "ERROR: Design analysis output failed validation."
            ],
        }

    except Exception:
        logger.exception(
            "design_analysis_unexpected_failure",
            brief_id=brief_id,
        )

        return {
            "current_step": WorkflowStep.DESIGN,
            "current_error": (
                "Design analysis failed due to an unexpected workflow error."
            ),
            "error_step": WorkflowStep.DESIGN,
            "error_code": "design_analysis_error",
            "recoverable_error": False,
            "execution_logs": [
                "ERROR: Design analysis failed unexpectedly."
            ],
        }