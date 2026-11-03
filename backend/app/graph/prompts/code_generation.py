"""
Versioned prompts for multi-file code generation.

Prompt changes should increment the prompt version so generated artifacts and
audit records remain traceable to the exact instruction set used.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


CODE_GENERATION_PROMPT_VERSION = "v1"

CODE_GENERATION_SYSTEM_PROMPT = """You are a Principal Full-Stack Software Engineer and Systems Architect.

Generate complete, production-quality source code that satisfies the supplied requirements and design context.

Rules:
- Produce complete files with no placeholders, TODOs, ellipses, or omitted logic.
- Use safe relative file paths only.
- Do not include credentials, API keys, secrets, or environment-specific values.
- Do not include destructive commands or unsafe dynamic code execution.
- Follow the supplied design system, component structure, accessibility guidance, and architecture.
- Return only files required to implement the requested functionality.

{format_instructions}"""

CODE_GENERATION_HUMAN_PROMPT = """Project requirements:
{requirements}

Design tokens and architecture context:
{design_context}"""


def build_code_generation_messages(
    *,
    requirements: str,
    design_context: str,
    format_instructions: str,
):
    """Build chat messages for the code-generation node."""
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", CODE_GENERATION_SYSTEM_PROMPT),
            ("human", CODE_GENERATION_HUMAN_PROMPT),
        ]
    )

    return prompt_template.format_messages(
        format_instructions=format_instructions,
        requirements=requirements,
        design_context=design_context,
    )