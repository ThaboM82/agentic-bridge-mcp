"""
Comprehensive tests for design-analysis prompt construction.

These tests cover prompt versioning, message structure, requirement and design
context inclusion, format instructions, whitespace validation, content
normalization, prompt determinism, and secret-leak prevention.
"""

from __future__ import annotations

import asyncio

import pytest

from app.graph.prompts.design import (
    DESIGN_ANALYSIS_PROMPT_VERSION,
    build_design_analysis_messages,
    build_design_analysis_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def requirements() -> str:
    return "Build a dashboard."


@pytest.fixture
def design_context() -> str:
    return (
        "Use the design system card component."
    )


@pytest.fixture
def format_instructions() -> str:
    return "Return JSON."


# ---------------------------------------------------------------------------
# Prompt version
# ---------------------------------------------------------------------------


def test_prompt_version_is_set():
    assert (
        DESIGN_ANALYSIS_PROMPT_VERSION
        == "v1"
    )


def test_prompt_version_is_stable():
    assert isinstance(
        DESIGN_ANALYSIS_PROMPT_VERSION,
        str,
    )
    assert (
        DESIGN_ANALYSIS_PROMPT_VERSION
        != ""
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_build_design_analysis_prompt():
    prompt = build_design_analysis_prompt(
        requirements="Build a dashboard.",
        design_context=(
            "Use the design system card component."
        ),
        format_instructions="Return JSON.",
    )

    assert prompt is not None


def test_build_design_analysis_prompt_returns_string(
    requirements: str,
    design_context: str,
    format_instructions: str,
):
    prompt = build_design_analysis_prompt(
        requirements=requirements,
        design_context=design_context,
        format_instructions=format_instructions,
    )

    assert isinstance(prompt, str)
    assert prompt


def test_prompt_includes_requirements(
    requirements: str,
    design_context: str,
    format_instructions: str,
):
    prompt = build_design_analysis_prompt(
        requirements=requirements,
        design_context=design_context,
        format_instructions=format_instructions,
    )

    assert requirements in prompt


def test_prompt_includes_design_context(
    requirements: str,
    design_context: str,
    format_instructions: str,
):
    prompt = build_design_analysis_prompt(
        requirements=requirements,
        design_context=design_context,
        format_instructions=format_instructions,
    )

    assert design_context in prompt


def test_prompt_includes_format_instructions(
    requirements: str,
    design_context: str,
    format_instructions: str,
):
    prompt = build_design_analysis_prompt(
        requirements=requirements,
        design_context=design_context,
        format_instructions=format_instructions,
    )

    assert format_instructions in prompt


def test_prompt_is_deterministic(
    requirements: str,
    design_context: str,
    format_instructions: str,
):
    first_prompt = (
        build_design_analysis_prompt(
            requirements=requirements,
            design_context=design_context,
            format_instructions=format_instructions,
        )
    )
    second_prompt = (
        build_design_analysis_prompt(
            requirements=requirements,
            design_context=design_context,
            format_instructions=format_instructions,
        )
    )

    assert first_prompt == second_prompt


def test_prompt_normalizes_whitespace(
    requirements: str,
    design_context: str,
    format_instructions: str,
):
    prompt = build_design_analysis_prompt(
        requirements=(
            f"  {requirements}  "
        ),
        design_context=(
            f"  {design_context}  "
        ),
        format_instructions=(
            f"  {format_instructions}  "
        ),
    )

    assert requirements in prompt
    assert design_context in prompt
    assert format_instructions in prompt


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


def test_build_design_analysis_messages():
    messages = (
        build_design_analysis_messages(
            requirements="Build a dashboard.",
            design_context=(
                "Use accessible components."
            ),
            format_instructions=(
                "Return JSON."
            ),
        )
    )

    assert len(messages) == 2
    assert (
        "Build a dashboard."
        in messages[1].content
    )
    assert (
        "Use accessible components."
        in messages[1].content
    )


def test_messages_have_system_and_user_roles(
    requirements: str,
    design_context: str,
    format_instructions: str,
):
    messages = (
        build_design_analysis_messages(
            requirements=requirements,
            design_context=design_context,
            format_instructions=format_instructions,
        )
    )

    assert messages[0].role == "system"
    assert messages[1].role == "user"


def test_system_message_contains_design_instructions(
    requirements: str,
    design_context: str,
    format_instructions: str,
):
    messages = (
        build_design_analysis_messages(
            requirements=requirements,
            design_context=design_context,
            format_instructions=format_instructions,
        )
    )

    assert messages[0].content
    assert (
        "design" in messages[
            0
        ].content.lower()
    )


def test_user_message_contains_all_inputs(
    requirements: str,
    design_context: str,
    format_instructions: str,
):
    messages = (
        build_design_analysis_messages(
            requirements=requirements,
            design_context=design_context,
            format_instructions=format_instructions,
        )
    )

    user_content = messages[1].content

    assert requirements in user_content
    assert design_context in user_content
    assert (
        format_instructions
        in user_content
    )


def test_messages_are_deterministic(
    requirements: str,
    design_context: str,
    format_instructions: str,
):
    first_messages = (
        build_design_analysis_messages(
            requirements=requirements,
            design_context=design_context,
            format_instructions=format_instructions,
        )
    )
    second_messages = (
        build_design_analysis_messages(
            requirements=requirements,
            design_context=design_context,
            format_instructions=format_instructions,
        )
    )

    assert [
        message.content
        for message in first_messages
    ] == [
        message.content
        for message in second_messages
    ]


def test_messages_normalize_whitespace(
    requirements: str,
    design_context: str,
    format_instructions: str,
):
    messages = (
        build_design_analysis_messages(
            requirements=(
                f"  {requirements}  "
            ),
            design_context=(
                f"  {design_context}  "
            ),
            format_instructions=(
                f"  {format_instructions}  "
            ),
        )
    )

    user_content = messages[1].content

    assert requirements in user_content
    assert design_context in user_content
    assert (
        format_instructions
        in user_content
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_missing_requirements_are_rejected():
    with pytest.raises(ValueError):
        build_design_analysis_prompt(
            requirements="   ",
            design_context="Context",
            format_instructions=(
                "Return JSON."
            ),
        )


def test_missing_format_instructions_are_rejected():
    with pytest.raises(ValueError):
        build_design_analysis_prompt(
            requirements=(
                "Build a dashboard."
            ),
            design_context="Context",
            format_instructions="   ",
        )


@pytest.mark.parametrize(
    "invalid_requirements",
    [
        "",
        "   ",
        "\n",
        "\t",
        None,
    ],
)
def test_prompt_rejects_invalid_requirements(
    invalid_requirements,
):
    with pytest.raises(ValueError):
        build_design_analysis_prompt(
            requirements=invalid_requirements,
            design_context="Context",
            format_instructions=(
                "Return JSON."
            ),
        )


@pytest.mark.parametrize(
    "invalid_format_instructions",
    [
        "",
        "   ",
        "\n",
        "\t",
        None,
    ],
)
def test_prompt_rejects_invalid_format_instructions(
    invalid_format_instructions,
):
    with pytest.raises(ValueError):
        build_design_analysis_prompt(
            requirements=(
                "Build a dashboard."
            ),
            design_context="Context",
            format_instructions=invalid_format_instructions,
        )


@pytest.mark.parametrize(
    "invalid_requirements",
    [
        "",
        "   ",
        None,
    ],
)
def test_messages_reject_invalid_requirements(
    invalid_requirements,
):
    with pytest.raises(ValueError):
        build_design_analysis_messages(
            requirements=invalid_requirements,
            design_context="Context",
            format_instructions=(
                "Return JSON."
            ),
        )


@pytest.mark.parametrize(
    "invalid_format_instructions",
    [
        "",
        "   ",
        None,
    ],
)
def test_messages_reject_invalid_format_instructions(
    invalid_format_instructions,
):
    with pytest.raises(ValueError):
        build_design_analysis_messages(
            requirements=(
                "Build a dashboard."
            ),
            design_context="Context",
            format_instructions=invalid_format_instructions,
        )


# ---------------------------------------------------------------------------
# Optional design context
# ---------------------------------------------------------------------------


def test_prompt_allows_empty_design_context(
    requirements: str,
    format_instructions: str,
):
    prompt = build_design_analysis_prompt(
        requirements=requirements,
        design_context="",
        format_instructions=format_instructions,
    )

    assert requirements in prompt
    assert (
        format_instructions in prompt
    )


def test_prompt_allows_none_design_context(
    requirements: str,
    format_instructions: str,
):
    prompt = build_design_analysis_prompt(
        requirements=requirements,
        design_context=None,
        format_instructions=format_instructions,
    )

    assert requirements in prompt
    assert (
        format_instructions in prompt
    )


def test_messages_allow_empty_design_context(
    requirements: str,
    format_instructions: str,
):
    messages = (
        build_design_analysis_messages(
            requirements=requirements,
            design_context="",
            format_instructions=format_instructions,
        )
    )

    assert len(messages) == 2
    assert requirements in (
        messages[1].content
    )


def test_messages_allow_none_design_context(
    requirements: str,
    format_instructions: str,
):
    messages = (
        build_design_analysis_messages(
            requirements=requirements,
            design_context=None,
            format_instructions=format_instructions,
        )
    )

    assert len(messages) == 2
    assert requirements in (
        messages[1].content
    )


# ---------------------------------------------------------------------------
# Security and prompt hygiene
# ---------------------------------------------------------------------------


def test_prompt_does_not_leak_secrets(
    requirements: str,
    design_context: str,
    format_instructions: str,
):
    prompt = build_design_analysis_prompt(
        requirements=requirements,
        design_context=design_context,
        format_instructions=format_instructions,
    )

    serialized_prompt = prompt.lower()

    for forbidden_fragment in (
        "api_key",
        "apikey",
        "password",
        "secret",
        "token",
        "authorization",
        "postgresql://",
        "redis://",
    ):
        assert (
            forbidden_fragment
            not in serialized_prompt
        )


def test_messages_do_not_leak_secrets(
    requirements: str,
    design_context: str,
    format_instructions: str,
):
    messages = (
        build_design_analysis_messages(
            requirements=requirements,
            design_context=design_context,
            format_instructions=format_instructions,
        )
    )

    serialized_messages = str(
        messages
    ).lower()

    for forbidden_fragment in (
        "api_key",
        "apikey",
        "password",
        "secret",
        "token",
        "authorization",
        "postgresql://",
        "redis://",
    ):
        assert (
            forbidden_fragment
            not in serialized_messages
        )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    asyncio.run(
        pytest.main(
            [
                __file__,
                "-v",
            ]
        )
    )
