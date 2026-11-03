"""
Comprehensive tests for the code-generation workflow node.

These tests cover generated-file path safety, content validation, duplicate
detection, schema constraints, workflow prerequisites, state transitions,
generated-file persistence, and immutable state behavior.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.core.enums import WorkflowStep
from app.core.exceptions import WorkflowStateError
from app.graph.nodes.code_generation_node import (
    GeneratedCodeFile,
    MultiFileCodeGenerationSchema,
    code_generation_node,
)
from app.graph.state import create_initial_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state():
    return create_initial_state(
        execution_id="wf_123",
        brief_id="brief_456",
        raw_brief_text="Build a dashboard.",
    )


@pytest.fixture
def state_with_requirements(state):
    return {
        **state,
        "extracted_requirements": [
            "Create a dashboard",
            "Show key metrics",
        ],
    }


@pytest.fixture
def state_with_design_context(
    state_with_requirements,
):
    return {
        **state_with_requirements,
        "design_tokens": {
            "accessibility": (
                "Use semantic HTML and ARIA labels."
            ),
        },
        "accessibility_requirements": [
            "Keyboard navigation must be supported."
        ],
    }


# ---------------------------------------------------------------------------
# Generated file validation
# ---------------------------------------------------------------------------


def test_generated_file_rejects_absolute_path():
    with pytest.raises(ValueError):
        GeneratedCodeFile(
            filename="/etc/passwd",
            content="print('hello')",
        )


def test_generated_file_rejects_parent_directory():
    with pytest.raises(ValueError):
        GeneratedCodeFile(
            filename="../secret.py",
            content="print('hello')",
        )


def test_generated_file_rejects_empty_content():
    with pytest.raises(ValueError):
        GeneratedCodeFile(
            filename="app/main.py",
            content="   ",
        )


@pytest.mark.parametrize(
    "unsafe_filename",
    [
        "/etc/passwd",
        "/absolute/path.py",
        "../secret.py",
        "../../secret.py",
        "app/../secret.py",
        "app/../../secret.py",
        "C:\\Windows\\system32\\config",
        "app\\..\\secret.py",
    ],
)
def test_generated_file_rejects_unsafe_paths(
    unsafe_filename: str,
):
    with pytest.raises(ValueError):
        GeneratedCodeFile(
            filename=unsafe_filename,
            content="print('hello')",
        )


@pytest.mark.parametrize(
    "safe_filename",
    [
        "app/main.py",
        "src/components/Dashboard.tsx",
        "docs/README.md",
        "tests/test_app.py",
        "config/settings.json",
    ],
)
def test_generated_file_accepts_safe_paths(
    safe_filename: str,
):
    generated_file = GeneratedCodeFile(
        filename=safe_filename,
        content="print('hello')",
    )

    assert (
        generated_file.filename
        == safe_filename
    )
    assert (
        generated_file.content
        == "print('hello')"
    )


@pytest.mark.parametrize(
    "invalid_content",
    [
        "",
        "   ",
        "\n",
        "\t",
        None,
    ],
)
def test_generated_file_rejects_invalid_content(
    invalid_content,
):
    with pytest.raises(ValueError):
        GeneratedCodeFile(
            filename="app/main.py",
            content=invalid_content,
        )


def test_generated_file_preserves_content():
    content = "print('hello world')"

    generated_file = GeneratedCodeFile(
        filename="app/main.py",
        content=content,
    )

    assert (
        generated_file.content
        == content
    )


def test_generated_file_strips_whitespace():
    generated_file = GeneratedCodeFile(
        filename="app/main.py",
        content="  print('hello')  ",
    )

    assert (
        generated_file.content
        == "print('hello')"
    )


# ---------------------------------------------------------------------------
# Multi-file schema validation
# ---------------------------------------------------------------------------


def test_schema_rejects_duplicate_files():
    file = GeneratedCodeFile(
        filename="app/main.py",
        content="print('hello')",
    )

    with pytest.raises(ValueError):
        MultiFileCodeGenerationSchema(
            files=[file, file]
        )


def test_schema_accepts_unique_files():
    schema = MultiFileCodeGenerationSchema(
        files=[
            GeneratedCodeFile(
                filename="app/main.py",
                content="print('hello')",
            ),
            GeneratedCodeFile(
                filename="app/utils.py",
                content="value = 42",
            ),
        ]
    )

    assert len(schema.files) == 2


def test_schema_rejects_empty_file_list():
    with pytest.raises(ValueError):
        MultiFileCodeGenerationSchema(
            files=[]
        )


def test_schema_rejects_none_file_list():
    with pytest.raises(ValueError):
        MultiFileCodeGenerationSchema(
            files=None
        )


def test_schema_rejects_too_many_files():
    files = [
        GeneratedCodeFile(
            filename=f"app/file_{index}.py",
            content="print('hello')",
        )
        for index in range(51)
    ]

    with pytest.raises(ValueError):
        MultiFileCodeGenerationSchema(
            files=files
        )


def test_schema_enforces_file_count_boundary():
    files = [
        GeneratedCodeFile(
            filename=f"app/file_{index}.py",
            content="print('hello')",
        )
        for index in range(50)
    ]

    schema = MultiFileCodeGenerationSchema(
        files=files
    )

    assert len(schema.files) == 50


def test_schema_rejects_duplicate_paths_with_different_content():
    first_file = GeneratedCodeFile(
        filename="app/main.py",
        content="print('first')",
    )
    second_file = GeneratedCodeFile(
        filename="app/main.py",
        content="print('second')",
    )

    with pytest.raises(ValueError):
        MultiFileCodeGenerationSchema(
            files=[
                first_file,
                second_file,
            ]
        )


# ---------------------------------------------------------------------------
# Workflow prerequisites
# ---------------------------------------------------------------------------


def test_node_requires_requirements(state):
    with pytest.raises(
        WorkflowStateError
    ):
        asyncio.run(
            code_generation_node(state)
        )


def test_node_requires_design_context(state):
    state_with_requirements = {
        **state,
        "extracted_requirements": [
            "Create a dashboard"
        ],
    }

    with pytest.raises(
        WorkflowStateError
    ):
        asyncio.run(
            code_generation_node(
                state_with_requirements
            )
        )


async def test_node_requires_non_empty_requirements(
    state_with_design_context,
):
    empty_requirements_state = {
        **state_with_design_context,
        "extracted_requirements": [],
    }

    with pytest.raises(
        WorkflowStateError
    ):
        await code_generation_node(
            empty_requirements_state
        )


async def test_node_requires_none_requirements_rejection(
    state_with_design_context,
):
    none_requirements_state = {
        **state_with_design_context,
        "extracted_requirements": None,
    }

    with pytest.raises(
        WorkflowStateError
    ):
        await code_generation_node(
            none_requirements_state
        )


async def test_node_requires_design_tokens(
    state_with_requirements,
):
    missing_design_tokens_state = {
        **state_with_requirements,
        "design_tokens": {},
    }

    with pytest.raises(
        WorkflowStateError
    ):
        await code_generation_node(
            missing_design_tokens_state
        )


async def test_node_requires_accessibility_requirements(
    state_with_requirements,
):
    missing_accessibility_state = {
        **state_with_requirements,
        "design_tokens": {
            "accessibility": (
                "Use semantic HTML."
            )
        },
        "accessibility_requirements": [],
    }

    with pytest.raises(
        WorkflowStateError
    ):
        await code_generation_node(
            missing_accessibility_state
        )


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


async def test_node_sets_current_step(
    state_with_design_context,
):
    result = await code_generation_node(
        state_with_design_context
    )

    assert (
        result["current_step"]
        == WorkflowStep.CODE_GENERATION
    )


async def test_node_sets_previous_step(
    state_with_design_context,
):
    result = await code_generation_node(
        state_with_design_context
    )

    assert (
        result["previous_step"]
        == WorkflowStep.DESIGN
    )


async def test_node_does_not_mutate_original_state(
    state_with_design_context,
):
    original_step = (
        state_with_design_context[
            "current_step"
        ]
    )

    await code_generation_node(
        state_with_design_context
    )

    assert (
        state_with_design_context[
            "current_step"
        ]
        == original_step
    )


# ---------------------------------------------------------------------------
# Generated-file persistence
# ---------------------------------------------------------------------------


async def test_node_persists_generated_files(
    state_with_design_context,
):
    result = await code_generation_node(
        state_with_design_context
    )

    assert isinstance(
        result["generated_code_files"],
        dict,
    )
    assert result["generated_code_files"]


async def test_node_generated_files_have_safe_paths(
    state_with_design_context,
):
    result = await code_generation_node(
        state_with_design_context
    )

    for filename in result[
        "generated_code_files"
    ]:
        assert not filename.startswith(
            "/"
        )
        assert ".." not in filename
        assert filename.strip()


async def test_node_generated_files_have_non_empty_content(
    state_with_design_context,
):
    result = await code_generation_node(
        state_with_design_context
    )

    for content in result[
        "generated_code_files"
    ].values():
        assert content.strip()


async def test_node_generated_file_count_is_bounded(
    state_with_design_context,
):
    result = await code_generation_node(
        state_with_design_context
    )

    assert (
        len(
            result[
                "generated_code_files"
            ]
        )
        <= 50
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
