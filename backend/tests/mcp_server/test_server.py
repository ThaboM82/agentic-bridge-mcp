"""
Comprehensive tests for the AgenticBridge MCP server.

These tests cover server identity, tool-family registration, server metadata,
health checks, capability discovery, security boundaries, secret leakage
prevention, and deterministic server configuration.
"""

from __future__ import annotations

import asyncio

import pytest

from app.mcp_server.server import (
    AVAILABLE_TOOL_FAMILIES,
    MCP_SERVER_NAME,
    MCP_SERVER_VERSION,
    SECURITY_BOUNDARIES,
    get_mcp_server,
    get_registered_tool_families,
    get_server_health,
    get_server_info,
    list_server_capabilities,
)


# ---------------------------------------------------------------------------
# Server identity
# ---------------------------------------------------------------------------


def test_get_mcp_server():
    server = get_mcp_server()

    assert server is not None
    assert (
        server.name == MCP_SERVER_NAME
    )


def test_mcp_server_name_is_stable():
    assert (
        MCP_SERVER_NAME
        == "agenticbridge-mcp"
    )


def test_mcp_server_version():
    assert (
        MCP_SERVER_VERSION == "1.1.0"
    )


def test_get_mcp_server_returns_consistent_instance():
    first_server = get_mcp_server()
    second_server = get_mcp_server()

    assert first_server is not None
    assert second_server is not None
    assert (
        first_server.name
        == second_server.name
    )
    assert (
        first_server.version
        == second_server.version
    )


# ---------------------------------------------------------------------------
# Tool-family registration
# ---------------------------------------------------------------------------


def test_get_registered_tool_families():
    assert (
        get_registered_tool_families()
        == AVAILABLE_TOOL_FAMILIES
    )
    assert "figma" in (
        AVAILABLE_TOOL_FAMILIES
    )


def test_available_tool_families_are_unique():
    assert len(
        AVAILABLE_TOOL_FAMILIES
    ) == len(
        set(AVAILABLE_TOOL_FAMILIES)
    )


@pytest.mark.parametrize(
    "tool_family",
    [
        "workflow",
        "documents",
        "retrieval",
        "review",
        "cms",
        "figma",
    ],
)
def test_required_tool_families_are_registered(
    tool_family: str,
):
    assert (
        tool_family
        in AVAILABLE_TOOL_FAMILIES
    )
    assert (
        tool_family
        in get_registered_tool_families()
    )


def test_registered_tool_families_are_immutable_snapshot():
    registered = (
        get_registered_tool_families()
    )

    assert registered == (
        AVAILABLE_TOOL_FAMILIES
    )


# ---------------------------------------------------------------------------
# Server metadata
# ---------------------------------------------------------------------------


async def test_get_server_info():
    server_info = await get_server_info()

    assert (
        server_info["server_name"]
        == MCP_SERVER_NAME
    )
    assert (
        server_info["server_version"]
        == MCP_SERVER_VERSION
    )
    assert "figma" in server_info[
        "available_tool_families"
    ]
    assert (
        "tenant_isolation"
        in server_info[
            "security_boundaries"
        ]
    )


async def test_server_info_includes_all_tool_families():
    server_info = await get_server_info()

    assert (
        server_info[
            "available_tool_families"
        ]
        == AVAILABLE_TOOL_FAMILIES
    )


async def test_server_info_includes_all_security_boundaries():
    server_info = await get_server_info()

    assert (
        server_info["security_boundaries"]
        == SECURITY_BOUNDARIES
    )


async def test_server_info_is_deterministic():
    first_info = await get_server_info()
    second_info = await get_server_info()

    assert first_info == second_info


async def test_server_info_includes_required_metadata_fields():
    server_info = await get_server_info()

    assert "server_name" in server_info
    assert (
        "server_version" in server_info
    )
    assert (
        "available_tool_families"
        in server_info
    )
    assert (
        "security_boundaries"
        in server_info
    )


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------


async def test_get_server_health():
    health = await get_server_health()

    assert (
        health["status"] == "healthy"
    )
    assert (
        health["checks"]["server"]
        == "healthy"
    )
    assert (
        health["checks"][
            "tool_registry"
        ]
        == "healthy"
    )
    assert (
        health["checks"][
            "security_policy"
        ]
        == "healthy"
    )


async def test_server_health_includes_required_checks():
    health = await get_server_health()

    assert "server" in health["checks"]
    assert (
        "tool_registry"
        in health["checks"]
    )
    assert (
        "security_policy"
        in health["checks"]
    )


async def test_server_health_checks_are_all_healthy():
    health = await get_server_health()

    assert all(
        check == "healthy"
        for check in health[
            "checks"
        ].values()
    )


async def test_server_health_is_deterministic():
    first_health = (
        await get_server_health()
    )
    second_health = (
        await get_server_health()
    )

    assert first_health == second_health


# ---------------------------------------------------------------------------
# Capability discovery
# ---------------------------------------------------------------------------


async def test_list_server_capabilities():
    capabilities = (
        await list_server_capabilities()
    )

    capability_families = [
        capability["family"]
        for capability in capabilities[
            "capabilities"
        ]
    ]

    assert (
        "workflow" in capability_families
    )
    assert (
        "documents"
        in capability_families
    )
    assert (
        "retrieval"
        in capability_families
    )
    assert (
        "review" in capability_families
    )
    assert "cms" in capability_families
    assert (
        "figma" in capability_families
    )


async def test_capabilities_include_required_fields():
    capabilities = (
        await list_server_capabilities()
    )

    for capability in capabilities[
        "capabilities"
    ]:
        assert "family" in capability
        assert (
            "description"
            in capability
        )
        assert "tools" in capability


async def test_capabilities_cover_registered_tool_families():
    capabilities = (
        await list_server_capabilities()
    )

    capability_families = {
        capability["family"]
        for capability in capabilities[
            "capabilities"
        ]
    }

    assert capability_families == set(
        AVAILABLE_TOOL_FAMILIES
    )


async def test_capability_tools_are_non_empty():
    capabilities = (
        await list_server_capabilities()
    )

    for capability in capabilities[
        "capabilities"
    ]:
        assert capability["tools"]
        assert isinstance(
            capability["tools"],
            list,
        )


async def test_capabilities_are_deterministic():
    first_capabilities = (
        await list_server_capabilities()
    )
    second_capabilities = (
        await list_server_capabilities()
    )

    assert (
        first_capabilities
        == second_capabilities
    )


# ---------------------------------------------------------------------------
# Security boundaries
# ---------------------------------------------------------------------------


def test_security_boundaries():
    assert (
        "tenant_isolation"
        in SECURITY_BOUNDARIES
    )
    assert (
        "scope_authorization"
        in SECURITY_BOUNDARIES
    )
    assert (
        "rate_limiting"
        in SECURITY_BOUNDARIES
    )
    assert (
        "result_sanitization"
        in SECURITY_BOUNDARIES
    )
    assert (
        "human_approval_for_publication"
        in SECURITY_BOUNDARIES
    )


def test_security_boundaries_are_unique():
    assert len(
        SECURITY_BOUNDARIES
    ) == len(set(SECURITY_BOUNDARIES))


@pytest.mark.parametrize(
    "boundary",
    [
        "tenant_isolation",
        "scope_authorization",
        "rate_limiting",
        "result_sanitization",
        "human_approval_for_publication",
    ],
)
def test_required_security_boundaries_are_declared(
    boundary: str,
):
    assert boundary in SECURITY_BOUNDARIES


async def test_server_info_declares_security_boundaries():
    server_info = await get_server_info()

    for boundary in SECURITY_BOUNDARIES:
        assert (
            boundary
            in server_info[
                "security_boundaries"
            ]
        )


# ---------------------------------------------------------------------------
# Secret-leak prevention
# ---------------------------------------------------------------------------


def test_server_info_does_not_leak_secrets():
    server_info = asyncio.run(
        get_server_info()
    )

    serialized_info = str(
        server_info
    ).lower()

    forbidden_fragments = (
        "api_key",
        "apikey",
        "secret",
        "password",
        "token",
        "authorization",
        "postgresql://",
        "redis://",
    )

    for fragment in forbidden_fragments:
        assert (
            fragment
            not in serialized_info
        )


def test_server_health_does_not_leak_secrets():
    health = asyncio.run(
        get_server_health()
    )

    serialized_health = str(
        health
    ).lower()

    forbidden_fragments = (
        "api_key",
        "apikey",
        "secret",
        "password",
        "token",
        "authorization",
        "postgresql://",
        "redis://",
    )

    for fragment in forbidden_fragments:
        assert (
            fragment
            not in serialized_health
        )


def test_capabilities_do_not_leak_secrets():
    capabilities = asyncio.run(
        list_server_capabilities()
    )

    serialized_capabilities = str(
        capabilities
    ).lower()

    forbidden_fragments = (
        "api_key",
        "apikey",
        "secret",
        "password",
        "token",
        "authorization",
        "postgresql://",
        "redis://",
    )

    for fragment in forbidden_fragments:
        assert (
            fragment
            not in serialized_capabilities
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
