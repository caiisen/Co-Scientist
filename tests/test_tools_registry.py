from __future__ import annotations

import pytest

from co_scientist.tools import registry


def test_registry_registers_and_lists_tools() -> None:
    @registry.register_tool("example_registry_test")
    def example_tool() -> str:
        return "ok"

    assert registry.get_tool("example_registry_test") is example_tool
    assert "example_registry_test" in registry.list_tools()


def test_registry_rejects_duplicates_and_unknown_tools() -> None:
    @registry.register_tool("duplicate_registry_test")
    def example_tool() -> str:
        return "ok"

    with pytest.raises(ValueError, match="already registered"):
        registry.register_tool("duplicate_registry_test")(example_tool)
    with pytest.raises(KeyError, match="unknown tool"):
        registry.get_tool("missing")
