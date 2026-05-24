from __future__ import annotations

from collections.abc import Callable
from typing import Any

ToolCallable = Callable[..., Any]
_TOOLS: dict[str, ToolCallable] = {}


def register_tool(name: str) -> Callable[[ToolCallable], ToolCallable]:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("tool name cannot be empty")

    def decorator(func: ToolCallable) -> ToolCallable:
        if normalized in _TOOLS:
            raise ValueError(f"tool already registered: {normalized}")
        _TOOLS[normalized] = func
        return func

    return decorator


def get_tool(name: str) -> ToolCallable:
    normalized = name.strip().lower()
    try:
        return _TOOLS[normalized]
    except KeyError as exc:
        raise KeyError(f"unknown tool: {normalized}") from exc


def list_tools() -> list[str]:
    return sorted(_TOOLS)


def clear_tools() -> None:
    _TOOLS.clear()
