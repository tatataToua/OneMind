"""Tool abstraction.

Tools are plain callables with a JSON Schema attached. The schema is handed to
the model for argument extraction via constrained decoding, the same mechanism
the router uses - so an argument that violates the schema is unrepresentable
rather than merely discouraged.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    async def call(self, **kwargs: Any) -> Any:
        result = self.fn(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def spec(self) -> dict[str, Any]:
        """Model-facing description, without the callable."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"unknown tool {name!r}; known: {sorted(self._tools)}") from None

    def subset(self, names: list[str]) -> list[Tool]:
        return [self.get(n) for n in names]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


def tool(
    registry: ToolRegistry,
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering a function as a tool."""

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        registry.register(Tool(name=name, description=description, parameters=parameters, fn=fn))
        return fn

    return decorate


def obj_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
    }


tools = ToolRegistry()
