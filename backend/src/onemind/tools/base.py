"""Tool abstraction.

Tools are plain callables with a JSON Schema attached. The schema is handed to
the model for argument extraction via constrained decoding, the same mechanism
the router uses - so an argument that violates the schema is unrepresentable
rather than merely discouraged.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Leading sign and digits anywhere inside whatever the model actually emitted.
_INT = re.compile(r"-?\d+")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    async def call(self, **kwargs: Any) -> Any:
        result = self.fn(**self._coerce(kwargs))
        if inspect.isawaitable(result):
            return await result
        return result

    def _coerce(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Bring arguments to the types the schema already declares.

        Constrained decoding makes a schema violation unrepresentable in
        principle. In practice a 4B model reached `telemetry_series` with
        `days: "{7}"` - the value it meant, wrapped in the braces of the
        template it was copying - and `min(days, 21)` raised
        `'<' not supported between instances of 'int' and 'str'`. The
        specialist then reported a system error to the user for a request that
        was otherwise fine, because the only thing wrong with it was punctuation
        around a number.

        So the schema is enforced here rather than assumed. A value that cannot
        be coerced is dropped instead of passed through: every such parameter
        has a default, and falling back to it answers the question asked, where
        a `TypeError` answers nothing. Values are never invented - a dropped
        argument is one the model failed to supply, which is a case the tools
        already handle.
        """
        properties = self.parameters.get("properties", {})
        out: dict[str, Any] = {}
        for key, value in kwargs.items():
            declared = properties.get(key, {}).get("type")
            if declared == "integer" and not isinstance(value, int):
                digits = _INT.search(str(value))
                if digits is None:
                    continue
                out[key] = int(digits.group())
            elif declared == "string" and not isinstance(value, str):
                out[key] = str(value)
            else:
                out[key] = value
        return out

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
