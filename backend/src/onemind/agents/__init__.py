"""Specialist agents.

Importing this package registers the roster. Specialists are built from specs
rather than subclassed - there is one implementation, `BaseSpecialist`, and four
configurations of it.
"""

from __future__ import annotations

from ..llm.base import LLMProvider
from ..orchestrator.registry import SpecialistRegistry, registry
from ..tools import tools as tool_registry
from ..tools.base import ToolRegistry
from . import catalog  # noqa: F401  (import registers the four specs)
from .base import BaseSpecialist, SpecialistResult


def build_specialists(
    provider: LLMProvider,
    roster: SpecialistRegistry | None = None,
    tools: ToolRegistry | None = None,
) -> dict[str, BaseSpecialist]:
    """Instantiate every registered specialist."""
    roster = roster or registry
    tools = tools or tool_registry
    return {spec.key: BaseSpecialist(spec, provider, tools) for spec in roster.all()}


__all__ = [
    "BaseSpecialist",
    "SpecialistResult",
    "build_specialists",
    "registry",
]
