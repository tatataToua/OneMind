"""Specialist registry - the single source of truth about who exists.

The router builds its prompt from this registry, the API exposes it, and the UI
renders its agent rail from it. Nothing hardcodes the roster anywhere else, so
adding a specialist is one `SpecialistSpec` and a handler - no changes to the
router, the graph, or the frontend.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpecialistSpec:
    """Everything the system needs to know about one specialist.

    `data_plane` is load-bearing, not decoration. Each specialist owns exactly
    one data source that no other specialist can reach. That constraint is what
    keeps routing unambiguous - if two specialists could answer the same
    question from the same data, they should have been one specialist.
    """

    key: str
    display_name: str
    data_plane: str
    description: str
    tool_names: list[str] = field(default_factory=list)
    exemplars: list[str] = field(default_factory=list)


class SpecialistRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, SpecialistSpec] = {}

    def register(self, spec: SpecialistSpec) -> SpecialistSpec:
        if spec.key in self._specs:
            raise ValueError(f"specialist {spec.key!r} already registered")
        self._specs[spec.key] = spec
        return spec

    def get(self, key: str) -> SpecialistSpec:
        try:
            return self._specs[key]
        except KeyError:
            raise KeyError(f"unknown specialist {key!r}; known: {sorted(self._specs)}") from None

    def keys(self) -> list[str]:
        return list(self._specs)

    def all(self) -> list[SpecialistSpec]:
        return list(self._specs.values())

    def __iter__(self) -> Iterator[SpecialistSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, key: object) -> bool:
        return key in self._specs

    def routing_block(self) -> str:
        """Render the roster as the agent catalogue injected into the router prompt."""
        lines: list[str] = []
        for spec in self._specs.values():
            lines.append(f"- {spec.key}: {spec.description}")
            for example in spec.exemplars:
                lines.append(f"    e.g. {example}")
        return "\n".join(lines)


registry = SpecialistRegistry()
