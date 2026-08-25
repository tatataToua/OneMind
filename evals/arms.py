"""The two architectures under comparison.

The orchestrator's premise is that splitting the work across four specialists,
with a router in front, beats handing one agent every tool. That is an
architectural claim, and until something measures it, it is only a claim.

Both arms answer the same question - *which data planes should this request
touch?* - so both can be scored by the same function against the same labels.
The router answers it explicitly, as specialist keys. The monolith answers it
implicitly, through the tools it chooses; mapping each tool back to its owning
specialist recovers the same set.

Fairness is the whole point, so it is worth being explicit about what the
monolith gets. It is not a strawman built to lose:

  - the same constrained decode (`LLMProvider.structured`), not free-text
    parsing that would fail for reasons unrelated to architecture;
  - the real tool descriptions, taken from the tool registry, exactly as the
    specialists see them - not paraphrases written by someone who knew which
    arm they wanted to win;
  - the same ability to abstain, via `is_actionable`, so the vague prompts are
    winnable rather than a rigged 10 cases;
  - a system prompt that mirrors the router's sentence for sentence, so the
    comparison isolates the architecture rather than prompt effort.

The one thing it does not get is the router's roster description, because that
is the thing being tested. It gets tool descriptions instead - which is exactly
what a single-agent implementation would have.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Protocol

from pydantic import BaseModel, create_model

from onemind.llm.base import LLMProvider, Message
from onemind.orchestrator.registry import SpecialistRegistry
from onemind.orchestrator.router import Router
from onemind.tools import tools

# Mirrors `router._SYSTEM` clause for clause. Where the router says "agent",
# this says "tool"; the instructions about partial detail and about when to
# refuse are identical, because those are the parts that are not under test.
_MONOLITH_SYSTEM = """You are a healthcare assistant with direct access to every tool \
listed below. You decide WHICH TOOLS to call for a request. You do NOT need every \
detail to choose a tool.

Tools:
{catalogue}

Set is_actionable=true whenever you can tell what the request is about, EVEN IF \
specific details like a patient ID, claim number, or date range are missing. You will \
ask for those yourself. Then list ALL tools needed and set clarifying_question to "".

Set is_actionable=false ONLY when the subject itself is unclear - the request could \
plausibly need any of the tools, or names no topic at all. Then put ONE short question \
in clarifying_question and leave tools empty."""


class Arm(Protocol):
    """One architecture, reduced to the decision both architectures make."""

    name: str

    async def select(self, prompt: str) -> tuple[set[str], str]:
        """Return the specialist keys this arm would reach, and its question if
        it declined to reach any."""
        ...


class RouterArm:
    """The system as built: one routing call over the specialist roster."""

    name = "router"

    def __init__(self, provider: LLMProvider, roster: SpecialistRegistry) -> None:
        self._router = Router(provider, roster)

    async def select(self, prompt: str) -> tuple[set[str], str]:
        decision = await self._router.route(prompt)
        return set(decision.agents), decision.clarifying_question


class MonolithArm:
    """The obvious alternative: one agent holding every tool, no router."""

    name = "monolith"

    def __init__(self, provider: LLMProvider, roster: SpecialistRegistry) -> None:
        self._provider = provider
        # Only tools some specialist actually owns. The orchestrator cannot
        # reach anything else, so offering more would compare two different
        # capability sets rather than two architectures.
        self._owner: dict[str, str] = {}
        for spec in roster.all():
            for tool_name in spec.tool_names:
                self._owner[tool_name] = spec.key

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._owner)

    def _catalogue(self) -> str:
        return "\n".join(f"- {name}: {tools.get(name).description}" for name in self._owner)

    async def select(self, prompt: str) -> tuple[set[str], str]:
        schema = _selection_schema(self.tool_names)
        messages = [
            Message(
                role="system",
                content=_MONOLITH_SYSTEM.format(catalogue=self._catalogue()),
            ),
            Message(role="user", content=prompt),
        ]
        raw = await self._provider.structured(messages, schema)
        data = raw.model_dump()

        chosen = {self._owner[t] for t in data.get("tools", []) if t in self._owner}
        # The same charity the router's `_normalise` extends to itself: a claim
        # of actionability with nothing selected is not actionable.
        actionable = bool(data.get("is_actionable")) and bool(chosen)
        if actionable:
            return chosen, ""

        question = (data.get("clarifying_question") or "").strip()
        return set(), question or "Could you say a bit more about what you need?"


@lru_cache(maxsize=8)
def _selection_schema(names: tuple[str, ...]) -> type[BaseModel]:
    """Tool selection, constrained to the live tool roster.

    Built the same way `router._decision_schema` is built, so an unknown tool
    name is unrepresentable in both arms rather than only in one.
    """
    tool_name = Literal[names]  # type: ignore[valid-type]
    return create_model(
        "ToolSelection",
        is_actionable=(bool, ...),
        clarifying_question=(str, ...),
        tools=(list[tool_name], ...),  # type: ignore[valid-type]
        rationale=(str, ...),
    )


def build_arm(name: str, provider: LLMProvider, roster: SpecialistRegistry) -> Arm:
    if name == "router":
        return RouterArm(provider, roster)
    if name == "monolith":
        return MonolithArm(provider, roster)
    raise ValueError(f"unknown arm {name!r}; expected 'router' or 'monolith'")
