"""Routing: decide which specialists should see a request.

Design note worth knowing before you touch the prompt.

The first version asked the model for a numeric ``confidence`` and gated on a
threshold. That failed against qwen3.5:4b: given "Can you help me with the thing
for the patient?" it returned confidence 0.95 while its own rationale said the
request was "highly ambiguous". Small models do not calibrate numeric
self-assessment.

The fix was to stop asking for a number and ask for a judgement - a boolean
``is_actionable``. Binary classification is something a 4B model does reliably.

The second correction: actionability is about whether the *subject area* is
clear, not whether every parameter is present. A missing patient ID is the
specialist's problem to raise, not a reason to refuse routing. Conflating the
two made the router reject perfectly routable requests.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, create_model

from ..llm.base import LLMProvider, Message
from .registry import SpecialistRegistry, registry

_SYSTEM = """You route healthcare requests to specialist agents. You decide WHICH \
SPECIALIST should handle a request. You do NOT need every detail to route.

Agents:
{catalogue}

Set is_actionable=true whenever you can tell which subject area the request is \
about, EVEN IF specific details like a patient ID, claim number, or date range are \
missing. The specialist will ask for those itself. Then list ALL agents needed and \
set clarifying_question to "".

Set is_actionable=false ONLY when the subject area itself is unclear - the request \
could plausibly belong to any of the agents, or names no topic at all. Then put ONE \
short question in clarifying_question and leave agents empty."""


class RoutingDecision(BaseModel):
    """Normalised routing result handed to the graph."""

    is_actionable: bool
    clarifying_question: str = ""
    agents: list[str] = Field(default_factory=list)
    rationale: str = ""


@lru_cache(maxsize=8)
def _decision_schema(keys: tuple[str, ...]) -> type[BaseModel]:
    """Build a routing schema whose ``agents`` enum is the live roster.

    Generated from the registry rather than hardcoded so a newly registered
    specialist is immediately routable - the constrained decode will not even
    permit a key that does not exist.
    """
    agent_key = Literal[keys]  # type: ignore[valid-type]
    return create_model(
        "RoutingDecisionRaw",
        is_actionable=(bool, ...),
        clarifying_question=(str, ...),
        agents=(list[agent_key], ...),  # type: ignore[valid-type]
        rationale=(str, ...),
    )


class Router:
    def __init__(self, provider: LLMProvider, roster: SpecialistRegistry | None = None) -> None:
        self.provider = provider
        self.registry = roster or registry

    async def route(self, request: str) -> RoutingDecision:
        keys = tuple(self.registry.keys())
        if not keys:
            raise RuntimeError("no specialists registered")

        schema = _decision_schema(keys)
        messages = [
            Message(
                role="system",
                content=_SYSTEM.format(catalogue=self.registry.routing_block()),
            ),
            Message(role="user", content=request),
        ]
        raw = await self.provider.structured(messages, schema)
        return self._normalise(raw)

    def _normalise(self, raw: BaseModel) -> RoutingDecision:
        """Defend against a model that satisfies the schema but not the intent.

        Constrained decoding guarantees shape, never sense. Two failure modes
        show up in practice: duplicate keys, and is_actionable=true paired with
        an empty roster. Both are cheap to repair here and expensive to debug
        three stages downstream.
        """
        data = raw.model_dump()
        seen: list[str] = []
        for key in data.get("agents", []):
            if key in self.registry and key not in seen:
                seen.append(key)

        actionable = bool(data.get("is_actionable")) and bool(seen)
        question = (data.get("clarifying_question") or "").strip()
        if not actionable and not question:
            question = "Could you say a bit more about what you need?"

        return RoutingDecision(
            is_actionable=actionable,
            clarifying_question="" if actionable else question,
            agents=seen if actionable else [],
            rationale=(data.get("rationale") or "").strip(),
        )
