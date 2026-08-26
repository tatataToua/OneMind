"""Test fixtures.

Everything here runs without Ollama. The orchestration logic - routing
normalisation, PHI boundary crossing, fan-out, synthesis selection - is
deterministic and deserves deterministic tests. Model quality is measured
separately by `evals/`, which does need a live model.

`StubProvider` dispatches on the shape of the schema it is handed rather than
on call order, so a test does not break when an extra call is added elsewhere.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from onemind.agents import build_specialists
from onemind.guardrails.phi import PHIRedactor, load_known_names
from onemind.llm.base import Message
from onemind.orchestrator.graph import Orchestrator
from onemind.orchestrator.registry import registry
from onemind.tools import store


class StubProvider:
    """Deterministic stand-in for a language model."""

    name = "stub"

    def __init__(
        self,
        *,
        agents: list[str] | None = None,
        is_actionable: bool = True,
        clarifying_question: str = "",
        plans: dict[str, list[dict[str, Any]]] | None = None,
        answer: str = "stub specialist answer",
        synthesis: str = "stub synthesis",
        delay: float = 0.0,
    ) -> None:
        self.agents = agents if agents is not None else ["clinical"]
        self.is_actionable = is_actionable
        self.clarifying_question = clarifying_question
        self.plans = plans or {}
        self.answer = answer
        self.synthesis = synthesis
        # Real inference yields to the event loop. Without a delay the stub
        # returns inline, agents never interleave, and a concurrency assertion
        # would pass or fail on scheduling luck rather than on behaviour.
        self.delay = delay
        self.structured_calls: list[str] = []
        self.complete_calls: list[str] = []
        # Full message lists, not just the user turn. A guard that changes the
        # *system* prompt - the injection fence around tool results - is
        # invisible to `complete_calls` and needs this to be testable at all.
        self.prompts: list[list[Message]] = []
        self.stream_calls: int = 0

    async def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> str:
        self.complete_calls.append(messages[-1].content)
        self.prompts.append(list(messages))
        await asyncio.sleep(self.delay)
        return self.answer

    async def stream(
        self, messages: Sequence[Message], *, temperature: float = 0.0
    ) -> AsyncIterator[str]:
        self.stream_calls += 1
        for chunk in self.synthesis.split(" "):
            yield chunk + " "

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[BaseModel],
        *,
        temperature: float = 0.0,
    ) -> BaseModel:
        fields = set(schema.model_fields)
        self.structured_calls.append(schema.__name__)
        await asyncio.sleep(self.delay)

        if "is_actionable" in fields:
            return schema.model_validate(
                {
                    "is_actionable": self.is_actionable,
                    "clarifying_question": self.clarifying_question,
                    "agents": self.agents if self.is_actionable else [],
                    "rationale": "stub rationale",
                }
            )

        if "calls" in fields:
            # The schema name is "<agent_key>_Plan"; use it to pick that
            # agent's scripted tool calls.
            key = schema.__name__.removesuffix("_Plan")
            return schema.model_validate({"calls": self.plans.get(key, [])})

        raise AssertionError(f"StubProvider cannot satisfy schema {schema.__name__}")


@pytest.fixture
def redactor() -> PHIRedactor:
    return PHIRedactor(known_names=load_known_names(store.patients()))


@pytest.fixture
def session(redactor: PHIRedactor):
    return redactor.session()


@pytest.fixture
def make_orchestrator(redactor: PHIRedactor):
    def _make(provider: StubProvider) -> Orchestrator:
        return Orchestrator(
            provider=provider,
            specialists=build_specialists(provider),
            redactor=redactor,
            roster=registry,
        )

    return _make


@pytest.fixture
def patient() -> dict[str, Any]:
    return store.patients()[0]
