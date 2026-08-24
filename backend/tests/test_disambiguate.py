"""Resolving a shared name across two turns.

Two things are being protected here. That a person can answer "which patient did
you mean" with one identifier instead of retyping the question - and that a turn
which merely *mentions* an identifier is not mistaken for that answer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from onemind.agents import build_specialists
from onemind.agents.base import SpecialistResult
from onemind.llm.base import Message
from onemind.orchestrator import disambiguate
from onemind.orchestrator.conversation import ConversationStore
from onemind.orchestrator.graph import Orchestrator
from onemind.orchestrator.registry import registry
from onemind.tools import store


def _result(output: dict[str, Any]) -> SpecialistResult:
    return SpecialistResult(
        agent="clinical",
        display_name="Clinical",
        answer="",
        tool_calls=[{"tool": "fhir_search_patient", "arguments": {}, "result": output}],
    )


# -- reading the refusal -----------------------------------------------------


def test_ambiguity_reports_the_count() -> None:
    assert (
        disambiguate.ambiguity([_result({"found": False, "ambiguous": True, "match_count": 2})])
        == 2
    )


def test_a_clean_result_is_not_ambiguous() -> None:
    assert disambiguate.ambiguity([_result({"found": True, "patient_id": "12345"})]) == 0


def test_a_plain_miss_is_not_ambiguous() -> None:
    assert disambiguate.ambiguity([_result({"found": False, "searched_by": "name"})]) == 0


# -- recognising a reply -----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "PHI_MRN_2",
        "MRN-672113",
        "it's PHI_MRN_2",
        "the MRN is MRN-672113",
        "PHI_DOB_1",
        "1979-10-22",
        "PHI_PATIENT_3",
    ],
)
def test_an_identifier_only_turn_is_a_reply(text: str) -> None:
    assert disambiguate.is_reply(text)


@pytest.mark.parametrize(
    "text",
    [
        # Carries its own question - a new request, not an answer to the old one.
        "what are MRN-672113's labs?",
        # Two identifiers is not an answer to "which one".
        "MRN-672113 or MRN-217621",
        # No identifier at all.
        "the first one",
        "yes",
        # Long enough to be saying something of its own.
        "actually never mind that, tell me about the claims for MRN-672113 instead",
    ],
)
def test_other_turns_are_not_replies(text: str) -> None:
    assert not disambiguate.is_reply(text)


# -- rebuilding the question -------------------------------------------------


def test_an_exact_identifier_replaces_the_name() -> None:
    """Substituted, not appended: leaving both in would ask the planner to pick
    between the exact key and the one that already failed."""
    resumed = disambiguate.resume("What medications is PHI_NAME_1 taking?", "PHI_MRN_2")
    assert resumed == "What medications is PHI_MRN_2 taking?"
    assert "PHI_NAME_1" not in resumed


def test_a_birth_date_is_appended_not_substituted() -> None:
    """A date of birth identifies nobody on its own - it narrows the name, which
    is what `store.match_patients` uses it for."""
    resumed = disambiguate.resume("What medications is PHI_NAME_1 taking?", "PHI_DOB_1")
    assert "PHI_NAME_1" in resumed
    assert "birth_date=PHI_DOB_1" in resumed


def test_only_the_first_name_is_replaced() -> None:
    resumed = disambiguate.resume("Compare PHI_NAME_1 and PHI_NAME_2", "PHI_MRN_9")
    assert resumed == "Compare PHI_MRN_9 and PHI_NAME_2"


def test_resume_without_an_identifier_leaves_the_question_alone() -> None:
    assert disambiguate.resume("What is PHI_NAME_1 taking?", "no idea") == (
        "What is PHI_NAME_1 taking?"
    )


# -- end to end --------------------------------------------------------------


class TwinProvider:
    """Searches by name, then by whatever identifier the request now carries."""

    name = "twin"

    def __init__(self) -> None:
        self.requests: list[str] = []

    async def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> str:
        return "specialist answer"

    async def stream(
        self, messages: Sequence[Message], *, temperature: float = 0.0
    ) -> AsyncIterator[str]:
        yield "synthesised answer"

    async def structured(
        self, messages: Sequence[Message], schema: type[BaseModel], *, temperature: float = 0.0
    ) -> BaseModel:
        fields = set(schema.model_fields)
        if "is_actionable" in fields:
            return schema.model_validate(
                {
                    "is_actionable": True,
                    "clarifying_question": "",
                    "agents": ["clinical"],
                    "rationale": "stub",
                }
            )

        request = messages[-1].content
        self.requests.append(request)
        args: dict[str, str] = {}
        for token in request.replace("?", " ").replace("(", " ").replace(")", " ").split():
            if token.startswith("PHI_MRN"):
                args = {"mrn": token}
                break
            if token.startswith("PHI_NAME"):
                args = {"name": token}
        return schema.model_validate(
            {"calls": [{"tool": "fhir_search_patient", "arguments": args}] if args else []}
        )


@pytest.fixture
def twins() -> list[dict[str, Any]]:
    from collections import Counter

    counts = Counter(p["name"] for p in store.patients())
    shared = next(n for n, c in counts.items() if c > 1)
    return [p for p in store.patients() if p["name"] == shared]


async def test_a_shared_name_is_resolved_by_a_follow_up(twins, redactor) -> None:
    """The whole point: the second turn is an MRN, not the question again."""
    provider = TwinProvider()
    orchestrator = Orchestrator(
        provider=provider,
        specialists=build_specialists(provider),
        redactor=redactor,
        roster=registry,
    )
    conversation = ConversationStore(redactor).get(None)

    first = await orchestrator.run(
        f"What medications is {twins[0]['name']} taking?", conversation=conversation
    )
    assert conversation.pending is not None
    assert conversation.pending.match_count == len(twins)
    assert not first["facts"], "an ambiguous match must establish no subject"

    await orchestrator.run(twins[1]["mrn"], conversation=conversation)

    # The specialist was asked the original question, scoped by the new key.
    assert "medications" in provider.requests[-1]
    assert conversation.pending is None
    resolved = conversation.phi.rehydrate(conversation.facts.value("patient_id"))
    assert resolved == twins[1]["patient_id"]


async def test_an_unrelated_turn_clears_the_pending_question(twins, redactor) -> None:
    """A held question must not hijack the next turn that mentions an id."""
    provider = TwinProvider()
    orchestrator = Orchestrator(
        provider=provider,
        specialists=build_specialists(provider),
        redactor=redactor,
        roster=registry,
    )
    conversation = ConversationStore(redactor).get(None)

    await orchestrator.run(
        f"What medications is {twins[0]['name']} taking?", conversation=conversation
    )
    assert conversation.pending is not None

    await orchestrator.run("what does the policy say about retention?", conversation=conversation)
    assert conversation.pending is None
    assert "medications" not in provider.requests[-1]
