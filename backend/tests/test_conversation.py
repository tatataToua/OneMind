"""Session memory and the second wave.

The provider here is deliberately less of a stub than `StubProvider`: it reads
the `ESTABLISHED FACTS` block out of the planning prompt and uses the
placeholder it finds. That is what a real model is being asked to do, and it
means these tests fail if the block never reaches the prompt - rather than
passing on a script that pretends it did.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from onemind.agents import build_specialists
from onemind.agents.base import SpecialistResult
from onemind.config import settings
from onemind.llm.base import Message
from onemind.orchestrator.conversation import Conversation, ConversationStore
from onemind.orchestrator.graph import Orchestrator, latest_per_agent
from onemind.orchestrator.registry import registry

_FACT = re.compile(r"^- (\w+): (PHI_[A-Z_]*\d+)$", re.MULTILINE)


class FactAwareProvider:
    """Plans from the established facts when the prompt offers any.

    Without facts, Remote Monitoring invents an identifier - the exact failure
    `_is_grounded` blocks, and the reason the specialist reports itself blocked
    in the first place.
    """

    name = "fact-aware"

    def __init__(self, agents: list[str], clinical_name: str) -> None:
        self.agents = agents
        self.clinical_name = clinical_name
        self.plan_prompts: list[str] = []
        self.route_prompts: list[str] = []

    async def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> str:
        return "specialist answer"

    async def stream(
        self, messages: Sequence[Message], *, temperature: float = 0.0
    ) -> AsyncIterator[str]:
        yield "synthesised answer"

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[BaseModel],
        *,
        temperature: float = 0.0,
    ) -> BaseModel:
        system = messages[0].content
        fields = set(schema.model_fields)
        await asyncio.sleep(0)

        if "is_actionable" in fields:
            self.route_prompts.append(system)
            return schema.model_validate(
                {
                    "is_actionable": True,
                    "clarifying_question": "",
                    "agents": self.agents,
                    "rationale": "stub",
                }
            )

        self.plan_prompts.append(system)
        key = schema.__name__.removesuffix("_Plan")
        facts = dict(_FACT.findall(system))

        if key == "clinical":
            return schema.model_validate(
                {
                    "calls": [
                        {
                            "tool": "fhir_search_patient",
                            # `name`, not `patient_id`: the search tool no longer
                            # overloads one field, and the telemetry plane below
                            # now rejects a name outright.
                            "arguments": {"name": self.clinical_name},
                        }
                    ]
                }
            )

        if key == "remote_monitoring":
            # With a fact on the board, use it. Without one, invent - which is
            # what the grounding guard is there to stop.
            patient = facts.get("patient_id", "99999")
            return schema.model_validate(
                {
                    "calls": [
                        {
                            "tool": "telemetry_series",
                            "arguments": {"patient_id": patient, "metric": "blood_pressure"},
                        }
                    ]
                }
            )

        return schema.model_validate({"calls": []})


@pytest.fixture
def subject() -> dict[str, Any]:
    """A patient whose name resolves to exactly one record.

    `store.patients()[0]` deliberately shares a name with another fixture row,
    so a name lookup there returns the ambiguous-match refusal - correctly, and
    uselessly for a test about what happens once a chart *does* resolve. This
    patient also owns a telemetry device, so the second wave has something to
    retrieve.
    """
    from onemind.tools import store

    return next(p for p in store.patients() if p["patient_id"] == "12678")


@pytest.fixture
def two_hop(redactor):
    """Clinical resolves a name; Remote Monitoring needs the id it produces."""

    def _make() -> tuple[Orchestrator, FactAwareProvider]:
        provider = FactAwareProvider(
            agents=["clinical", "remote_monitoring"],
            # The name is redacted in the request, so this is a placeholder the
            # session issued and `_is_grounded` accepts it.
            clinical_name="PHI_NAME_1",
        )
        orchestrator = Orchestrator(
            provider=provider,
            specialists=build_specialists(provider),
            redactor=redactor,
            roster=registry,
        )
        return orchestrator, provider

    return _make


# -- the second wave ---------------------------------------------------------


async def test_a_blocked_specialist_is_retried_once_a_sibling_unblocks_it(two_hop, subject) -> None:
    orchestrator, provider = two_hop()
    outcome = await orchestrator.run(
        f"Look up {subject['name']} and check the blood pressure trend"
    )

    # Two planning prompts for remote_monitoring: the blocked wave and the retry.
    monitoring_prompts = [p for p in provider.plan_prompts if "Remote Monitoring" in p]
    assert len(monitoring_prompts) == 2

    # Neither prompt offers the facts: this request names its subject, so the
    # request wins and memory stays quiet. The retry works anyway, because the
    # resolved identifier is substituted into the argument rather than
    # suggested to the model. That distinction is the point of the test.
    assert all("ESTABLISHED FACTS" not in p for p in monitoring_prompts)

    assert outcome["agents"].count("remote_monitoring") == 1
    assert any(f["key"] == "patient_id" for f in outcome["facts"])

    # The retry actually reached the telemetry plane with the resolved id.
    telemetry = [
        span
        for span in outcome["trace"]["spans"]
        if span["kind"] == "tool" and span["detail"].get("tool") == "telemetry_series"
    ]
    assert telemetry, "the second wave never ran a lookup"
    assert telemetry[-1]["detail"]["arguments"]["patient_id"].startswith("PHI_PATIENT")


async def test_the_retried_result_replaces_the_blocked_one(two_hop, subject) -> None:
    orchestrator, _ = two_hop()
    outcome = await orchestrator.run(
        f"Look up {subject['name']} and check the blood pressure trend"
    )
    assert sorted(outcome["agents"]) == ["clinical", "remote_monitoring"]


async def test_no_retry_when_nothing_unblocks(redactor) -> None:
    """Compliance declares no needs, so a blocked Compliance stays blocked."""
    provider = FactAwareProvider(agents=["compliance"], clinical_name="PHI_NAME_1")
    orchestrator = Orchestrator(
        provider=provider,
        specialists=build_specialists(provider),
        redactor=redactor,
        roster=registry,
    )
    await orchestrator.run("what does the policy say")
    assert len([p for p in provider.plan_prompts if "Compliance" in p]) == 1


async def test_waves_never_exceed_the_cap(two_hop, subject, monkeypatch) -> None:
    """A specialist still blocked after its retry does not earn a third pass."""

    orchestrator, provider = two_hop()
    # Force the retry to stay blocked: the fact is on the board, but the plan
    # ignores it and invents.
    monkeypatch.setattr(
        provider,
        "structured",
        _always_invents(provider),
    )
    await orchestrator.run(f"Look up {subject['name']} and check the blood pressure trend")

    monitoring = [p for p in provider.plan_prompts if "Remote Monitoring" in p]
    assert len(monitoring) <= settings.max_waves


def _always_invents(provider: FactAwareProvider):
    original = FactAwareProvider.structured

    async def patched(messages, schema, *, temperature: float = 0.0):
        result = await original(provider, messages, schema, temperature=temperature)
        data = result.model_dump()
        for call in data.get("calls", []):
            if call.get("tool") == "telemetry_series":
                call["arguments"]["patient_id"] = "99999"
        if "calls" in data:
            return schema.model_validate(data)
        return result

    return patched


# -- memory across turns -----------------------------------------------------


async def test_a_follow_up_reuses_the_established_fact(two_hop, subject, redactor) -> None:
    """The payoff: turn two needs no retry because turn one left a fact."""
    orchestrator, provider = two_hop()
    store = ConversationStore(redactor)
    conversation = store.get(None)

    await orchestrator.run(
        f"Look up {subject['name']} and check the blood pressure trend",
        conversation=conversation,
    )
    provider.plan_prompts.clear()

    await orchestrator.run("and the blood pressure trend again?", conversation=conversation)

    monitoring = [p for p in provider.plan_prompts if "Remote Monitoring" in p]
    assert len(monitoring) == 1, "turn two should not need a second wave"
    assert "ESTABLISHED FACTS" in monitoring[0]


async def test_phi_tokens_mean_the_same_person_across_turns(two_hop, subject, redactor) -> None:
    """The reason the vocabulary had to stop being per-request."""
    orchestrator, _ = two_hop()
    conversation = ConversationStore(redactor).get(None)

    first = await orchestrator.run(
        f"Look up {subject['name']} and check the blood pressure trend",
        conversation=conversation,
    )
    second = await orchestrator.run("and again?", conversation=conversation)

    assert first["session_id"] == second["session_id"]
    token = conversation.facts.value("patient_id")
    assert token.startswith("PHI_")
    assert conversation.phi.rehydrate(token) == str(subject["patient_id"])


async def test_history_reaches_the_router_and_facts_do_not(two_hop, subject, redactor) -> None:
    orchestrator, provider = two_hop()
    conversation = ConversationStore(redactor).get(None)

    await orchestrator.run(
        f"Look up {subject['name']} and check the blood pressure trend",
        conversation=conversation,
    )
    provider.route_prompts.clear()
    provider.plan_prompts.clear()
    await orchestrator.run("and again?", conversation=conversation)

    assert "EARLIER IN THIS CONVERSATION" in provider.route_prompts[0]
    # Specialists get facts, never the transcript.
    assert all("EARLIER IN THIS CONVERSATION" not in p for p in provider.plan_prompts)


async def test_a_standalone_request_creates_no_memory(two_hop, subject) -> None:
    """`conversation=None` must reproduce the original behaviour exactly."""
    orchestrator, _ = two_hop()
    outcome = await orchestrator.run(f"Look up {subject['name']}")
    assert outcome["session_id"] == ""


# -- the store ---------------------------------------------------------------


def test_two_sessions_share_no_vocabulary(redactor) -> None:
    store = ConversationStore(redactor)
    a, b = store.get(None), store.get(None)
    assert a.session_id != b.session_id
    assert a.phi is not b.phi
    assert a.facts is not b.facts


def test_an_unknown_session_id_starts_a_new_conversation(redactor) -> None:
    """An evicted id is the normal case after an idle spell, not an error."""
    store = ConversationStore(redactor)
    conversation = store.get("does-not-exist")
    assert conversation.session_id != "does-not-exist"


def test_the_same_id_returns_the_same_conversation(redactor) -> None:
    store = ConversationStore(redactor)
    first = store.get(None)
    assert store.get(first.session_id) is first


def test_idle_conversations_are_evicted(redactor) -> None:
    """Back-dated rather than slept on: `monotonic()` is coarse enough on
    Windows that a zero TTL and a just-created conversation compare equal."""
    store = ConversationStore(redactor)
    stale = store.get(None)
    stale.last_seen -= settings.session_ttl_s + 1

    assert store.get(stale.session_id).session_id != stale.session_id
    assert len(store) == 1


def test_live_sessions_are_capped(redactor, monkeypatch) -> None:
    monkeypatch.setattr(settings, "max_sessions", 3)
    store = ConversationStore(redactor)
    for _ in range(6):
        store.get(None)
    assert len(store) == 3


async def test_concurrent_turns_on_one_session_serialise(two_hop, subject, redactor) -> None:
    orchestrator, _ = two_hop()
    conversation = ConversationStore(redactor).get(None)

    await asyncio.gather(
        orchestrator.run(f"Look up {subject['name']}", conversation=conversation),
        orchestrator.run("and again?", conversation=conversation),
    )
    assert len(conversation.turns) == 2


# -- retention ---------------------------------------------------------------


def _retained(agent: str, tool: str) -> SpecialistResult:
    return SpecialistResult(
        agent=agent,
        display_name=agent.title(),
        answer="a",
        tool_calls=[{"tool": tool, "arguments": {}, "result": {"found": True}}],
    )


def test_retention_replaces_a_re_fetched_tool(redactor) -> None:
    """A record must never be compared against a stale copy of itself."""
    conversation = Conversation("s", redactor.session())
    conversation.retain([_retained("revenue_cycle", "claim_lookup")])
    conversation.retain([_retained("revenue_cycle", "claim_lookup")])
    assert len(conversation.evidence) == 1


def test_retention_keeps_different_tools(redactor) -> None:
    conversation = Conversation("s", redactor.session())
    conversation.retain([_retained("revenue_cycle", "claim_lookup")])
    conversation.retain([_retained("clinical", "fhir_search_patient")])
    assert len(conversation.evidence) == 2


def test_switching_subject_clears_retained_evidence(redactor) -> None:
    conversation = Conversation("s", redactor.session())
    conversation.facts.set("patient_id", "PHI_PATIENT_1", source="x")
    conversation.retain([_retained("revenue_cycle", "claim_lookup")])
    assert conversation.evidence

    conversation.facts.set("patient_id", "PHI_PATIENT_2", source="x")
    conversation.retain([_retained("clinical", "fhir_search_patient")])
    assert [c["tool"] for r in conversation.evidence for c in r.tool_calls] == [
        "fhir_search_patient"
    ]


def test_a_blocked_result_is_not_retained(redactor) -> None:
    conversation = Conversation("s", redactor.session())
    conversation.retain([SpecialistResult(agent="a", display_name="A", answer="", blocked=True)])
    assert conversation.evidence == []


def test_retention_is_capped(redactor, monkeypatch) -> None:
    monkeypatch.setattr(settings, "max_retained_results", 2)
    conversation = Conversation("s", redactor.session())
    for i in range(5):
        conversation.retain([_retained(f"agent{i}", f"tool{i}")])
    assert len(conversation.evidence) == 2


# -- helpers -----------------------------------------------------------------


def test_latest_per_agent_prefers_the_later_wave() -> None:
    blocked = SpecialistResult(agent="a", display_name="A", answer="", blocked=True)
    answered = SpecialistResult(agent="a", display_name="A", answer="done")
    assert latest_per_agent([blocked, answered]) == [answered]


def test_history_block_is_empty_before_any_turn(redactor) -> None:
    assert Conversation("s", redactor.session()).history_block() == ""
