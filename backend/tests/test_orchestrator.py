"""Router normalisation and end-to-end orchestration, with a stubbed model."""

from __future__ import annotations

import pytest

from onemind.agents.base import SpecialistResult
from onemind.observability.trace import SpanKind, Trace
from onemind.orchestrator.registry import SpecialistRegistry, SpecialistSpec
from onemind.orchestrator.router import Router
from onemind.orchestrator.synthesizer import Synthesizer

from .conftest import StubProvider

# -- registry ---------------------------------------------------------------


def test_registry_rejects_duplicates() -> None:
    reg = SpecialistRegistry()
    spec = SpecialistSpec(key="a", display_name="A", data_plane="x", description="d")
    reg.register(spec)
    with pytest.raises(ValueError):
        reg.register(spec)


def test_routing_block_lists_every_specialist() -> None:
    from onemind.orchestrator.registry import registry

    block = registry.routing_block()
    for spec in registry.all():
        assert spec.key in block


# -- router normalisation ---------------------------------------------------


async def test_router_returns_selected_agents() -> None:
    router = Router(StubProvider(agents=["clinical", "revenue_cycle"]))
    decision = await router.route("anything")
    assert decision.is_actionable
    assert decision.agents == ["clinical", "revenue_cycle"]


async def test_router_drops_duplicate_agents() -> None:
    router = Router(StubProvider(agents=["clinical", "clinical"]))
    assert (await router.route("anything")).agents == ["clinical"]


async def test_actionable_with_no_agents_becomes_a_question() -> None:
    """Constrained decoding guarantees shape, never sense. A model can return
    is_actionable=true with an empty roster; that must not reach dispatch."""
    router = Router(StubProvider(agents=[], is_actionable=True))
    decision = await router.route("anything")
    assert decision.is_actionable is False
    assert decision.clarifying_question


async def test_non_actionable_always_carries_a_question() -> None:
    router = Router(StubProvider(agents=[], is_actionable=False, clarifying_question=""))
    decision = await router.route("vague")
    assert decision.clarifying_question, "must never return an empty prompt to the user"


# -- synthesis selection ----------------------------------------------------


def _result(key: str, answer: str) -> SpecialistResult:
    return SpecialistResult(agent=key, display_name=key.title(), answer=answer)


def test_single_answer_skips_synthesis() -> None:
    assert Synthesizer.needs_synthesis([_result("clinical", "one")]) is False


def test_two_answers_need_synthesis() -> None:
    assert (
        Synthesizer.needs_synthesis([_result("clinical", "one"), _result("compliance", "two")])
        is True
    )


async def test_single_specialist_answer_is_passed_through_verbatim() -> None:
    """Rewriting one good answer costs latency and reliably makes it worse."""
    provider = StubProvider()
    synth = Synthesizer(provider)
    chunks = [c async for c in synth.stream("q", [_result("clinical", "the answer")])]
    assert "".join(chunks) == "the answer"
    assert provider.stream_calls == 0


async def test_all_specialists_failing_produces_an_explanation() -> None:
    synth = Synthesizer(StubProvider())
    failed = SpecialistResult(agent="clinical", display_name="Clinical", answer="", error="boom")
    text = "".join([c async for c in synth.stream("q", [failed])])
    assert "boom" in text


# -- end to end -------------------------------------------------------------


async def test_clarifying_question_short_circuits_dispatch(make_orchestrator) -> None:
    provider = StubProvider(agents=[], is_actionable=False, clarifying_question="Which patient?")
    trace = Trace()
    outcome = await make_orchestrator(provider).run("check the numbers", trace)

    assert outcome["is_actionable"] is False
    assert outcome["answer"] == "Which patient?"
    assert outcome["agents"] == []
    assert not [s for s in trace.spans() if s.kind is SpanKind.AGENT]


async def test_single_agent_dispatch_runs_its_tool(make_orchestrator) -> None:
    provider = StubProvider(
        agents=["clinical"],
        plans={
            "clinical": [
                {"tool": "fhir_search_patient", "arguments": {"patient_id": "PHI_PATIENT_1"}}
            ]
        },
        answer="Metformin 500 mg",
    )
    trace = Trace()
    outcome = await make_orchestrator(provider).run("What is patient 12345 taking?", trace)

    assert outcome["agents"] == ["clinical"]
    assert outcome["answer"] == "Metformin 500 mg"
    assert [s.name for s in trace.spans() if s.kind is SpanKind.TOOL] == ["fhir_search_patient"]


async def test_fan_out_runs_specialists_concurrently(make_orchestrator) -> None:
    provider = StubProvider(
        agents=["clinical", "revenue_cycle"],
        plans={
            "clinical": [
                {"tool": "fhir_search_patient", "arguments": {"patient_id": "PHI_PATIENT_1"}}
            ],
            "revenue_cycle": [{"tool": "denial_summary", "arguments": {}}],
        },
        delay=0.05,
    )
    trace = Trace()
    outcome = await make_orchestrator(provider).run(
        "What is patient 12345 taking, and what is our denial rate?", trace
    )

    assert sorted(outcome["agents"]) == ["clinical", "revenue_cycle"]

    agent_spans = [s for s in trace.spans() if s.kind is SpanKind.AGENT]
    assert len(agent_spans) == 2
    first, second = agent_spans
    assert second.started_at < first.ended_at, "agent spans must overlap"


async def test_results_follow_router_order_not_completion_order(
    make_orchestrator,
) -> None:
    provider = StubProvider(
        agents=["revenue_cycle", "clinical"],
        plans={
            "clinical": [
                {"tool": "fhir_search_patient", "arguments": {"patient_id": "PHI_PATIENT_1"}}
            ],
            "revenue_cycle": [{"tool": "denial_summary", "arguments": {}}],
        },
    )
    outcome = await make_orchestrator(provider).run(
        "What is patient 12345 taking, and what is our denial rate?"
    )
    assert outcome["agents"] == ["revenue_cycle", "clinical"]


async def test_phi_never_reaches_the_model(make_orchestrator) -> None:
    """The request the model saw must contain no identifiers, while the answer
    returned to the caller has them restored."""
    provider = StubProvider(
        agents=["clinical"],
        plans={
            "clinical": [
                {
                    "tool": "fhir_search_patient",
                    "arguments": {"patient_id": "PHI_PATIENT_1"},
                }
            ],
        },
        answer="Patient PHI_PATIENT_1 is stable",
    )
    outcome = await make_orchestrator(provider).run("What is patient 12345 taking?")

    assert "12345" not in outcome["redacted_request"]
    assert "PHI_PATIENT_1" in outcome["redacted_request"]
    assert outcome["phi_redactions"] >= 1
    assert outcome["answer"] == "Patient 12345 is stable"


async def test_tool_receives_the_real_identifier(make_orchestrator) -> None:
    """The regression guard: a placeholder must not reach the data plane."""
    provider = StubProvider(
        agents=["clinical"],
        plans={
            "clinical": [
                {
                    "tool": "fhir_search_patient",
                    "arguments": {"patient_id": "PHI_PATIENT_1"},
                }
            ],
        },
    )
    trace = Trace()
    await make_orchestrator(provider).run("What is patient 12345 taking?", trace)

    tool_span = next(s for s in trace.spans() if s.kind is SpanKind.TOOL)
    # The trace is an audit record, so it keeps the redacted form...
    assert tool_span.detail["arguments"]["patient_id"] == "PHI_PATIENT_1"
    # ...but the tool itself resolved a real patient.
    assert tool_span.status.value == "ok"


async def test_a_failing_specialist_does_not_sink_the_request(
    make_orchestrator,
) -> None:
    provider = StubProvider(
        agents=["clinical", "revenue_cycle"],
        plans={
            "clinical": [{"tool": "fhir_search_patient", "arguments": {}}],
            "revenue_cycle": [{"tool": "denial_summary", "arguments": {}}],
        },
    )
    outcome = await make_orchestrator(provider).run("both please")
    assert len(outcome["agents"]) == 2


async def test_trace_covers_the_whole_pipeline(make_orchestrator) -> None:
    provider = StubProvider(
        agents=["clinical"],
        plans={
            "clinical": [
                {"tool": "fhir_search_patient", "arguments": {"patient_id": "PHI_PATIENT_1"}}
            ]
        },
    )
    trace = Trace()
    await make_orchestrator(provider).run("What is patient 12345 taking?", trace)

    kinds = {span.kind for span in trace.spans()}
    assert kinds == {
        SpanKind.GUARDRAIL,
        SpanKind.ROUTE,
        SpanKind.AGENT,
        SpanKind.TOOL,
        SpanKind.RECONCILE,
        SpanKind.SYNTHESIZE,
    }
    assert all(span.ended_at is not None for span in trace.spans())
