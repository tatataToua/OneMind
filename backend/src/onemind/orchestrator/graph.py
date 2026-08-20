"""The orchestration graph.

    redact -> route -+-> [specialist] --+-> END
                     |   [specialist]   |     (fan-in)
                     +-> END (clarify)

Fan-out uses LangGraph's `Send`, so each selected specialist is a real
concurrent node invocation and their results merge through an `operator.add`
reducer. That matters beyond tidiness: the trace timeline shows overlapping
agent spans, which is the visible evidence that this is parallel dispatch rather
than a loop.

Synthesis deliberately sits outside the graph. The graph's job is deciding and
gathering - both discrete state transitions. Streaming the final answer to the
client is a transport concern, and forcing a token stream through a state
reducer buys nothing. `Orchestrator` below joins the two halves.
"""

from __future__ import annotations

import asyncio
import operator
from collections.abc import AsyncIterator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..agents.base import BaseSpecialist, SpecialistResult
from ..config import settings
from ..guardrails.phi import PHIRedactor, PHISession
from ..llm.base import LLMProvider
from ..observability.trace import SpanKind, Trace
from .registry import SpecialistRegistry
from .registry import registry as default_registry
from .router import Router, RoutingDecision
from .synthesizer import Synthesizer, collect_citations


class OrchestratorState(TypedDict, total=False):
    request: str
    redacted: str
    decision: RoutingDecision
    results: Annotated[list[SpecialistResult], operator.add]


class SpecialistTask(TypedDict):
    agent: str
    redacted: str


def build_graph(
    specialists: dict[str, BaseSpecialist],
    router: Router,
    session: PHISession,
    trace: Trace,
) -> Any:
    """Compile the graph.

    `trace` and `session` are both per-request, so the graph is built per
    request. Compilation is cheap; sharing a redaction vocabulary between two
    users' requests would not be.
    """

    semaphore = asyncio.Semaphore(settings.max_parallel_agents)

    async def redact_node(state: OrchestratorState) -> dict[str, Any]:
        span = trace.start(SpanKind.GUARDRAIL, "PHI redaction (inbound)")
        if not settings.phi_redaction_enabled:
            trace.end(span, enabled=False, redacted_count=0)
            return {"redacted": state["request"]}

        redacted = session.redact(state["request"])
        trace.end(span, enabled=True, redacted_count=session.count, kinds=session.kinds())
        return {"redacted": redacted}

    async def route_node(state: OrchestratorState) -> dict[str, Any]:
        span = trace.start(SpanKind.ROUTE, "Router")
        decision = await router.route(state["redacted"])
        trace.end(
            span,
            agents=decision.agents,
            is_actionable=decision.is_actionable,
            rationale=decision.rationale,
            clarifying_question=decision.clarifying_question,
        )
        return {"decision": decision}

    def fan_out(state: OrchestratorState) -> list[Send] | str:
        decision = state["decision"]
        if not decision.is_actionable:
            return END
        return [
            Send("specialist", SpecialistTask(agent=key, redacted=state["redacted"]))
            for key in decision.agents
        ]

    async def specialist_node(task: SpecialistTask) -> dict[str, Any]:
        agent = specialists[task["agent"]]
        async with semaphore:
            result = await agent.run(task["redacted"], trace, session)
        return {"results": [result]}

    builder = StateGraph(OrchestratorState)
    builder.add_node("redact", redact_node)
    builder.add_node("route", route_node)
    builder.add_node("specialist", specialist_node)

    builder.add_edge(START, "redact")
    builder.add_edge("redact", "route")
    builder.add_conditional_edges("route", fan_out, ["specialist", END])
    builder.add_edge("specialist", END)

    return builder.compile()


class OrchestratorOutcome(TypedDict):
    request_id: str
    answer: str
    agents: list[str]
    citations: list[str]
    clarifying_question: str
    is_actionable: bool
    rationale: str
    redacted_request: str
    phi_redactions: int
    trace: dict[str, Any]


class Orchestrator:
    """Public entry point: one request in, one streamed answer plus a trace out."""

    def __init__(
        self,
        provider: LLMProvider,
        specialists: dict[str, BaseSpecialist],
        redactor: PHIRedactor,
        roster: SpecialistRegistry | None = None,
    ) -> None:
        self.provider = provider
        self.specialists = specialists
        self.redactor = redactor
        self.registry = roster or default_registry
        self.router = Router(provider, self.registry)
        self.synthesizer = Synthesizer(provider)

    async def run(self, request: str, trace: Trace | None = None) -> OrchestratorOutcome:
        """Drain the stream and return the terminal outcome.

        The answer comes from the `done` event, never from concatenating the
        token stream. Tokens are the model's redacted output; only the `done`
        payload has been through outbound re-hydration. Joining tokens here
        silently handed callers of `/api/chat` an answer full of PHI_ tokens.
        """
        trace = trace or Trace()
        outcome: dict[str, Any] = {}
        async for event in self.stream(request, trace):
            if event["event"] == "done":
                outcome = event["data"]
        return outcome  # type: ignore[return-value]

    async def stream(
        self, request: str, trace: Trace | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield trace events and answer tokens as they happen."""
        trace = trace or Trace()
        session = self.redactor.session()
        graph = build_graph(self.specialists, self.router, session, trace)

        # Forward span events while the graph runs. Deliberately not
        # `trace.events()`: that consumes until close, and synthesis spans are
        # emitted after the graph returns, so they would never reach the client.
        task = asyncio.create_task(graph.ainvoke(OrchestratorState(request=request)))
        while not task.done():
            event = await trace.next_event(timeout=0.05)
            if event is not None:
                yield event
        for event in trace.pending():
            yield event

        state: OrchestratorState = await task
        decision: RoutingDecision = state.get("decision")  # type: ignore[assignment]
        results: list[SpecialistResult] = state.get("results", [])

        # Preserve the router's ordering; Send fan-in returns completion order.
        if decision and decision.agents:
            rank = {key: i for i, key in enumerate(decision.agents)}
            results = sorted(results, key=lambda r: rank.get(r.agent, 99))

        chunks: list[str] = []
        if decision and not decision.is_actionable:
            text = decision.clarifying_question
            chunks.append(text)
            yield {"event": "token", "data": {"text": text}}
        else:
            span = trace.start(
                SpanKind.SYNTHESIZE,
                "Synthesis" if Synthesizer.needs_synthesis(results) else "Direct answer",
                synthesised=Synthesizer.needs_synthesis(results),
                agent_count=len(results),
            )
            for event in trace.pending():
                yield event
            async for token in self.synthesizer.stream(state["redacted"], results):
                chunks.append(token)
                yield {"event": "token", "data": {"text": token}}
            trace.end(span, answer_chars=sum(len(c) for c in chunks))
            for event in trace.pending():
                yield event

        redacted_answer = "".join(chunks)
        restore = trace.start(
            SpanKind.GUARDRAIL, "PHI re-hydration (outbound)", tokens=session.count
        )
        answer = session.rehydrate(redacted_answer)
        trace.end(restore, restored=session.count)
        for event in trace.pending():
            yield event
        trace.close()

        yield {
            "event": "done",
            "data": OrchestratorOutcome(
                request_id=trace.request_id,
                answer=answer,
                agents=[r.agent for r in results],
                citations=collect_citations(results),
                clarifying_question=decision.clarifying_question if decision else "",
                is_actionable=bool(decision and decision.is_actionable),
                rationale=decision.rationale if decision else "",
                redacted_request=state.get("redacted", ""),
                phi_redactions=session.count,
                trace=trace.to_dict(),
            ),
        }
