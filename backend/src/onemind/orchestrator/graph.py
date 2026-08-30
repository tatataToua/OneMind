"""The orchestration graph.

    redact -> route -+-> [specialist] --+-> reconcile -+-> END
                     |   [specialist]   |   (fan-in)   |
                     |                  |              +-> [specialist]  (wave 2)
                     +-> END (clarify)  <--------------+

Fan-out uses LangGraph's `Send`, so each selected specialist is a real
concurrent node invocation and their results merge through an `operator.add`
reducer. That matters beyond tidiness: the trace timeline shows overlapping
agent spans, which is the visible evidence that this is parallel dispatch rather
than a loop.

`reconcile` runs after fan-in and does two jobs. It computes the comparisons
that span two data planes - the ones no single specialist can make, because each
sees only its own source - and it harvests the identifiers those specialists
established into `Facts`. It holds no tools and reads only what the specialists
already retrieved. See `reconcile.py` and `facts.py`.

## The second wave

A specialist that reported itself blocked for want of an identifier is
dispatched again once a sibling has established one. That is the whole of
"agents communicating": they never address each other, they read a blackboard
the orchestrator fills from their own tool output.

Two properties keep this from becoming the open plan-act-observe loop that
`agents/base.py` exists to avoid.

The decision is not a model call. `plan_second_wave` is a set intersection -
does a blocked specialist's declared `needs` overlap the facts now available -
computed in code, the same rule this codebase applies to arithmetic and to
cross-plane comparison.

The count is a counter, not a judgement. `wave` is incremented by `reconcile`
and checked against `settings.max_waves`. There is no prompt that talks the
system into a third pass.

Synthesis deliberately sits outside the graph. The graph's job is deciding and
gathering - both discrete state transitions. Streaming the final answer to the
client is a transport concern, and forcing a token stream through a state
reducer buys nothing. `Orchestrator` below joins the two halves.
"""

from __future__ import annotations

import asyncio
import operator
from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..agents.base import BaseSpecialist, SpecialistResult
from ..config import settings
from ..guardrails.identity import check_subject, confirmation_statement
from ..guardrails.phi import PHIRedactor, PHISession
from ..llm.base import LLMProvider, live_identity
from ..observability.trace import SpanKind, Trace
from . import disambiguate
from .conversation import Conversation, Pending
from .facts import Facts, collect
from .reconcile import Finding, reconcile
from .registry import SpecialistRegistry
from .registry import registry as default_registry
from .router import Router, RoutingDecision
from .synthesizer import Synthesizer, collect_citations, collect_unverified


class OrchestratorState(TypedDict, total=False):
    request: str
    redacted: str
    decision: RoutingDecision
    results: Annotated[list[SpecialistResult], operator.add]
    findings: list[Finding]
    # Dispatch rounds completed. Incremented by `reconcile`, which runs once per
    # round, and read by the conditional edge that decides whether another may
    # start. Not a reducer: conditional edges observe state after the node's
    # write, so a plain int is enough and a plain int cannot drift.
    wave: int
    # Fact keys this round established that were not on the board when the round
    # began. The retry gate, not merely a trace detail - see `plan_second_wave`.
    established: list[str]


class SpecialistTask(TypedDict):
    agent: str
    redacted: str
    wave: int


def build_graph(
    specialists: dict[str, BaseSpecialist],
    router: Router,
    session: PHISession,
    trace: Trace,
    facts: Facts | None = None,
    history: str = "",
    turn: int = 1,
    retained: list[SpecialistResult] | None = None,
    resolve_name: Callable[[str], list[str]] | None = None,
) -> Any:
    """Compile the graph.

    `trace` and `session` are both per-request, so the graph is built per
    request. Compilation is cheap; sharing a redaction vocabulary between two
    users' requests would not be.

    `facts` belongs to the conversation, not the request, and is mutated in
    place - which is exactly how an identifier resolved this turn is still
    available next turn.
    """

    semaphore = asyncio.Semaphore(settings.max_parallel_agents)
    board = facts if facts is not None else Facts()
    carried = retained or []
    # A confirmed name/identifier pair, held from the identity check until
    # `reconcile_node` can put it with the other computed findings. One value
    # per request, written before any specialist runs and read after they all
    # have, so there is nothing to race.
    confirmed_subject: list[Finding] = []

    async def redact_node(state: OrchestratorState) -> dict[str, Any]:
        span = trace.start(SpanKind.GUARDRAIL, "PHI redaction (inbound)")
        if not settings.phi_redaction_enabled:
            trace.end(span, enabled=False, redacted_count=0)
            return {"redacted": state["request"]}

        redacted = session.redact(state["request"])
        trace.end(span, enabled=True, redacted_count=session.count, kinds=session.kinds())
        return {"redacted": redacted}

    async def route_node(state: OrchestratorState) -> dict[str, Any]:
        conflict = _check_identity(state)
        if conflict is not None:
            return {"decision": conflict}

        span = trace.start(SpanKind.ROUTE, "Router", has_history=bool(history.strip()))
        decision = await router.route(state["redacted"], history)
        trace.end(
            span,
            agents=decision.agents,
            is_actionable=decision.is_actionable,
            rationale=decision.rationale,
            clarifying_question=decision.clarifying_question,
        )
        return {"decision": decision}

    def _check_identity(state: OrchestratorState) -> RoutingDecision | None:
        """Does the request agree with itself about whose records to read?

        Runs before routing rather than after retrieval, because a request that
        names two patients has already read the wrong one's records by the time
        there is evidence to compare. See `guardrails/identity.py`.

        Returns a decision only to stop the request. A confirmed pair is kept
        as a finding and the router proceeds normally.
        """
        if resolve_name is None:
            return None

        span = trace.start(SpanKind.GUARDRAIL, "Subject identity")
        try:
            check = check_subject(session, resolve_name, state.get("redacted", ""))
        except Exception as exc:  # noqa: BLE001 - a failed check must not sink the request
            trace.fail(span, exc)
            return None

        trace.end(
            span,
            verdict=check.verdict,
            # Placeholders, never the identifiers themselves: the trace is read
            # by people, and `access-control-and-audit.md` keeps PHI out of it.
            name=check.name_token,
            supplied=check.patient_token,
            candidate_count=len(check.candidates),
        )

        if check.verdict == "confirmed":
            confirmed_subject.append(
                Finding(
                    check="named_patient_matches_supplied_id",
                    verdict="match",
                    statement=confirmation_statement(check),
                    provenance="request name vs request patient_id, against the patient index",
                )
            )
            return None

        if check.verdict != "conflict":
            return None

        return RoutingDecision(
            is_actionable=False,
            clarifying_question=check.question,
            agents=[],
            rationale=(
                "The request names a patient and supplies an identifier that "
                "belongs to a different patient. Answering either one would be "
                "a confident answer about the wrong person."
            ),
        )

    def fan_out(state: OrchestratorState) -> list[Send] | str:
        decision = state["decision"]
        if not decision.is_actionable:
            return END
        return [
            Send("specialist", SpecialistTask(agent=key, redacted=state["redacted"], wave=1))
            for key in decision.agents
        ]

    async def specialist_node(task: SpecialistTask) -> dict[str, Any]:
        agent = specialists[task["agent"]]
        async with semaphore:
            # Facts go in on every wave, not just the second. On a follow-up
            # turn the board is already populated from an earlier one, which is
            # what lets "and her telemetry?" work without any retry at all.
            result = await agent.run(
                task["redacted"],
                trace,
                session,
                facts=board,
                wave=task.get("wave", 1),
            )
        return {"results": [result]}

    async def reconcile_node(state: OrchestratorState) -> dict[str, Any]:
        """Harvest facts, compute the cross-plane comparisons, count the wave.

        Runs once per dispatch round, after every `Send` branch has completed.
        Deliberately inside the graph rather than alongside synthesis in
        `Orchestrator.stream`: the graph decides and gathers, and both of these
        are gathering. They also earn spans this way, and an audit reader needs
        to see both the comparison and the identifier that unblocked a retry.

        A failing check must not sink a request that the specialists already
        answered, so each pass is caught and recorded rather than raised.
        """
        results = state.get("results", [])
        wave = state.get("wave", 0) + 1

        # Values, not just keys. On a follow-up turn `patient_id` is already a
        # key before anyone runs, so a key-set diff reports nothing new when
        # the conversation moves to a second patient - and the specialist
        # blocked for want of *that* patient's id never earns its retry.
        # A key that now names somebody else was established this round every
        # bit as much as a key that did not exist before.
        known = {f.key: f.value for f in board.all()}
        established: list[str] = []
        fact_span = trace.start(SpanKind.MEMORY, "Facts established", wave=wave)
        try:
            collect(results, session, turn=turn, into=board)
            established = sorted(f.key for f in board.all() if known.get(f.key) != f.value)
            # Values are redaction placeholders (see `facts.py`), so recording
            # them here is safe and is the point - the trace has to show what
            # unblocked a second wave, not merely that one happened.
            trace.end(
                fact_span,
                new=established,
                known={f.key: f.value for f in board.all()},
                subject=board.subject,
            )
        except Exception as exc:  # noqa: BLE001 - a bad extractor is not a failed request
            trace.fail(fact_span, exc)

        # Evidence from earlier turns joins this turn's, so a claim retrieved at
        # turn two and a chart retrieved at turn six can be compared. `reconcile`
        # is unchanged: it reads whatever evidence it is handed. Retention is
        # subject-scoped by `Conversation`, so this can only ever widen the
        # comparison within one patient.
        span = trace.start(SpanKind.RECONCILE, "Reconciliation", wave=wave, carried=len(carried))
        try:
            findings = confirmed_subject + reconcile(
                list(carried) + results, state.get("redacted", "")
            )
        except Exception as exc:  # noqa: BLE001 - a bad check is not a failed request
            trace.fail(span, exc)
            return {"findings": list(confirmed_subject), "wave": wave, "established": established}

        trace.end(
            span,
            checks=len(findings),
            verdicts=[f.verdict for f in findings],
        )
        return {"findings": findings, "wave": wave, "established": established}

    def plan_second_wave(state: OrchestratorState) -> list[Send] | str:
        """Re-dispatch specialists a sibling has just unblocked.

        Computed, not decided. A specialist qualifies when it reported itself
        blocked and one of the fact keys it declared it needs was established
        *by this round*. Nothing else re-runs: a specialist that answered is
        finished, and a specialist blocked for a reason no fact addresses stays
        blocked.

        Newly established, not merely available, and the distinction is load
        bearing. On a follow-up turn the board already holds the identifier
        before anyone runs, so a specialist is dispatched with it in hand; if it
        still retrieved nothing, the patient genuinely has no such record.
        Re-running it would ask the identical question and get the identical
        answer, one agent round later.

        A key whose value changed counts as new, which is what makes switching
        patient work: turn two holds a `patient_id` from turn one, so nothing
        would look new, and the specialist blocked on the *new* patient would
        never be retried. It changed, so it is new.
        """
        if state.get("wave", 0) >= settings.max_waves:
            return END

        fresh = set(state.get("established", []))
        if not fresh:
            return END

        retry: list[str] = []
        for result in state.get("results", []):
            spec = specialists.get(result.agent)
            if spec is None or not result.blocked:
                continue
            if set(spec.spec.needs) & fresh and result.agent not in retry:
                retry.append(result.agent)

        if not retry:
            return END

        for key in retry:
            unblocked = {k: board.value(k) for k in specialists[key].spec.needs if k in fresh}
            source = ", ".join(sorted({board.get(k).source for k in unblocked if board.get(k)}))
            span = trace.start(
                SpanKind.MEMORY,
                f"Second wave: {specialists[key].spec.display_name}",
                agent=key,
                unblocked_by=unblocked,
                source=source,
            )
            trace.end(span)

        return [
            Send(
                "specialist",
                SpecialistTask(
                    agent=key,
                    redacted=state.get("redacted", ""),
                    wave=state.get("wave", 1) + 1,
                ),
            )
            for key in retry
        ]

    builder = StateGraph(OrchestratorState)
    builder.add_node("redact", redact_node)
    builder.add_node("route", route_node)
    builder.add_node("specialist", specialist_node)
    builder.add_node("reconcile", reconcile_node)

    builder.add_edge(START, "redact")
    builder.add_edge("redact", "route")
    builder.add_conditional_edges("route", fan_out, ["specialist", END])
    builder.add_edge("specialist", "reconcile")
    builder.add_conditional_edges("reconcile", plan_second_wave, ["specialist", END])

    return builder.compile()


def latest_per_agent(results: list[SpecialistResult]) -> list[SpecialistResult]:
    """Keep one result per specialist, preferring the later wave.

    `results` uses an `operator.add` reducer, so a retried specialist appears
    twice: once blocked with no tool calls, once with its answer. `Evidence`
    ignores the blocked entry either way, but the synthesiser would render two
    sections under a single specialist heading.
    """
    seen: dict[str, SpecialistResult] = {}
    for result in results:
        seen[result.agent] = result
    return list(seen.values())


class OrchestratorOutcome(TypedDict):
    request_id: str
    session_id: str
    # The provider and model that actually served this turn, read after
    # synthesis. With a fallback wired in (`llm/fallback.py`) a turn can run on
    # the hosted model even though `/api/health` named the local one at page
    # load - so the header trusts this over that.
    provider: str
    model: str
    answer: str
    agents: list[str]
    citations: list[str]
    unverified: list[str]
    findings: list[dict[str, Any]]
    facts: list[dict[str, Any]]
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
        resolve_name: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.provider = provider
        self.specialists = specialists
        self.redactor = redactor
        self.registry = roster or default_registry
        # Maps a patient name to the identifiers it matches, for the identity
        # check in `guardrails/identity.py`. Injected, like the redactor's
        # roster of names, so nothing in the orchestrator reaches a store.
        # Absent, the check abstains and behaviour is exactly as before.
        self.resolve_name = resolve_name
        self.router = Router(provider, self.registry)
        self.synthesizer = Synthesizer(provider)

    async def run(
        self,
        request: str,
        trace: Trace | None = None,
        conversation: Conversation | None = None,
    ) -> OrchestratorOutcome:
        """Drain the stream and return the terminal outcome.

        The answer comes from the `done` event, never from concatenating the
        token stream. Tokens are the model's redacted output; only the `done`
        payload has been through outbound re-hydration. Joining tokens here
        silently handed callers of `/api/chat` an answer full of PHI_ tokens.
        """
        trace = trace or Trace()
        outcome: dict[str, Any] = {}
        async for event in self.stream(request, trace, conversation):
            if event["event"] == "done":
                outcome = event["data"]
        return outcome  # type: ignore[return-value]

    async def stream(
        self,
        request: str,
        trace: Trace | None = None,
        conversation: Conversation | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield trace events and answer tokens as they happen.

        `conversation` is optional and its absence reproduces the original
        behaviour exactly - a fresh vocabulary, an empty board, no history - so
        the eval harness and every existing test call this unchanged.
        """
        trace = trace or Trace()
        if conversation is None:
            async for event in self._turn(request, trace, None):
                yield event
            return

        # One turn at a time per conversation. Two concurrent requests would
        # otherwise mutate the same PHISession mid-redaction.
        async with conversation.lock:
            async for event in self._turn(request, trace, conversation):
                yield event

    async def _turn(
        self,
        request: str,
        trace: Trace,
        conversation: Conversation | None,
    ) -> AsyncIterator[dict[str, Any]]:
        session = conversation.phi if conversation else self.redactor.session()
        facts = conversation.facts if conversation else Facts()
        history = conversation.history_block() if conversation else ""
        turn = conversation.turn_number if conversation else 1
        session_id = conversation.session_id if conversation else ""

        # A turn that is nothing but an identifier, arriving after a lookup that
        # could not tell two patients apart, is an answer rather than a new
        # question. Reissue what was asked, scoped by what was just supplied.
        #
        # Done in redaction space, before the graph, so the substitution works on
        # placeholders. Redacting here is safe: `redact` reuses the token it
        # already issued for a value, so the pass inside the graph mints nothing
        # new and the two agree.
        if conversation is not None and conversation.pending is not None:
            probe = session.redact(request) if settings.phi_redaction_enabled else request
            if disambiguate.is_reply(probe):
                resumed = disambiguate.resume(conversation.pending.question, probe)
                span = trace.start(
                    SpanKind.MEMORY,
                    "Resumed after disambiguation",
                    asked_at_turn=conversation.pending.turn,
                    matched=conversation.pending.match_count,
                    question=resumed,
                )
                trace.end(span)
                request = resumed
            # Cleared either way. A held question that survives an unrelated turn
            # would hijack the next one that happens to mention an identifier.
            conversation.pending = None

        graph = build_graph(
            self.specialists,
            self.router,
            session,
            trace,
            facts,
            history,
            turn,
            retained=list(conversation.evidence) if conversation else None,
            resolve_name=self.resolve_name,
        )

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
        results: list[SpecialistResult] = latest_per_agent(state.get("results", []))
        findings: list[Finding] = state.get("findings", [])

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
            synthesised = Synthesizer.needs_synthesis(results, findings)
            span = trace.start(
                SpanKind.SYNTHESIZE,
                "Synthesis" if synthesised else "Direct answer",
                synthesised=synthesised,
                agent_count=len(results),
                findings=len(findings),
            )
            for event in trace.pending():
                yield event
            async for token in self.synthesizer.stream(state["redacted"], results, findings):
                chunks.append(token)
                yield {"event": "token", "data": {"text": token}}
            trace.end(span, answer_chars=sum(len(c) for c in chunks))
            for event in trace.pending():
                yield event

        redacted_answer = "".join(chunks)

        # Recorded before rehydration, deliberately. The router reads this next
        # turn and the router is the model, which must keep seeing placeholders.
        if conversation is not None:
            conversation.record(
                request=state.get("redacted", ""),
                answer=redacted_answer,
                agents=[r.agent for r in results],
            )
            conversation.retain(results)

            # A name that matched several patients leaves the question open. Hold
            # it so the next turn can supply one identifier instead of restating
            # everything - see `disambiguate.py`.
            matched = disambiguate.ambiguity(results)
            if matched > 1:
                conversation.pending = Pending(
                    question=state.get("redacted", ""),
                    match_count=matched,
                    turn=turn,
                )

        restore = trace.start(
            SpanKind.GUARDRAIL, "PHI re-hydration (outbound)", tokens=session.count
        )
        answer = session.rehydrate(redacted_answer)
        # Findings are computed from redacted tool output, so a statement can
        # carry a placeholder wherever it names a patient. They leave through
        # the same door as the answer.
        restored_findings = [
            {
                **finding.model_dump(),
                "statement": session.rehydrate(finding.statement),
                "provenance": session.rehydrate(finding.provenance),
            }
            for finding in findings
        ]
        trace.end(restore, restored=session.count)
        for event in trace.pending():
            yield event
        trace.close()

        # After synthesis on purpose: the synthesiser's stream is the last model
        # call of the turn, so a primary that died anywhere in it has already
        # opened its cooldown and `live_identity` now names the model that
        # carried the answer the client is looking at.
        provider_name, model_name = live_identity(self.provider)

        yield {
            "event": "done",
            "data": OrchestratorOutcome(
                request_id=trace.request_id,
                session_id=session_id,
                provider=provider_name,
                model=model_name,
                answer=answer,
                agents=[r.agent for r in results],
                citations=collect_citations(results),
                unverified=collect_unverified(results),
                findings=restored_findings,
                # Left in redaction space on purpose. This is what the UI shows
                # as "what the system is currently tracking", and a panel that
                # displayed real identifiers would undo the guardrail the rest
                # of the pipeline maintains.
                facts=[f.as_dict() for f in facts.all()],
                clarifying_question=decision.clarifying_question if decision else "",
                is_actionable=bool(decision and decision.is_actionable),
                rationale=decision.rationale if decision else "",
                redacted_request=state.get("redacted", ""),
                phi_redactions=session.count,
                trace=trace.to_dict(),
            ),
        }
