"""The specialist execution loop.

Every specialist runs the same three steps: choose tools, run them, answer from
what came back. Only the spec and the tool set differ - which is the point. A
specialist is a configuration, not a class hierarchy, so adding one does not add
code paths.

The loop is bounded at a single planning round rather than an open
plan-act-observe cycle. A 4B model given an open loop will re-call the same tool
with cosmetically different arguments and never decide it is finished. One
planning round with up to three calls covers every question in the demo set and
cannot fail to terminate.

Between planning and execution sits a grounding check. Constrained decoding
guarantees an identifier argument is a well-formed string; it cannot guarantee
the string came from the request. A small model asked a question that names
nobody will fill the argument from the nearest example it can see - including
the ones in the tool descriptions it was just handed. That call would succeed,
and the specialist would answer with a real patient's record. See `_is_grounded`.

A second check runs after the answer is written, for the independent failure
where a correctly-scoped lookup returns the right record and the model then
asserts something the record does not say. See `guardrails/grounding.py`.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from ..config import settings
from ..guardrails.grounding import ungrounded_values
from ..guardrails.injection import fence, suspicious_spans
from ..guardrails.phi import PHISession
from ..llm.base import LLMProvider, Message
from ..observability.trace import SpanKind, SpanStatus, Trace
from ..orchestrator.facts import Facts
from ..orchestrator.registry import SpecialistSpec
from ..tools.base import Tool, ToolRegistry

MAX_CALLS = 3

_PLAN_SYSTEM = """You are the {display_name} specialist in a healthcare system.
Your data source: {data_plane}

You have these tools:
{tool_specs}
{facts}
Choose the tool calls needed to answer the request. Rules:
- Use at most {max_calls} calls. Prefer one good call over three redundant ones.
- Put every argument in `arguments` as a string; omit arguments you do not know.
- Never invent an identifier. If neither the request nor the established facts \
name one, leave it out and the tool will report what is available.
"""

_ANSWER_SYSTEM = """You are the {display_name} specialist in a healthcare system.
Your data source: {data_plane}

Answer the request using ONLY the tool results below. Rules:
- Every factual claim must come from the tool results. Do not add outside knowledge.
- If the results do not answer the request, say exactly what is missing and what \
identifier you would need.
- Do not do arithmetic the tools already did - quote their computed values.
- Be concise and specific. No preamble, no restating the question.
- Report only what bears on the request. Leave out fields and records that do \
not answer what was asked, and never recite someone's details in the course of \
explaining that those details are not relevant.
- Identifiers that look like PHI_NAME_1 or PHI_PATIENT_2 are redaction \
placeholders. Reuse them verbatim; never guess what is behind them.

TOOL RESULTS
{tool_results}
"""


class ToolCall(BaseModel):
    tool: str
    arguments: dict[str, str] = Field(default_factory=dict)


class SpecialistResult(BaseModel):
    agent: str
    display_name: str
    answer: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    # Values the answer asserts that the tool results do not contain. Carried
    # rather than suppressed: see the note in `_run` on why the answer still
    # ships.
    unverified: list[str] = Field(default_factory=list)
    error: str | None = None
    # Retrieved nothing for want of an identifier, rather than for want of an
    # answer. A structural flag rather than a string match on `error`, because
    # the second-wave trigger in `graph.py` reads it and matching on the text
    # of a message is how that trigger would quietly stop firing.
    blocked: bool = False


def _coerce(value: str, schema: dict[str, Any]) -> Any:
    """Cast a string argument to the type the tool declared.

    The plan schema forces string arguments because a dict with mixed value
    types is awkward to constrain. Tools still want real ints, so convert here
    and fall back to the string when the cast fails.
    """
    kind = schema.get("type")
    text = value.strip()
    try:
        if kind == "integer":
            return int(float(text))
        if kind == "number":
            return float(text)
        if kind == "boolean":
            return text.lower() in {"true", "yes", "1"}
    except ValueError:
        return value
    return value


def _is_identifier(name: str) -> bool:
    """Argument names that select *whose* record is returned.

    These are the ones worth grounding. `metric`, `resource` or `days` being
    wrong yields a wrong answer to the right question; `patient_id` being wrong
    yields a confident answer about a different person.

    `name` is in the set for exactly that reason. It became a lookup key of its
    own when `fhir_search_patient` stopped overloading `patient_id` to mean
    "id or MRN or name", and a name the request never mentioned selects the
    wrong person just as effectively as a wrong id - more quietly, because a
    fabricated name looks like something the user said.
    """
    return name.endswith("_id") or name in {"mrn", "birth_date", "name"}


def _is_coded_value(name: str) -> bool:
    """Argument names carrying a clinical or billing code.

    A different harm from `_is_identifier`, which is why it is a different
    predicate rather than two more entries in that set. A fabricated code does
    not open the wrong person's chart - it produces a confident statement about
    the wrong procedure, in the canonical form that makes such a statement look
    checked.

    Observed live: asked why a claim billed under N18.3 was denied, the planner
    called `validate_code` with E11.9, the sample value in that tool's own
    parameter description. The lookup succeeded, and the answer reported a
    coding mismatch between a real code and one nobody had mentioned.
    """
    return name in {"code", "icd10_code", "cpt_code", "denial_code"}


def _must_be_grounded(name: str) -> bool:
    """Arguments whose value has to trace back to the request.

    The union of the two harms above. Kept separate from both so the call site
    reads as one rule while each rationale stays with its own predicate.
    """
    return _is_identifier(name) or _is_coded_value(name)


# Matches through to the trailing index deliberately. An earlier version ended
# at `\b` after the kind, which never fires: `_` is a word character, so there
# is no boundary between `NAME` and `_1`, and the predicate silently returned
# False for every request it was given.
_SUBJECT_TOKEN = re.compile(r"PHI[_ ]?(?:NAME|PATIENT|MRN|SSN|DOB)[_ ]?\d+", re.IGNORECASE)


def _names_a_subject(request: str) -> bool:
    """True when the request is plainly about a particular person.

    Read off the redaction placeholders rather than by looking for names: by
    the time a specialist sees a request, every identifier the guardrail
    recognised is a token, and a token is a far more reliable signal than any
    name-detection this module could attempt.

    Used for two decisions that turn out to be the same question.

    An omitted scoping argument is only a problem when the question is about
    somebody - "what is the SpO2 threshold" is legitimately unscoped.

    And the board may only speak when the request names nobody - into the
    prompt, and into an omitted argument. Observed live: asked "what is
    PHI_NAME_1 taking?" and then "now look up PHI_NAME_2 and show their labs",
    the planner used the `patient_id` on the board - the first patient's - and
    reported that no data existed for the second. Memory silently outranked an
    explicit instruction, and the system answered confidently about the wrong
    person.

    Both routes have to be closed, and closing only the prompt was the first
    attempt. The planner then omitted the id it did not have, the argument
    repair below filled it in from the board, and the wrong chart was read with
    nothing in the trace showing a model that had been told to.

    So the rule is that the request always wins and memory speaks only into
    silence. The second wave does not depend on this: it supplies the resolved
    identifier by substituting the argument directly, which no prompt can
    talk it out of - and by then the board has been refreshed by this turn's
    own retrieval, so what it substitutes is this turn's patient.

    The consequence when redaction is disabled is deliberate and worth knowing:
    no tokens means no signal, so an under-scoped call proceeds exactly as it
    did before this check existed. Turning the guardrail off turns this off.
    """
    return bool(_SUBJECT_TOKEN.search(request))


def _is_grounded(value: str, request: str, session: PHISession) -> bool:
    """True when `value` traces back to something the request actually said.

    There are exactly two legitimate origins for an identifier at this point.

    It is a redaction placeholder this session issued, in which case
    `rehydrate` turns it back into a real value. Asking whether rehydration
    *changed* the string is the whole test, and it deliberately reuses
    `PHISession`'s tolerance for tokens the model re-spaced or misspelled
    rather than re-implementing that matching here.

    Or it is a literal the request contains verbatim. That is how record ids
    which are deliberately never redacted - claim, device, encounter - reach a
    tool, and it is also the path taken when redaction is switched off.

    Anything else, the model produced from nowhere: the sample value in a tool
    description, a number from pre-training, a plausible-looking guess. Those
    must not become a lookup. The failure is not that such a call errors - it
    is that it *succeeds*, and hands back a real patient's chart in answer to a
    question that never named them.
    """
    text = value.strip()
    if not text:
        return False
    if session.rehydrate(text) != text:
        return True
    return text.casefold() in request.casefold()


class BaseSpecialist:
    def __init__(
        self,
        spec: SpecialistSpec,
        provider: LLMProvider,
        registry: ToolRegistry,
    ) -> None:
        self.spec = spec
        self.provider = provider
        self.tools: list[Tool] = registry.subset(spec.tool_names)
        self._by_name = {t.name: t for t in self.tools}

    # -- prompt fragments ----------------------------------------------------

    def _tool_specs(self) -> str:
        return json.dumps([t.spec() for t in self.tools], indent=2)

    def _plan_model(self) -> type[BaseModel]:
        names = tuple(self._by_name)
        call = create_model(
            f"{self.spec.key}_Call",
            tool=(Literal[names], ...),  # type: ignore[valid-type]
            arguments=(dict[str, str], ...),
        )
        return create_model(
            f"{self.spec.key}_Plan",
            calls=(list[call], ...),  # type: ignore[valid-type]
        )

    # -- execution -----------------------------------------------------------

    async def run(
        self,
        request: str,
        trace: Trace,
        session: PHISession,
        parent_id: str | None = None,
        facts: Facts | None = None,
        wave: int = 1,
    ) -> SpecialistResult:
        span = trace.start(
            SpanKind.AGENT,
            self.spec.display_name,
            parent_id=parent_id,
            agent=self.spec.key,
            data_plane=self.spec.data_plane,
            wave=wave,
        )
        try:
            result = await asyncio.wait_for(
                self._run(request, trace, span, session, facts, wave),
                timeout=settings.agent_timeout_s,
            )
            trace.end(
                span,
                tool_calls=[c["tool"] for c in result.tool_calls],
                answer_chars=len(result.answer),
            )
            return result
        except TimeoutError:
            trace.fail(span, "timed out")
            return SpecialistResult(
                agent=self.spec.key,
                display_name=self.spec.display_name,
                answer="",
                error=f"{self.spec.display_name} timed out",
            )
        except Exception as exc:  # noqa: BLE001 - one specialist must not sink the request
            trace.fail(span, exc)
            return SpecialistResult(
                agent=self.spec.key,
                display_name=self.spec.display_name,
                answer="",
                error=f"{self.spec.display_name} failed: {exc}",
            )

    async def _run(
        self,
        request: str,
        trace: Trace,
        parent: str,
        session: PHISession,
        facts: Facts | None = None,
        wave: int = 1,
    ) -> SpecialistResult:
        plan = await self.provider.structured(
            [
                Message(
                    role="system",
                    content=_PLAN_SYSTEM.format(
                        display_name=self.spec.display_name,
                        data_plane=self.spec.data_plane,
                        tool_specs=self._tool_specs(),
                        max_calls=MAX_CALLS,
                        # The request always beats memory: facts are offered
                        # only when the request names nobody of its own. See
                        # `_names_a_subject`.
                        facts=(
                            facts.as_prompt_block()
                            if facts and not _names_a_subject(request)
                            else ""
                        ),
                    ),
                ),
                Message(role="user", content=request),
            ],
            self._plan_model(),
        )

        executed: list[dict[str, Any]] = []
        citations: list[str] = []
        seen: set[str] = set()
        ungrounded = False

        for call in plan.model_dump().get("calls", [])[:MAX_CALLS]:
            name = call["tool"]
            tool = self._by_name.get(name)
            if tool is None:
                continue

            props = tool.parameters.get("properties", {})
            args = {
                key: _coerce(val, props.get(key, {}))
                for key, val in (call.get("arguments") or {}).items()
                if key in props and str(val).strip()
            }

            # An omitted scoping argument is not the harmless case it looks
            # like. `telemetry_series(patient_id="")` does not fail - it reads
            # every device in the store, so a question about one person becomes
            # a scan across all of them, and the specialist then answers "no
            # data" perfectly confidently.
            #
            # Two repairs, in order. If the board already holds the identifier,
            # use it: it was established by a sibling's retrieval, and it is a
            # placeholder, so it passes the grounding check below unchanged.
            # Otherwise, if the request is plainly about a person we cannot
            # name, decline rather than widen - and report it as blocked, which
            # is what earns this specialist a second wave once someone resolves
            # the identifier.
            # A `needs` key missing from a call that already carries some other
            # identifier is not under-scoping. `claim_lookup(claim_id=...)` is
            # fully scoped to one record and wants no patient id; treating the
            # absence as a gap blocked a correct call and cost a pointless
            # second wave. Only a call with no identifier at all can widen.
            scoped_by_something = any(_is_identifier(k) and str(v).strip() for k, v in args.items())

            # The same rule that governs the prompt governs the arguments, and
            # it has to, or hiding the facts from the planner achieves nothing.
            # Observed live: asked about one patient and then another, the
            # planner correctly omitted the id it did not have, and this repair
            # filled it in from the board - with the *previous* patient's id.
            # The block never reached the prompt and the wrong chart was read
            # anyway, which is worse, because nothing in the trace shows a
            # model that was told to do it.
            names_subject = _names_a_subject(request)

            under_scoped: list[str] = []
            for key in self.spec.needs:
                if key not in props:
                    continue
                established = facts.value(key) if facts else ""
                supplied = str(args.get(key, "")).strip()

                if not supplied:
                    if established and (wave > 1 or not names_subject):
                        # Memory speaks into silence. On wave one that means the
                        # request naming somebody wins outright; on a retry the
                        # board wins, because a retry only happens after this
                        # turn's own retrieval refreshed it - see below.
                        args[key] = established
                    elif names_subject and not scoped_by_something:
                        under_scoped.append(key)
                elif established and wave > 1 and supplied != established:
                    # On a retry only, the board wins. Wave one demonstrably
                    # retrieved nothing with what the model chose - typically a
                    # name placeholder passed as a patient id, which is grounded,
                    # plausible, and unresolvable on a plane that keys by id. The
                    # board holds an identifier a sibling actually resolved, so
                    # preferring it is the entire point of running again.
                    #
                    # Restricted to wave > 1 deliberately: overriding a
                    # first-round argument would override the request itself.
                    args[key] = established

            if under_scoped:
                widened = trace.start(
                    SpanKind.GUARDRAIL,
                    "Unscoped lookup declined",
                    parent_id=parent,
                    tool=name,
                    arguments=sorted(under_scoped),
                )
                trace.end(widened, status=SpanStatus.ERROR, blocked=True)
                ungrounded = True
                continue

            invented = [
                key
                for key, val in args.items()
                if _must_be_grounded(key) and not _is_grounded(str(val), request, session)
            ]

            if invented:
                # The whole call goes, not just the offending argument. These
                # arguments are what scope the lookup, so a call stripped of one
                # either fails on a missing required argument or widens into an
                # unscoped query - both worse than not calling at all.
                #
                # Recorded as a guardrail span because it is exactly the sort of
                # thing an audit reader needs to see: the model tried to open a
                # record for someone the request never mentioned. The rejected
                # values are not logged; they are unverified and may be a real
                # identifier the model guessed correctly.
                blocked = trace.start(
                    SpanKind.GUARDRAIL,
                    "Ungrounded identifier blocked",
                    parent_id=parent,
                    tool=name,
                    arguments=sorted(invented),
                )
                trace.end(blocked, status=SpanStatus.ERROR, blocked=True)
                ungrounded = True
                continue

            # Cheap idempotence guard: the same call twice adds latency, not information.
            signature = f"{name}:{sorted(args.items())}"
            if signature in seen:
                continue
            seen.add(signature)

            # The trace records the model-facing (redacted) arguments, because
            # the trace is an audit record and must not carry PHI values.
            tool_span = trace.start(
                SpanKind.TOOL, name, parent_id=parent, tool=name, arguments=args
            )
            try:
                # Crossing into the trust boundary: placeholders become real
                # lookup keys, because the tool is the system of record and
                # hiding an id from the store it came from protects nothing.
                output = await tool.call(**session.rehydrate_args(args))
                trace.end(tool_span)
            except Exception as exc:  # noqa: BLE001 - report, keep going
                trace.end(tool_span, status=SpanStatus.ERROR, error=str(exc))
                output = {"error": str(exc)}

            # Crossing back out: whatever the record contained is re-redacted
            # before the model is allowed to read it.
            safe_output = session.redact_json(output)
            executed.append({"tool": name, "arguments": args, "result": safe_output})
            citations.extend(_citations_from(safe_output))

        if not executed:
            # Distinguished because the two cases mean different things to the
            # person asking. "No usable call" is the system failing to plan;
            # a blocked lookup is the system declining to answer a question
            # that did not say who it was about, and the fix is theirs.
            reason = (
                "the request does not identify a record it can look up"
                if ungrounded
                else "selected no usable tool call"
            )
            return SpecialistResult(
                agent=self.spec.key,
                display_name=self.spec.display_name,
                answer="",
                error=f"{self.spec.display_name}: {reason}",
                blocked=True,
            )

        evidence = json.dumps(executed, indent=2, default=str)

        # Truncate first, then fence. The other order cuts the closing marker
        # off a long result and reopens the escape the fence exists to close.
        shown = evidence[:12000]

        # Records are not a trusted source of instructions. `fence` is the
        # defence and is unconditional; this is the audit signal, reported the
        # way an unsupported claim is - because a specialist that quietly
        # obeyed a poisoned note and one that answered honestly look identical
        # from the outside, and only the trace can tell them apart.
        if settings.injection_detection_enabled:
            injected = suspicious_spans(shown)
            if injected:
                span = trace.start(
                    SpanKind.GUARDRAIL,
                    "Instruction-like text in retrieved data",
                    parent_id=parent,
                    # Already redacted: `executed` holds `safe_output`. Kept
                    # verbatim otherwise, because the fragment is the finding.
                    fragments=injected[:5],
                )
                trace.end(span, status=SpanStatus.ERROR, count=len(injected))

        answer = await self.provider.complete(
            [
                Message(
                    role="system",
                    content=_ANSWER_SYSTEM.format(
                        display_name=self.spec.display_name,
                        data_plane=self.spec.data_plane,
                        tool_results=fence(shown),
                    ),
                ),
                Message(role="user", content=request),
            ]
        )

        # Checked against the whole evidence set, not the truncated prompt: a
        # value the model could not have seen is a fabrication either way, and
        # grounding against more text can only reduce false positives.
        #
        # The request counts as evidence for this check. A specialist on the
        # wrong data plane correctly reports what it cannot see - "no data on
        # CLM-8849" - and quoting the question is not inventing a record. See
        # the note in `ungrounded_values`.
        unverified = ungrounded_values(answer, evidence, request)
        if unverified:
            # Surfaced, not suppressed. A false positive here would delete a
            # correct answer over a rounded figure, which is a worse failure
            # than showing one flagged value - and a system that says which of
            # its claims it cannot vouch for is more useful than one that
            # silently drops them.
            #
            # Redacted on the way into the trace, and the reason is the whole
            # point of this list: "absent from the evidence" is precisely what
            # PHI the inbound patterns missed looks like. Found by
            # `evals/edge_cases.py` - a phone number spelled out in words
            # reached the model intact, the model re-typed it as digits, and
            # this span published it to the audit log. Codes and identifiers
            # match no PHI pattern and survive the call unchanged, so the
            # diagnostic value is kept.
            flagged = trace.start(
                SpanKind.GUARDRAIL,
                "Unverified values",
                parent_id=parent,
                values=[session.redact(value) for value in unverified],
            )
            trace.end(flagged, status=SpanStatus.ERROR, count=len(unverified))

        return SpecialistResult(
            agent=self.spec.key,
            display_name=self.spec.display_name,
            answer=answer.strip(),
            tool_calls=executed,
            citations=list(dict.fromkeys(citations)),
            unverified=unverified,
            blocked=_retrieved_nothing(executed),
        )


def _retrieved_nothing(executed: list[dict[str, Any]]) -> bool:
    """True when every lookup this specialist ran reported no match.

    The other half of "blocked", and the half the live model actually hits. An
    invented identifier is caught before the call; this is the case where the
    argument was perfectly grounded and still useless - asked about a patient by
    name, the planner passes `PHI_NAME_1` as `patient_id`, which is a real
    placeholder from the request and resolves to a real name, and the telemetry
    plane keys by id and matches nothing.

    Reported as blocked so the second wave can offer the id a sibling resolved.
    The cost of being wrong is one extra agent round returning the same answer;
    the cost of not doing it is that a two-hop question never works.

    Only an explicit `found: False` counts. A tool that does not use the
    convention - `policy_search` returns prose - is never read as empty.
    """
    if not executed:
        return False
    return all(
        isinstance(call.get("result"), dict) and call["result"].get("found") is False
        for call in executed
    )


def _citations_from(output: Any) -> list[str]:
    """Pull citation strings out of a tool result, if it offers any."""
    if not isinstance(output, dict):
        return []
    found: list[str] = []
    for row in output.get("results", []) or []:
        if isinstance(row, dict) and row.get("citation"):
            found.append(str(row["citation"]))
    return found
