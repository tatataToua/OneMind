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
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from ..config import settings
from ..guardrails.grounding import ungrounded_values
from ..guardrails.phi import PHISession
from ..llm.base import LLMProvider, Message
from ..observability.trace import SpanKind, SpanStatus, Trace
from ..orchestrator.registry import SpecialistSpec
from ..tools.base import Tool, ToolRegistry

MAX_CALLS = 3

_PLAN_SYSTEM = """You are the {display_name} specialist in a healthcare system.
Your data source: {data_plane}

You have these tools:
{tool_specs}

Choose the tool calls needed to answer the request. Rules:
- Use at most {max_calls} calls. Prefer one good call over three redundant ones.
- Put every argument in `arguments` as a string; omit arguments you do not know.
- Never invent an identifier. If the request does not name one, leave it out and \
the tool will report what is available.
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
    """
    return name.endswith("_id") or name in {"mrn", "birth_date"}


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
    ) -> SpecialistResult:
        span = trace.start(
            SpanKind.AGENT,
            self.spec.display_name,
            parent_id=parent_id,
            agent=self.spec.key,
            data_plane=self.spec.data_plane,
        )
        try:
            result = await asyncio.wait_for(
                self._run(request, trace, span, session),
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
        self, request: str, trace: Trace, parent: str, session: PHISession
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
            )

        evidence = json.dumps(executed, indent=2, default=str)
        answer = await self.provider.complete(
            [
                Message(
                    role="system",
                    content=_ANSWER_SYSTEM.format(
                        display_name=self.spec.display_name,
                        data_plane=self.spec.data_plane,
                        tool_results=evidence[:12000],
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
