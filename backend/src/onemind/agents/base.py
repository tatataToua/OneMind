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
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from ..config import settings
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
            return SpecialistResult(
                agent=self.spec.key,
                display_name=self.spec.display_name,
                answer="",
                error=f"{self.spec.display_name} selected no usable tool call",
            )

        answer = await self.provider.complete(
            [
                Message(
                    role="system",
                    content=_ANSWER_SYSTEM.format(
                        display_name=self.spec.display_name,
                        data_plane=self.spec.data_plane,
                        tool_results=json.dumps(executed, indent=2, default=str)[:12000],
                    ),
                ),
                Message(role="user", content=request),
            ]
        )

        return SpecialistResult(
            agent=self.spec.key,
            display_name=self.spec.display_name,
            answer=answer.strip(),
            tool_calls=executed,
            citations=list(dict.fromkeys(citations)),
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
