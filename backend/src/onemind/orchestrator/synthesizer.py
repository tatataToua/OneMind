"""Merge specialist answers into one response.

When a single specialist ran, its answer is passed through unchanged. Asking a
4B model to rewrite one good answer costs about two seconds and reliably makes
it worse - it pads, hedges, and occasionally contradicts the tool output it was
handed. Synthesis earns its keep only when there is genuinely something to
reconcile.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from ..agents.base import SpecialistResult
from ..llm.base import LLMProvider, Message
from .reconcile import Finding

# Attribution phrasing is given as a rule, not as examples. An earlier version
# offered samples ("Clinical records show...", "On the billing side...") and the
# 4B model reproduced those phrases verbatim regardless of which specialists had
# actually run - attributing telemetry findings to Clinical and policy findings
# to billing. Small models copy examples more readily than they generalise from
# them, so the only specialist names in this prompt are the real ones.
_SYSTEM = """You are combining answers from separate healthcare specialists into \
one reply for the person who asked.

Each section below is headed with the name of the specialist that produced it, \
and each specialist answered from its own data source.

Rules:
- Use ONLY what the specialists reported and the VERIFIED FINDINGS below. Add \
no facts of your own.
- VERIFIED FINDINGS were computed directly from the records the specialists \
retrieved. State them as established fact. Never contradict, hedge, soften or \
re-derive them, and never attribute them to a specialist. If a finding answers \
the question, lead with it.
- When you attribute a statement, name the specialist EXACTLY as it appears in \
the headings below. Never attribute anything to a specialist that is not listed.
- Cover every specialist's contribution. Address the parts of the question in \
the order they were asked.
- Do NOT use the words "conflict", "disagreement", or "discrepancy" unless two \
specialists state OPPOSING facts about the SAME thing. A specialist saying it \
has no data, cannot see a record, or lacks an identifier is a GAP, not a \
conflict. Each specialist only sees its own data source, so silence from one is \
expected and is never evidence against another. Report a gap in one short \
clause and move on.
- Lead with the answer. No preamble, no restating the question, no summary of \
what you are about to do.
- Identifiers like PHI_NAME_1 are redaction placeholders. Reuse them verbatim.
{findings}
SPECIALIST ANSWERS
{answers}
"""

# Rendered only when there is something to render. An empty heading invites a
# 4B model to explain that it has no verified findings, which is noise in an
# answer that may be perfectly well supported without any.
_FINDINGS_BLOCK = """
VERIFIED FINDINGS - computed directly from the retrieved records, not written \
by a specialist. These are established fact.
{lines}
"""


def _format(results: Sequence[SpecialistResult]) -> str:
    blocks: list[str] = []
    for result in results:
        body = result.answer.strip() if result.answer else f"(no answer: {result.error})"
        blocks.append(f"### {result.display_name}\n{body}")
    return "\n\n".join(blocks)


def _format_findings(findings: Sequence[Finding]) -> str:
    """Render findings as their own section, never under a specialist heading.

    Attributing a computed comparison to "Clinical" would be false - no
    specialist could have made it, which is the entire reason the reconciler
    exists - and this prompt already works hard to stop the model inventing
    attributions.
    """
    if not findings:
        return ""
    lines = "\n".join(f"- {f.statement}\n  [{f.provenance}]" for f in findings)
    return _FINDINGS_BLOCK.format(lines=lines)


def collect_citations(results: Sequence[SpecialistResult]) -> list[str]:
    seen: list[str] = []
    for result in results:
        for citation in result.citations:
            if citation not in seen:
                seen.append(citation)
    return seen


def collect_unverified(results: Sequence[SpecialistResult]) -> list[str]:
    """Every value a specialist asserted but could not support from its tools."""
    seen: list[str] = []
    for result in results:
        for value in result.unverified:
            if value not in seen:
                seen.append(value)
    return seen


class Synthesizer:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    @staticmethod
    def needs_synthesis(
        results: Sequence[SpecialistResult],
        findings: Sequence[Finding] = (),
    ) -> bool:
        """Whether there is genuinely something to reconcile.

        Findings count. A specialist can retrieve evidence successfully and
        still return an empty answer - a timeout after its tools ran, a model
        returning whitespace - and without this the single-answer fast path
        would pass the other specialist's prose through untouched and drop
        every computed finding on the floor.
        """
        return len([r for r in results if r.answer.strip()]) > 1 or bool(findings)

    async def stream(
        self,
        request: str,
        results: Sequence[SpecialistResult],
        findings: Sequence[Finding] = (),
    ) -> AsyncIterator[str]:
        answered = [r for r in results if r.answer.strip()]

        if not answered:
            errors = "; ".join(r.error for r in results if r.error) or "no answer"
            yield f"I could not complete that request: {errors}"
            return

        if len(answered) == 1 and not findings:
            # Chunked rather than one blob so the UI renders identically to a
            # real token stream.
            text = answered[0].answer
            for i in range(0, len(text), 24):
                yield text[i : i + 24]
            return

        messages = [
            Message(
                role="system",
                content=_SYSTEM.format(
                    answers=_format(results),
                    findings=_format_findings(findings),
                ),
            ),
            Message(role="user", content=request),
        ]
        async for token in self.provider.stream(messages, temperature=0.2):
            yield token
