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
- Use ONLY what the specialists reported. Add no facts of your own.
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

SPECIALIST ANSWERS
{answers}
"""


def _format(results: Sequence[SpecialistResult]) -> str:
    blocks: list[str] = []
    for result in results:
        body = result.answer.strip() if result.answer else f"(no answer: {result.error})"
        blocks.append(f"### {result.display_name}\n{body}")
    return "\n\n".join(blocks)


def collect_citations(results: Sequence[SpecialistResult]) -> list[str]:
    seen: list[str] = []
    for result in results:
        for citation in result.citations:
            if citation not in seen:
                seen.append(citation)
    return seen


class Synthesizer:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    @staticmethod
    def needs_synthesis(results: Sequence[SpecialistResult]) -> bool:
        return len([r for r in results if r.answer.strip()]) > 1

    async def stream(self, request: str, results: Sequence[SpecialistResult]) -> AsyncIterator[str]:
        answered = [r for r in results if r.answer.strip()]

        if not answered:
            errors = "; ".join(r.error for r in results if r.error) or "no answer"
            yield f"I could not complete that request: {errors}"
            return

        if len(answered) == 1:
            # Chunked rather than one blob so the UI renders identically to a
            # real token stream.
            text = answered[0].answer
            for i in range(0, len(text), 24):
                yield text[i : i + 24]
            return

        messages = [
            Message(role="system", content=_SYSTEM.format(answers=_format(results))),
            Message(role="user", content=request),
        ]
        async for token in self.provider.stream(messages, temperature=0.2):
            yield token
