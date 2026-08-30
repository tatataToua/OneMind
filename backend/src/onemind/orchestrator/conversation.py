"""Session memory: what the system keeps between one question and the next.

## Remember everything, prompt with almost nothing

These are different problems and conflating them is how a context window gets
blown. The store keeps the whole session - every turn, every fact, the whole
redaction vocabulary. What reaches a *prompt* is deliberately narrow, because
`ollama_num_ctx` is 16384 and a specialist's answer prompt already carries up
to 12000 characters of tool output.

    Router        last few turns, compact   resolving "her" is a routing problem
    Specialist    Facts only                needs identifiers, not narrative
    Synthesiser   current turn only         unchanged

Specialists never see conversation history at all. Their prompts stay exactly
the size they are today however long the conversation runs, because `Facts` is
a handful of key-value pairs that does not grow with turn count.

## Why the vocabulary has to span turns

`PHISession` was minted per request. It cannot be, once follow-ups exist: a
fresh vocabulary at turn three means `PHI_PATIENT_1` either fails to rehydrate
or - worse - resolves to a different person than it did at turn one. So the
conversation owns one session across every turn.

That widens the window in which a mapping from placeholder to real identifier
exists, from seconds to the life of the conversation. It is bounded three ways
and none of them are optional: nothing is written to disk, an idle conversation
is evicted, and the number of live conversations is capped. What is never
shared, exactly as before, is one vocabulary between two conversations.

## The id is minted here, not by the client

A client-chosen session id means guessing someone else's hands you their
redaction vocabulary. The first request omits it, the `done` event returns the
one the server made, and later requests echo it back.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from ..agents.base import SpecialistResult
from ..config import settings
from ..guardrails.phi import PHIRedactor, PHISession
from .facts import Facts


@dataclass
class Pending:
    """A question that stopped because it could not tell which patient it meant.

    Names are not unique, so a name-only lookup can match several people, and
    the data plane refuses rather than picking one. Before session memory that
    was the end of it: the asker had to retype the whole question with an MRN.

    Holding the question here turns the refusal into a real question-and-answer.
    The next turn supplies one identifier and the original request is reissued
    with it, so what the person types is `MRN-672113` rather than everything
    they already said.

    `question` is the redacted request, as everything the router reads is.
    """

    question: str
    match_count: int
    turn: int


@dataclass
class Turn:
    """One exchange, in redacted form.

    Redacted because this is what the router is shown on the next turn, and the
    router is the model. Rehydration happens on the way out to the client and
    nowhere else.
    """

    request: str
    answer: str
    agents: list[str] = field(default_factory=list)


class Conversation:
    """One chat session: a redaction vocabulary, established facts, history."""

    def __init__(self, session_id: str, session: PHISession) -> None:
        self.session_id = session_id
        self.phi = session
        self.facts = Facts()
        self.turns: list[Turn] = []
        # Redacted tool output from earlier turns, so the reconciler can compare
        # a claim retrieved at turn two against a chart retrieved at turn six.
        # Scoped to one subject - see `retain`.
        self.evidence: list[SpecialistResult] = []
        self._subject = ""
        # A question waiting on "which of these people did you mean". Cleared as
        # soon as it is resumed or the conversation moves on - a stale one would
        # hijack an unrelated turn that happened to mention an identifier.
        self.pending: Pending | None = None
        # Two requests on one session id would otherwise mutate the same
        # PHISession mid-redaction. Turns serialise; the second waits.
        self.lock = asyncio.Lock()
        self.last_seen = time.monotonic()

    @property
    def turn_number(self) -> int:
        """1-based number of the turn currently being served."""
        return len(self.turns) + 1

    def touch(self) -> None:
        self.last_seen = time.monotonic()

    def record(self, request: str, answer: str, agents: list[str]) -> None:
        self.turns.append(Turn(request=request, answer=answer, agents=list(agents)))
        self.touch()

    def retain(self, results: list[SpecialistResult]) -> None:
        """Keep this turn's retrieved evidence for later comparison.

        Two rules keep the pile from becoming a liability.

        It is scoped to one subject. When `Facts` switches patient, everything
        held about the previous one is dropped rather than merged - the same
        rule `facts.py` applies, for the same reason: a comparison spanning two
        patients is the worst output the reconciler can produce.

        Only the most recent retrieval per tool survives. A claim re-fetched at
        turn six replaces turn two's copy instead of sitting beside it, so the
        reconciler cannot compare a record against a stale version of itself.
        """
        if self.facts.subject != self._subject:
            self.evidence.clear()
            self._subject = self.facts.subject

        fresh = {str(call.get("tool")) for result in results for call in result.tool_calls or []}
        if fresh:
            self.evidence = [
                result
                for result in self.evidence
                if not any(str(c.get("tool")) in fresh for c in result.tool_calls or [])
            ]
        self.evidence.extend(r for r in results if r.tool_calls)
        del self.evidence[: max(0, len(self.evidence) - settings.max_retained_results)]

    def history_block(self, limit: int | None = None) -> str:
        """Recent turns, rendered for the router prompt.

        Answers are truncated hard. The router needs to know what the
        conversation has been about, not what was said about it, and a 4B model
        given several paragraphs of prior prose starts answering from them
        instead of routing.
        """
        limit = settings.history_turns if limit is None else limit
        recent = self.turns[-limit:] if limit > 0 else []
        if not recent:
            return ""
        lines = []
        for i, turn in enumerate(recent, start=len(self.turns) - len(recent) + 1):
            answer = " ".join(turn.answer.split())[:200]
            lines.append(f"{i}. asked: {turn.request}\n   answered: {answer}")
        return "\n".join(lines)


class ConversationStore:
    """In-process conversations, evicted when idle.

    Deliberately not a LangGraph checkpointer. That is the textbook answer and
    it fights this state: `results` carries an `operator.add` reducer, so
    persisting the graph state across turns would append turn two's results to
    turn one's indefinitely. The reset logic to undo that is larger than the
    store it would replace.
    """

    def __init__(self, redactor: PHIRedactor) -> None:
        self._redactor = redactor
        self._live: dict[str, Conversation] = {}

    def __len__(self) -> int:
        return len(self._live)

    def get(self, session_id: str | None) -> Conversation:
        """Return the named conversation, or start one.

        An unknown id starts a fresh conversation rather than erroring. A
        client holding an id the server has already evicted is the normal case
        after an idle spell, and refusing it would turn "your session expired"
        into an error page instead of a cold but working request.
        """
        self._evict()
        if session_id and session_id in self._live:
            conversation = self._live[session_id]
            conversation.touch()
            return conversation
        return self._create()

    def _create(self) -> Conversation:
        conversation = Conversation(uuid.uuid4().hex, self._redactor.session())
        self._live[conversation.session_id] = conversation
        self._enforce_cap()
        return conversation

    def _evict(self) -> None:
        """Drop conversations idle past the TTL.

        Called on access rather than on a timer: a background sweeper is a
        second thing to shut down cleanly, and a store nobody is reading holds
        nothing worth reclaiming urgently.
        """
        cutoff = time.monotonic() - settings.session_ttl_s
        for session_id in [k for k, c in self._live.items() if c.last_seen < cutoff]:
            del self._live[session_id]

    def _enforce_cap(self) -> None:
        """Bound total live sessions, oldest first.

        A cap that drops the least recently used is a worse experience than one
        that never fires, and a better one than unbounded PHI vocabularies
        accumulating in memory until the process dies.
        """
        excess = len(self._live) - settings.max_sessions
        if excess <= 0:
            return
        oldest = sorted(self._live.values(), key=lambda c: c.last_seen)[:excess]
        for conversation in oldest:
            del self._live[conversation.session_id]


__all__ = ["Conversation", "ConversationStore", "Turn"]
