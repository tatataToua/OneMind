"""The provider seam.

Deliberately narrow: three methods. Everything above this line (router, agents,
synthesiser) is written against `LLMProvider` and knows nothing about Ollama,
Bedrock, or HTTP. That is what makes "how would this go to production?" a
one-line answer instead of a rewrite.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Message(BaseModel):
    role: str
    content: str


class LLMProvider(Protocol):
    """Minimum surface the orchestrator needs from a language model."""

    name: str
    # The model identifier a caller would name to reproduce a run. Every
    # concrete provider sets it; the Protocol carries it so `live_identity`
    # below can read it without a `getattr` guessing game.
    model: str

    async def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> str:
        """Single-shot completion returning plain text."""
        ...

    async def stream(
        self, messages: Sequence[Message], *, temperature: float = 0.0
    ) -> AsyncIterator[str]:
        """Token stream, used for the final synthesised answer."""
        ...

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float = 0.0,
    ) -> T:
        """Constrained decode against a Pydantic schema.

        Implementations must enforce the schema at the decoding layer where the
        backend supports it, rather than parsing free text and hoping. Routing
        correctness depends on this.
        """
        ...


def live_identity(provider: LLMProvider) -> tuple[str, str]:
    """The provider name and model a provider is *currently* answering on.

    A `FallbackProvider` exposes `active` / `active_model` that follow whichever
    of its two backends is live right now; every other provider is simply
    itself. Anything that reports "what answered" - `/api/health`, the `done`
    event the header reads - goes through here, so the two cannot drift into
    disagreeing about a turn that fell through to the hosted model.
    """
    name = getattr(provider, "active", None) or provider.name
    model = getattr(provider, "active_model", None) or getattr(provider, "model", None) or "unknown"
    return name, model
