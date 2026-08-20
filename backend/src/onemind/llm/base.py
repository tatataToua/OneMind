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
