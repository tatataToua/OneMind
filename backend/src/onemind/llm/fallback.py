"""Prefer the GPU in the room; use the hosted provider when it is not there.

The deployed build wants both halves of a trade that #25 and #26 made
separately. It should run the model the eval numbers were measured on, on the
machine giving the demo, whenever that machine is reachable - and it should
still answer a stranger who opens the link on a Tuesday. Choosing one at
deploy time means choosing wrong half the time.

So the choice moves from deploy time to call time. `primary` is tried first;
when it cannot be reached, `secondary` answers instead and the primary is left
alone for a cooldown. Everything above `base.py` is unchanged - this satisfies
the same three-method protocol, so the router, the specialists and the
synthesiser never learn that there are two models behind it.

Three rules make it predictable rather than merely clever:

  * Only *reachability* failures fall through. An `httpx.HTTPError` covers a
    refused connection, a timeout, and the gateway answering 401 - all of them
    "the model is not available to us". A schema violation is not: that is the
    model answering badly, which is a real signal about the model and must not
    be laundered into a different one.
  * A failure opens a cooldown. One turn is roughly seven calls, and without
    this every one of them would pay its own failed connection first. Between
    demos the tunnel host is dead, so that is the common case, not the rare one.
  * A stream that has already emitted a token cannot fall back. The caller has
    seen text; continuing it from a different model would splice two answers
    together and look like one. Failing is the honest option there.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence
from typing import TypeVar

import httpx
from pydantic import BaseModel

from ..config import settings
from .base import LLMProvider, Message

T = TypeVar("T", bound=BaseModel)

log = logging.getLogger(__name__)


class FallbackProvider:
    """Two providers, one seam. `primary` wins whenever it is reachable."""

    name = "fallback"

    def __init__(
        self,
        primary: LLMProvider,
        secondary: LLMProvider,
        cooldown_s: float | None = None,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.cooldown_s = cooldown_s if cooldown_s is not None else settings.llm_fallback_cooldown_s
        # Monotonic, not wall clock: this is a duration since an event, and a
        # clock adjustment mid-demo should not reopen or extend it.
        self._down_until = 0.0

    # -- state ---------------------------------------------------------------

    @property
    def primary_is_down(self) -> bool:
        return time.monotonic() < self._down_until

    @property
    def active(self) -> str:
        """Which provider the next call would use. What `/api/health` reports."""
        return self.secondary.name if self.primary_is_down else self.primary.name

    @property
    def active_model(self) -> str:
        provider = self.secondary if self.primary_is_down else self.primary
        return getattr(provider, "model", "unknown")

    def _mark_down(self, exc: Exception) -> None:
        self._down_until = time.monotonic() + self.cooldown_s
        # Warning, not debug. Falling back is a correct outcome but never an
        # uninteresting one: it is the difference between the demo everyone
        # thinks they are watching and the one they are.
        log.warning(
            "%s unreachable (%s); answering with %s for the next %.0fs",
            self.primary.name,
            exc,
            self.secondary.name,
            self.cooldown_s,
        )

    # -- LLMProvider ---------------------------------------------------------

    async def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> str:
        if not self.primary_is_down:
            try:
                return await self.primary.complete(messages, temperature=temperature)
            except httpx.HTTPError as exc:
                self._mark_down(exc)
        return await self.secondary.complete(messages, temperature=temperature)

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float = 0.0,
    ) -> T:
        if not self.primary_is_down:
            try:
                return await self.primary.structured(messages, schema, temperature=temperature)
            except httpx.HTTPError as exc:
                self._mark_down(exc)
        return await self.secondary.structured(messages, schema, temperature=temperature)

    async def stream(
        self, messages: Sequence[Message], *, temperature: float = 0.0
    ) -> AsyncIterator[str]:
        if not self.primary_is_down:
            emitted = False
            try:
                async for token in self.primary.stream(messages, temperature=temperature):
                    emitted = True
                    yield token
                return
            except httpx.HTTPError as exc:
                # Past the first token there is no clean seam to fall back on.
                if emitted:
                    raise
                self._mark_down(exc)

        async for token in self.secondary.stream(messages, temperature=temperature):
            yield token

    async def aclose(self) -> None:
        for provider in (self.primary, self.secondary):
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()
