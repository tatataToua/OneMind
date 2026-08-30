"""Preferring the laptop's GPU without depending on it.

The hosted build runs `ollama` as its primary and `groq` behind it, so the
deployed link uses the demo machine when that machine is there and answers
strangers when it is not. The behaviour worth pinning down is not "it works"
but the three edges: what counts as unreachable, that a failure is not paid for
on every call of a turn, and that a stream which has already spoken does not
silently change model halfway through.

No network. Both providers are stubs, and the failure is the exception httpx
would actually raise.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

import httpx
import pytest
from pydantic import BaseModel

from onemind.bootstrap import build_provider
from onemind.config import settings
from onemind.llm.base import Message
from onemind.llm.fallback import FallbackProvider


class Decision(BaseModel):
    agents: list[str]


class StubProvider:
    """A provider that answers, or raises whatever it was given."""

    def __init__(self, name: str, *, model: str = "stub-model", raises: Exception | None = None):
        self.name = name
        self.model = model
        self.raises = raises
        self.calls = 0
        # Tokens emitted before `raises` fires, so a stream can fail mid-answer.
        self.tokens_before_failure = 0
        self.closed = False

    async def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> str:
        self.calls += 1
        if self.raises:
            raise self.raises
        return f"{self.name} answer"

    async def structured(self, messages, schema, *, temperature: float = 0.0):
        self.calls += 1
        if self.raises:
            raise self.raises
        return schema.model_validate({"agents": [self.name]})

    async def stream(self, messages, *, temperature: float = 0.0) -> AsyncIterator[str]:
        self.calls += 1
        for i in range(self.tokens_before_failure):
            yield f"{self.name}{i} "
        if self.raises:
            raise self.raises
        yield f"{self.name} stream"

    async def aclose(self) -> None:
        self.closed = True


UNREACHABLE = httpx.ConnectError("connection refused")
MESSAGES = [Message(role="user", content="hi")]


def _pair(**primary_kwargs) -> tuple[StubProvider, StubProvider, FallbackProvider]:
    primary = StubProvider("ollama", **primary_kwargs)
    secondary = StubProvider("groq", model="qwen/qwen3.8-27b")
    return primary, secondary, FallbackProvider(primary, secondary, cooldown_s=60)


# -- the happy path is the primary -------------------------------------------


async def test_a_reachable_primary_answers_and_the_secondary_is_never_called() -> None:
    primary, secondary, provider = _pair()

    assert await provider.complete(MESSAGES) == "ollama answer"
    assert secondary.calls == 0
    assert provider.active == "ollama"
    assert provider.active_model == "stub-model"


# -- unreachable means unreachable -------------------------------------------


async def test_a_refused_connection_falls_through_to_the_secondary() -> None:
    primary, secondary, provider = _pair(raises=UNREACHABLE)

    assert await provider.complete(MESSAGES) == "groq answer"
    assert primary.calls == 1 and secondary.calls == 1


async def test_the_gateway_refusing_the_token_also_counts_as_unreachable() -> None:
    """A 401 from `llm/gateway.py` is a misconfiguration, not a model failure -
    but from here it is still "we cannot use that model", and a demo that dies
    is worse than one that quietly runs on the hosted provider. `/api/health`
    is what makes the difference visible."""
    refused = httpx.HTTPStatusError(
        "401",
        request=httpx.Request("POST", "https://tunnel/api/chat"),
        response=httpx.Response(401),
    )
    _, secondary, provider = _pair(raises=refused)

    assert await provider.complete(MESSAGES) == "groq answer"
    assert secondary.calls == 1


async def test_a_model_answering_badly_is_not_a_reason_to_change_model() -> None:
    """A schema violation means the primary is reachable and got it wrong. That
    is a real signal about a 4B model, and laundering it into a 27B answer would
    hide the thing the eval exists to measure."""
    _, secondary, provider = _pair(raises=ValueError("failed schema validation"))

    with pytest.raises(ValueError):
        await provider.structured(MESSAGES, Decision)
    assert secondary.calls == 0


# -- the cooldown ------------------------------------------------------------


async def test_one_failure_is_not_paid_for_on_every_call_of_a_turn() -> None:
    """A turn is roughly seven calls. Without a cooldown each would open its own
    doomed connection first, and between demos the tunnel host is always dead."""
    primary, secondary, provider = _pair(raises=UNREACHABLE)

    for _ in range(7):
        await provider.complete(MESSAGES)

    assert primary.calls == 1, "the primary should be tried once, then left alone"
    assert secondary.calls == 7
    assert provider.primary_is_down
    assert provider.active == "groq"
    assert provider.active_model == "qwen/qwen3.8-27b"


async def test_live_identity_names_whichever_half_is_answering() -> None:
    """This is the string the header shows. After a fall-through it has to say
    the hosted model, not the local one the page-load health call reported."""
    from onemind.llm.base import live_identity

    _, _, provider = _pair(raises=UNREACHABLE)
    assert live_identity(provider) == ("ollama", "stub-model")

    await provider.complete(MESSAGES)  # primary refused, cooldown opens
    assert live_identity(provider) == ("groq", "qwen/qwen3.8-27b")


def test_live_identity_of_a_plain_provider_is_the_provider_itself() -> None:
    from onemind.llm.base import live_identity

    assert live_identity(StubProvider("ollama", model="qwen3.5:4b")) == ("ollama", "qwen3.5:4b")


async def test_the_primary_is_tried_again_once_the_cooldown_expires() -> None:
    """The laptop coming back must not need a redeploy to be noticed."""
    primary = StubProvider("ollama", raises=UNREACHABLE)
    secondary = StubProvider("groq")
    provider = FallbackProvider(primary, secondary, cooldown_s=0.05)

    await provider.complete(MESSAGES)
    assert primary.calls == 1

    await asyncio.sleep(0.08)
    primary.raises = None
    assert await provider.complete(MESSAGES) == "ollama answer"
    assert primary.calls == 2


# -- streaming ---------------------------------------------------------------


async def test_a_stream_that_fails_before_its_first_token_falls_back() -> None:
    _, secondary, provider = _pair(raises=UNREACHABLE)

    chunks = [chunk async for chunk in provider.stream(MESSAGES)]

    assert chunks == ["groq stream"]
    assert secondary.calls == 1


async def test_a_stream_that_has_already_spoken_fails_rather_than_switching() -> None:
    """Past the first token the caller has seen text. Continuing it from another
    model splices two answers into one that looks like neither."""
    primary, secondary, provider = _pair(raises=UNREACHABLE)
    primary.tokens_before_failure = 3

    with pytest.raises(httpx.ConnectError):
        async for _ in provider.stream(MESSAGES):
            pass

    assert secondary.calls == 0


# -- construction ------------------------------------------------------------


def test_no_fallback_configured_yields_a_single_provider(monkeypatch) -> None:
    """The local default. An unreachable Ollama is a broken environment and
    should say so rather than quietly reaching for the network."""
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "llm_fallback", "")

    assert build_provider().name == "ollama"


def test_a_configured_fallback_yields_the_pair(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "llm_fallback", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "test-key")

    provider = build_provider()

    assert isinstance(provider, FallbackProvider)
    assert provider.primary.name == "ollama"
    assert provider.secondary.name == "groq"
    assert provider.active == "ollama"


def test_a_fallback_that_cannot_be_built_is_not_fatal(monkeypatch) -> None:
    """`groq` with no key is the ordinary local case, not an error. The
    deployment simply has one provider."""
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "llm_fallback", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "")

    assert build_provider().name == "ollama"


def test_a_fallback_naming_the_primary_is_ignored(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "llm_fallback", "ollama")

    assert build_provider().name == "ollama"


async def test_closing_the_pair_closes_both() -> None:
    primary, secondary, provider = _pair()

    await provider.aclose()

    assert primary.closed and secondary.closed
