"""Groq provider.

The load-bearing test here is `structured`. Groq's strict mode is a constrained
decode - the same guarantee Ollama's `format` gives - but it refuses schemas
Pydantic emits by default, so `_strictify` has to rewrite them first. If that
rewrite is wrong the API rejects the request and routing stops working, which
is why the schema shape is asserted directly rather than only through a
round-trip.

No network: every test drives the provider through an `httpx.MockTransport`.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel, Field

from onemind.config import settings
from onemind.llm.base import Message
from onemind.llm.groq import GroqError, GroqProvider, _contains_open_map, _strictify


class Inner(BaseModel):
    code: str
    weight: float = 1.0


class Decision(BaseModel):
    is_actionable: bool
    clarifying_question: str = ""
    agents: list[str] = Field(default_factory=list)
    inner: Inner | None = None


def _provider(handler, **kwargs) -> GroqProvider:
    return GroqProvider(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _reply(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload)


def _text_reply(content: str) -> httpx.Response:
    return _reply({"choices": [{"message": {"content": content}}]})


# -- _strictify --------------------------------------------------------------


def test_strictify_marks_every_property_required():
    """Pydantic omits defaulted fields from `required`; strict mode demands all."""
    out = _strictify(Decision.model_json_schema())
    assert set(out["required"]) == set(out["properties"])
    assert "clarifying_question" in out["required"]
    assert "agents" in out["required"]


def test_strictify_forbids_additional_properties():
    out = _strictify(Decision.model_json_schema())
    assert out["additionalProperties"] is False


def test_strictify_recurses_into_defs():
    """Nested models live under `$defs` and are rewritten too, or Groq 400s."""
    out = _strictify(Decision.model_json_schema())
    inner = out["$defs"]["Inner"]
    assert inner["additionalProperties"] is False
    assert set(inner["required"]) == {"code", "weight"}


def test_strictify_does_not_mutate_its_input():
    original = Decision.model_json_schema()
    before = json.dumps(original, sort_keys=True)
    _strictify(original)
    assert json.dumps(original, sort_keys=True) == before


# -- structured --------------------------------------------------------------


async def test_structured_requests_strict_constrained_decode():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _text_reply(
            '{"is_actionable": true, "clarifying_question": "", '
            '"agents": ["clinical"], "inner": null}'
        )

    provider = _provider(handler)
    await provider.structured([Message(role="user", content="hi")], Decision)

    fmt = seen["response_format"]
    assert fmt["type"] == "json_schema"
    # Without strict the model is merely asked to comply, not made to.
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "Decision"
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False
    assert seen["stream"] is False


async def test_structured_parses_into_the_schema():
    def handler(request: httpx.Request) -> httpx.Response:
        return _text_reply(
            '{"is_actionable": true, "clarifying_question": "", '
            '"agents": ["clinical", "compliance"], "inner": null}'
        )

    result = await _provider(handler).structured([Message(role="user", content="hi")], Decision)
    assert isinstance(result, Decision)
    assert result.agents == ["clinical", "compliance"]


async def test_structured_surfaces_the_offending_text():
    """A parse failure names the model and shows what it actually returned."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _text_reply("not json at all")

    with pytest.raises(GroqError, match="not json at all"):
        await _provider(handler).structured([Message(role="user", content="hi")], Decision)


async def test_structured_suppresses_reasoning():
    """qwen3 emits chain-of-thought by default, which corrupts structured output.

    Ollama's provider turns this off with `think: false`; Groq's equivalent is
    `reasoning_format`. Same hazard, same decision - see docs/decisions.md.
    """
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _text_reply(
            '{"is_actionable": false, "clarifying_question": "which patient?", '
            '"agents": [], "inner": null}'
        )

    provider = _provider(handler)
    await provider.structured([Message(role="user", content="hi")], Decision)
    assert seen["reasoning_format"] == "hidden"


# -- complete / stream -------------------------------------------------------


async def test_complete_returns_message_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return _text_reply("an answer")

    out = await _provider(handler).complete([Message(role="user", content="hi")])
    assert out == "an answer"


async def test_stream_yields_content_deltas_and_stops_at_done():
    body = (
        'data: {"choices":[{"delta":{"content":"one "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"two"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=body)

    tokens = [t async for t in _provider(handler).stream([Message(role="user", content="hi")])]
    assert "".join(tokens) == "one two"


async def test_stream_ignores_deltas_carrying_no_content():
    """The first chunk announces the role and carries no text."""
    body = (
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    tokens = [t async for t in _provider(handler).stream([Message(role="user", content="hi")])]
    assert tokens == ["hi"]


# -- configuration -----------------------------------------------------------


async def test_missing_api_key_fails_before_the_request():
    """A key-less deploy should say so, not emit an unauthenticated call."""
    with pytest.raises(GroqError, match="ONEMIND_GROQ_API_KEY"):
        GroqProvider(api_key="")


async def test_authorization_header_carries_the_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return _text_reply("ok")

    await _provider(handler).complete([Message(role="user", content="hi")])
    assert seen["auth"] == "Bearer test-key"


# -- rate limits -------------------------------------------------------------
#
# Groq's free tier is 30 requests/minute. The eval harness runs sequentially and
# will cross that; so will a demo where someone clicks quickly. An unhandled 429
# surfaces to the clinician as a 500, so it is retried here rather than
# translated into a failure the caller cannot act on.


@pytest.fixture
def no_sleep(monkeypatch):
    """Assert on backoff without paying for it."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("onemind.llm.groq.asyncio.sleep", fake_sleep)
    return slept


async def test_retries_a_rate_limit_and_then_succeeds(no_sleep):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "rate limit"}})
        return _text_reply("recovered")

    out = await _provider(handler).complete([Message(role="user", content="hi")])
    assert out == "recovered"
    assert calls["n"] == 2
    assert len(no_sleep) == 1


async def test_retry_honours_the_retry_after_header(no_sleep):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "7"}, json={})
        return _text_reply("ok")

    await _provider(handler).complete([Message(role="user", content="hi")])
    assert no_sleep == [7.0]


async def test_retry_backs_off_when_no_header_is_given(no_sleep):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={})

    with pytest.raises(GroqError, match="429"):
        await _provider(handler).complete([Message(role="user", content="hi")])
    # Exponential, not a flat retry that hammers a limiter already saying no.
    assert no_sleep == sorted(no_sleep) and len(set(no_sleep)) == len(no_sleep)


async def test_gives_up_after_the_configured_attempts(no_sleep):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={})

    with pytest.raises(GroqError):
        await _provider(handler).complete([Message(role="user", content="hi")])
    assert calls["n"] == settings.groq_max_retries + 1


async def test_a_rejected_schema_is_not_retried(no_sleep):
    """400 means the request is wrong. Sending it again wastes the demo's clock
    and buries the message that says what to fix."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "invalid schema"}})

    with pytest.raises(GroqError, match="invalid schema"):
        await _provider(handler).structured([Message(role="user", content="hi")], Decision)
    assert calls["n"] == 1
    assert no_sleep == []


async def test_server_errors_are_retried(no_sleep):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={})
        return _text_reply("ok")

    assert await _provider(handler).complete([Message(role="user", content="hi")]) == "ok"
    assert calls["n"] == 2


# -- open maps and the limit of strict mode ----------------------------------
#
# A specialist plan carries `arguments: dict[str, str]`, which Pydantic emits as
# an open map - `additionalProperties` holding a schema rather than `false`.
# Groq's strict mode refuses that outright ("`additionalProperties:false` must be
# set on every object"), and forcing it closed is worse than refusing: the decode
# then guarantees `arguments` is always `{}`, so every tool call runs unscoped
# and every answer is a refusal. Routing has no open map, which is exactly why it
# scored 100% while every plan came back empty.


class Plan(BaseModel):
    """Shaped like `BaseSpecialist._plan_model`."""

    calls: list[dict[str, str]]


def test_open_map_is_detected():
    assert _contains_open_map(Plan.model_json_schema())


def test_closed_schema_is_not_flagged_as_an_open_map():
    assert not _contains_open_map(Decision.model_json_schema())


def test_strictify_leaves_an_open_map_alone():
    """Closing it is what produced empty arguments in production."""
    out = _strictify(Plan.model_json_schema())
    arguments = out["properties"]["calls"]["items"]
    assert arguments["additionalProperties"] == {"type": "string"}


async def test_a_schema_with_an_open_map_drops_out_of_strict_mode():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _text_reply('{"calls": [{"tool": "fhir_search_patient"}]}')

    await _provider(handler).structured([Message(role="user", content="hi")], Plan)
    fmt = seen["response_format"]["json_schema"]
    assert fmt["strict"] is False
    # Sent unmodified: strict mode is off, so its restrictions do not apply and
    # rewriting the schema can only lose information.
    assert fmt["schema"]["properties"]["calls"]["items"]["additionalProperties"] == {
        "type": "string"
    }


async def test_a_closed_schema_still_uses_strict_mode():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _text_reply(
            '{"is_actionable": true, "clarifying_question": "", "agents": [], "inner": null}'
        )

    await _provider(handler).structured([Message(role="user", content="hi")], Decision)
    assert seen["response_format"]["json_schema"]["strict"] is True


async def test_non_strict_output_is_retried_once_before_failing(no_sleep):
    """Without a decode-time guarantee the model can return prose. One retry
    costs a second; failing the specialist costs the answer."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _text_reply("Sure! Here is the plan you asked for.")
        return _text_reply('{"calls": [{"tool": "fhir_search_patient"}]}')

    result = await _provider(handler).structured([Message(role="user", content="hi")], Plan)
    assert calls["n"] == 2
    assert result.calls == [{"tool": "fhir_search_patient"}]


async def test_strict_output_is_not_retried_on_a_parse_failure(no_sleep):
    """Strict mode guarantees the shape. A parse failure there is a real fault,
    not something a second attempt fixes."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _text_reply("not json")

    with pytest.raises(GroqError):
        await _provider(handler).structured([Message(role="user", content="hi")], Decision)
    assert calls["n"] == 1
