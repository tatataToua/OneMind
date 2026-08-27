"""Groq provider - what runs when this is hosted.

Chosen over a frontier API for one reason: Groq serves open-weight models, so
`groq_model` can stay in the same family as the laptop's `qwen3.5:4b`. The eval
numbers in the README stay comparable, and decision 7 ("why this model, not the
9B") keeps meaning something. Swapping in a proprietary model would have made
the router-vs-monolith comparison a measurement of somebody else's model rather
than of this architecture.

The API is OpenAI-compatible, so this talks to it over plain `httpx` rather than
through a vendor SDK - the same call made in decision 8 for Ollama, for the same
reason: two endpoints and three parameters do not justify a dependency.

Two provider-specific controls matter enough to justify owning this code:

  * `response_format` with `strict: true` - a JSON Schema enforced during
    decoding. Groq's equivalent of Ollama's `format`, and the reason routing
    output parses reliably rather than being scraped. Strict mode is only
    offered on some models; see `groq_model` in config.py.
  * `reasoning_format` - qwen3 emits chain-of-thought by default, which
    corrupts structured output. This is Groq's `think: false`.

Strict mode will not accept the JSON Schema Pydantic emits, so `_strictify`
rewrites it first. That function is the fiddly part of this file.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from ..config import settings
from .base import Message

T = TypeVar("T", bound=BaseModel)


class GroqError(RuntimeError):
    pass


def _contains_open_map(schema: dict[str, Any]) -> bool:
    """Does this schema contain a map with arbitrary keys?

    `dict[str, str]` - which is how a specialist plan carries tool arguments -
    emits `additionalProperties` holding a *schema* rather than `false`. Strict
    mode rejects that outright, and closing it is worse than being rejected: the
    decode then guarantees the map is always empty, so every tool call runs with
    no arguments and every answer becomes a refusal.

    So a schema containing one drops out of strict mode instead. See
    `structured`.
    """

    def walk(node: Any) -> bool:
        if isinstance(node, list):
            return any(walk(item) for item in node)
        if not isinstance(node, dict):
            return False
        if isinstance(node.get("additionalProperties"), dict):
            return True
        return any(walk(value) for value in node.values())

    return walk(schema)


def _strictify(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a Pydantic JSON Schema into one Groq's strict mode accepts.

    Strict mode imposes two rules Pydantic does not follow:

      * every key in `properties` must appear in `required` - Pydantic omits
        any field that has a default, which is most of `RoutingDecision`;
      * every object must set `additionalProperties: false`.

    Both are applied recursively, including through `$defs`, because a nested
    model is where this is easiest to miss and the API rejects the whole
    request when one is wrong.

    Returns a new schema; the input is left alone so a cached
    `model_json_schema()` is not corrupted for other callers.
    """

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node

        out = {key: walk(value) for key, value in node.items()}
        # An open map is left exactly as it is. Closing it would forbid the very
        # keys it exists to carry, which is how tool arguments silently became
        # `{}` in production.
        if isinstance(out.get("additionalProperties"), dict):
            return out
        if out.get("type") == "object" or "properties" in out:
            properties = out.get("properties", {})
            out["additionalProperties"] = False
            # Order is not significant to the API, but a stable one keeps the
            # request body diffable when debugging a rejection.
            out["required"] = sorted(properties)
        return out

    return walk(copy.deepcopy(schema))


class GroqProvider:
    name = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key = api_key if api_key is not None else settings.groq_api_key
        if not key:
            raise GroqError(
                "GroqProvider needs an API key. Set ONEMIND_GROQ_API_KEY, "
                "or set ONEMIND_LLM_PROVIDER=ollama to run locally."
            )
        self.model = model or settings.groq_model
        self.base_url = (base_url or settings.groq_base_url).rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=settings.groq_timeout_s,
            headers={"Authorization": f"Bearer {key}"},
            transport=transport,
        )

    # -- internals -----------------------------------------------------------

    def _payload(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        stream: bool,
        response_format: dict | None = None,
    ) -> dict:
        body: dict = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "stream": stream,
            "temperature": temperature,
        }
        # Omitted rather than sent empty: not every Groq model accepts it, and
        # an unsupported parameter is a 400 rather than something ignored.
        if settings.groq_reasoning_format:
            body["reasoning_format"] = settings.groq_reasoning_format
        if response_format is not None:
            body["response_format"] = response_format
        return body

    @staticmethod
    def _retry_delay(resp: httpx.Response, attempt: int) -> float:
        """How long to wait before retrying. The server's answer wins.

        Groq sends `Retry-After` when it knows when the window reopens; guessing
        shorter than that just spends another request being refused.
        """
        header = resp.headers.get("retry-after")
        if header:
            try:
                return float(header)
            except ValueError:
                pass
        return settings.groq_retry_base_delay_s * (2**attempt)

    async def _send(self, body: dict) -> httpx.Response:
        """POST with backoff on the failures that are worth repeating.

        The free tier is 30 requests/minute and both the eval harness and an
        eager demo click cross it. A 429 reaching the orchestrator becomes a 500
        the clinician cannot act on, so it is absorbed here.

        A 4xx that is not 429 is not retried: the request itself is wrong -
        usually a schema strict mode refused - and repeating it only delays the
        message that says so.
        """
        last: httpx.Response | None = None
        for attempt in range(settings.groq_max_retries + 1):
            resp = await self._client.post(f"{self.base_url}/chat/completions", json=body)
            if resp.status_code < 400:
                return resp
            last = resp
            retriable = resp.status_code == 429 or resp.status_code >= 500
            if not retriable or attempt == settings.groq_max_retries:
                break
            await asyncio.sleep(self._retry_delay(resp, attempt))

        assert last is not None
        # Groq puts the reason a schema was refused in the body, and losing it
        # turns a ten-second fix into an afternoon.
        raise GroqError(f"groq returned {last.status_code}: {last.text[:600]}")

    async def _post(self, body: dict) -> dict:
        resp = await self._send(body)
        return resp.json()

    @staticmethod
    def _content(data: dict) -> str:
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise GroqError(f"unexpected groq response shape: {json.dumps(data)[:400]}") from exc

    # -- LLMProvider ---------------------------------------------------------

    async def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> str:
        data = await self._post(self._payload(messages, temperature=temperature, stream=False))
        return self._content(data)

    async def stream(
        self, messages: Sequence[Message], *, temperature: float = 0.0
    ) -> AsyncIterator[str]:
        body = self._payload(messages, temperature=temperature, stream=True)
        async with self._client.stream(
            "POST", f"{self.base_url}/chat/completions", json=body
        ) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                raise GroqError(f"groq returned {resp.status_code}: {resp.text[:600]}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                token = chunk["choices"][0].get("delta", {}).get("content")
                if token:
                    yield token

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float = 0.0,
    ) -> T:
        """Constrained decode where the schema allows it, validated JSON where
        it does not.

        Routing schemas are closed, so they get `strict: true` - a real
        decode-time guarantee, which is what the routing numbers rest on.
        Specialist plans carry an open map of tool arguments, which strict mode
        refuses, so they fall back to schema-guided JSON that is validated here
        instead. That fallback is weaker, and the retry below is what pays for
        the difference.
        """
        raw_schema = schema.model_json_schema()
        strict = not _contains_open_map(raw_schema)
        body = self._payload(
            messages,
            temperature=temperature,
            stream=False,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": strict,
                    # Only rewritten for strict mode, whose restrictions are the
                    # only reason to rewrite it.
                    "schema": _strictify(raw_schema) if strict else raw_schema,
                },
            },
        )

        # Strict mode cannot return the wrong shape, so a failure there is a
        # real fault and retrying only hides it. Without the guarantee, one
        # retry costs a second and saves the specialist's answer.
        attempts = 1 if strict else 2
        last: Exception | None = None
        for _ in range(attempts):
            raw = self._content(await self._post(body))
            try:
                return schema.model_validate_json(raw)
            except Exception as exc:  # noqa: BLE001 - surfaced with the text below
                last = exc
                offending = raw
        raise GroqError(
            f"{self.model} returned output failing {schema.__name__}: {offending[:400]}"
        ) from last

    async def aclose(self) -> None:
        await self._client.aclose()
