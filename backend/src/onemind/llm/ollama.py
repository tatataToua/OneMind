"""Ollama provider - the default, and what runs during the demo.

Talks to Ollama's HTTP API directly rather than going through a LangChain chat
wrapper. Two Ollama-specific controls matter enough to justify owning this code:

  * ``format``  - a JSON Schema enforced during decoding. This is why routing
                  output parses reliably instead of being regex-scraped.
  * ``think``   - qwen3.5 emits chain-of-thought by default, which corrupts
                  structured and tool-call output. It must be off for those
                  paths. See docs/decisions.md.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import TypeVar

import httpx
from pydantic import BaseModel

from ..config import settings
from .base import Message

T = TypeVar("T", bound=BaseModel)


class OllamaError(RuntimeError):
    pass


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        num_ctx: int | None = None,
        token: str | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_model
        self.num_ctx = num_ctx or settings.ollama_num_ctx
        # Ollama itself has no authentication, and a loopback call needs none.
        # The header exists for one shape only: `host` pointing at the tunnelled
        # gateway (`llm/gateway.py`) instead of at 127.0.0.1. Sending it when no
        # token is configured would put an empty bearer on every local request.
        self.token = token if token is not None else settings.ollama_auth_token
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self._client = httpx.AsyncClient(
            timeout=settings.ollama_timeout_s,
            headers=headers,
            transport=transport,
        )

    # -- internals -----------------------------------------------------------

    def _payload(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        stream: bool,
        think: bool,
        fmt: dict | None = None,
    ) -> dict:
        body: dict = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "stream": stream,
            "think": think,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }
        if fmt is not None:
            body["format"] = fmt
        return body

    async def _post(self, body: dict) -> dict:
        resp = await self._client.post(f"{self.host}/api/chat", json=body)
        resp.raise_for_status()
        return resp.json()

    # -- LLMProvider ---------------------------------------------------------

    async def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> str:
        data = await self._post(
            self._payload(messages, temperature=temperature, stream=False, think=False)
        )
        return data["message"]["content"]

    async def stream(
        self, messages: Sequence[Message], *, temperature: float = 0.0
    ) -> AsyncIterator[str]:
        body = self._payload(messages, temperature=temperature, stream=True, think=False)
        async with self._client.stream("POST", f"{self.host}/api/chat", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("done"):
                    break
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float = 0.0,
    ) -> T:
        data = await self._post(
            self._payload(
                messages,
                temperature=temperature,
                stream=False,
                think=False,
                fmt=schema.model_json_schema(),
            )
        )
        raw = data["message"]["content"]
        try:
            return schema.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001 - surfaced with the offending text
            raise OllamaError(
                f"{self.model} returned output failing {schema.__name__}: {raw[:400]}"
            ) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
