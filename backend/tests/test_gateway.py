"""The gateway that stands between a public tunnel and the laptop's GPU.

The demo shape this exists for is: hosted container -> Cloudflare tunnel ->
gateway -> Ollama on loopback. Everything worth asserting about it is a refusal.
Ollama has no authentication, so if the token check or the route allowlist is
wrong, the failure is not a broken demo - it is an open GPU and a model-deletion
endpoint on the internet.

No network: the upstream is an `httpx.MockTransport` and the token checks never
reach it at all.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from onemind.config import settings
from onemind.llm.base import Message
from onemind.llm.gateway import app
from onemind.llm.ollama import OllamaProvider

TOKEN = "demo-token-value"


@pytest.fixture
def upstream_calls() -> list[httpx.Request]:
    return []


@pytest.fixture
def client(monkeypatch, upstream_calls: list[httpx.Request]) -> TestClient:
    """A running gateway whose upstream Ollama is a mock transport."""
    monkeypatch.setattr(settings, "ollama_auth_token", TOKEN)

    def handler(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(request)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.32.9"})

        # An async iterator rather than bytes: `content=b"..."` builds a
        # response httpx treats as already read, which is not the shape the
        # gateway relays.
        async def ndjson():
            yield b'{"message":{"content":"hello"},"done":false}\n'
            yield b'{"done":true}\n'

        return httpx.Response(
            200,
            content=ndjson(),
            headers={"content-type": "application/x-ndjson"},
        )

    with TestClient(app) as test_client:
        # Replace the client the lifespan opened; the lifespan closes whatever
        # is on `app.state` when the block exits, mock included.
        app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        yield test_client


def _auth(token: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# -- the token is the whole control -----------------------------------------


def test_chat_without_a_token_is_refused(client: TestClient, upstream_calls) -> None:
    assert client.post("/api/chat", json={"model": "x"}).status_code == 401
    assert upstream_calls == [], "an unauthorised request must not reach Ollama"


def test_chat_with_the_wrong_token_is_refused(client: TestClient, upstream_calls) -> None:
    resp = client.post("/api/chat", json={"model": "x"}, headers=_auth("not-it"))
    assert resp.status_code == 401
    assert upstream_calls == []


def test_a_bare_token_without_the_bearer_scheme_is_refused(client: TestClient) -> None:
    resp = client.post("/api/chat", json={"model": "x"}, headers={"Authorization": TOKEN})
    assert resp.status_code == 401


def test_the_gateway_fails_closed_when_no_token_is_configured(client: TestClient, monkeypatch):
    """An unset token must never mean 'let everyone in'."""
    monkeypatch.setattr(settings, "ollama_auth_token", "")
    resp = client.post("/api/chat", json={"model": "x"}, headers=_auth())
    assert resp.status_code == 503


# -- the allowlist ----------------------------------------------------------


@pytest.mark.parametrize("path", ["/api/delete", "/api/pull", "/api/create", "/api/tags"])
def test_model_management_routes_do_not_exist(client: TestClient, path: str) -> None:
    """Ollama serves these; the gateway does not forward them, so a valid
    token buys inference and nothing else."""
    assert client.post(path, json={}, headers=_auth()).status_code == 404
    assert client.get(path, headers=_auth()).status_code == 404


# -- forwarding -------------------------------------------------------------


def test_chat_relays_the_body_unchanged(client: TestClient, upstream_calls) -> None:
    """`format` and `think` are what make routing parse and structured output
    survive. Re-encoding them in the middle would break constrained decoding
    without breaking the request."""
    body = {
        "model": "qwen3.5:4b",
        "messages": [{"role": "user", "content": "hi"}],
        "think": False,
        "format": {"type": "object", "properties": {"agents": {"type": "array"}}},
    }
    resp = client.post("/api/chat", json=body, headers=_auth())

    assert resp.status_code == 200
    assert json.loads(upstream_calls[-1].content) == body


def test_the_demo_token_is_not_forwarded_to_ollama(client: TestClient, upstream_calls) -> None:
    """The secret terminates here. Ollama would ignore it, but a header that
    never leaves the machine cannot leak from one that does."""
    client.post("/api/chat", json={"model": "x"}, headers=_auth())
    assert "authorization" not in upstream_calls[-1].headers


def test_chat_streams_the_ndjson_back(client: TestClient) -> None:
    resp = client.post("/api/chat", json={"model": "x"}, headers=_auth())
    assert resp.status_code == 200
    assert resp.text.count("\n") == 2


def test_version_answers_through_the_gateway(client: TestClient) -> None:
    """`run.ps1 tunnel` calls this end to end, so a broken path is found
    before the demo rather than during it."""
    resp = client.get("/api/version", headers=_auth())
    assert resp.status_code == 200
    assert resp.json()["version"] == "0.32.9"


def test_version_is_also_behind_the_token(client: TestClient) -> None:
    assert client.get("/api/version").status_code == 401


# -- the provider side of the same contract ---------------------------------


async def test_provider_presents_the_token_when_one_is_configured() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"message": {"content": "ok"}})

    provider = OllamaProvider(token=TOKEN, transport=httpx.MockTransport(handler))
    await provider.complete([Message(role="user", content="hi")])
    await provider.aclose()

    assert seen[0].headers["authorization"] == f"Bearer {TOKEN}"


async def test_provider_sends_no_authorization_header_for_a_local_run(monkeypatch) -> None:
    """The default path is loopback to Ollama, which has nothing to
    authenticate to. An empty bearer would be sent on every local call."""
    monkeypatch.setattr(settings, "ollama_auth_token", "")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"message": {"content": "ok"}})

    provider = OllamaProvider(transport=httpx.MockTransport(handler))
    await provider.complete([Message(role="user", content="hi")])
    await provider.aclose()

    assert "authorization" not in seen[0].headers
