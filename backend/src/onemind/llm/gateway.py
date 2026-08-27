"""A narrow, authenticated door in front of Ollama.

Exists for exactly one deployment shape: the hosted build reaching back through
a tunnel to the GPU on the demo laptop, so the deployed link runs the same
`qwen3.5:4b` the eval numbers were measured on and is not spending a hosted
provider's per-minute token budget. See docs/decisions.md.

Ollama has no authentication of its own, and its API is not only inference -
`/api/pull`, `/api/create` and `/api/delete` manage models. Pointing a public
tunnel at 11434 therefore publishes model management, and an hour of somebody
else's free GPU, to whoever finds the hostname.

So the tunnel is pointed here instead. This process forwards two routes and
nothing else, only for callers holding `ONEMIND_OLLAMA_AUTH_TOKEN`, and Ollama stays
bound to loopback where it started. It is a proxy rather than a provider: no
prompt is built here and no response is interpreted, which is what keeps the
provider abstraction in `base.py` the only place that knows how to talk to a
model.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..config import settings


def _upstream() -> str:
    """Ollama's own address.

    Read per call rather than captured at import: the gateway runs on the same
    machine as Ollama, so this is loopback, but a test that repoints
    `ollama_host` should not have to reimport the module.
    """
    return settings.ollama_host.rstrip("/")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # One client for the process. Four specialists dispatch concurrently, and a
    # fresh connection per call would add a handshake to each of them.
    app.state.client = httpx.AsyncClient(timeout=settings.ollama_timeout_s)
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(
    title="OneMind Ollama gateway",
    lifespan=lifespan,
    # Nothing on the public side needs a schema browser, and both routes are
    # documented by this file.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def _authorise(authorization: str | None) -> None:
    expected = settings.ollama_auth_token
    if not expected:
        # Fail closed. An unset token here would otherwise mean "let everyone
        # in", which is the precise failure this process exists to prevent.
        raise HTTPException(503, "gateway has no ONEMIND_OLLAMA_AUTH_TOKEN configured")

    presented = ""
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[len("bearer ") :]
    # Constant-time: the token is a shared secret and this endpoint is public.
    if not secrets.compare_digest(presented, expected):
        raise HTTPException(401, "missing or invalid bearer token")


@app.get("/api/version")
async def version(request: Request, authorization: str | None = Header(default=None)) -> dict:
    """Reachability check. `run.ps1 tunnel` calls this through the tunnel to
    prove the whole path works before the demo rather than during it."""
    _authorise(authorization)
    client: httpx.AsyncClient = request.app.state.client
    try:
        resp = await client.get(f"{_upstream()}/api/version", timeout=10.0)
    except httpx.HTTPError as exc:
        # Distinguishable on purpose: 502 means the tunnel and the token are
        # fine and Ollama is not running, which is a different five seconds of
        # debugging from 401.
        raise HTTPException(502, f"ollama unreachable at {_upstream()}: {exc}") from exc
    resp.raise_for_status()
    return resp.json()


@app.post("/api/chat")
async def chat(request: Request, authorization: str | None = Header(default=None)) -> Response:
    """The only route that runs inference.

    The body is relayed byte for byte, because it carries `format` and `think` -
    the two Ollama controls the provider depends on - and re-encoding a JSON
    Schema in the middle is a way to break constrained decoding invisibly.
    """
    _authorise(authorization)
    client: httpx.AsyncClient = request.app.state.client

    upstream = await client.send(
        client.build_request(
            "POST",
            f"{_upstream()}/api/chat",
            content=await request.body(),
            headers={"content-type": "application/json"},
        ),
        # Streamed either way: `stream: true` is ndjson arriving over the whole
        # generation, and holding it here would turn the synthesis into one
        # silent pause followed by a wall of text.
        stream=True,
    )

    if upstream.status_code >= 400:
        detail = (await upstream.aread()).decode(errors="replace")[:400]
        await upstream.aclose()
        raise HTTPException(upstream.status_code, detail)

    async def relay() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        relay(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/x-ndjson"),
    )


def main() -> None:
    import uvicorn

    if not settings.ollama_auth_token:
        raise SystemExit(
            "Refusing to start without ONEMIND_OLLAMA_AUTH_TOKEN. The gateway is what "
            "stands between a public tunnel and an unauthenticated GPU."
        )
    # Loopback on purpose: the tunnel connects from this machine. Binding
    # 0.0.0.0 would also hand the gateway to everyone on the local network,
    # which is a second exposure nobody asked for.
    uvicorn.run(app, host="127.0.0.1", port=settings.ollama_gateway_port, log_level="info")


if __name__ == "__main__":
    main()
