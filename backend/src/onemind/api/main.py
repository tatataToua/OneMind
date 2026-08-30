"""HTTP surface.

Thin by design: it wires requests to `Orchestrator` and translates its event
stream to server-sent events. No orchestration logic lives here.

What it does own is the boundary with an untrusted caller - request limits, what
an error is allowed to say, and what shape a session id may take. Those live
here because they are properties of the HTTP surface, not of orchestration; see
`api/limits.py` for the reasoning behind each limit.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from ..bootstrap import default_conversations, default_orchestrator
from ..config import settings
from ..examples import EXAMPLES
from ..llm.base import live_identity
from ..observability.trace import Trace
from ..orchestrator.registry import registry
from . import limits

log = logging.getLogger("onemind.api")

app = FastAPI(
    title="OneMind",
    version="0.1.0",
    description=(
        "Healthcare multi-agent orchestrator. Routes a request to the "
        "specialists that own the relevant data, runs them in parallel, and "
        "synthesises one cited answer."
    ),
)

# The Vite dev server runs on a different origin during development. Two exact
# origins, never a wildcard: this API answers with PHI-bearing content, and a
# permissive CORS policy would let any page a clinician has open read it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def guard_request(request: Request, call_next: Any) -> Any:
    """Body ceiling on the way in, hardening headers on the way out.

    The size check reads `Content-Length` rather than draining the body,
    because the point is to refuse a large payload *before* it is buffered and
    parsed. A chunked request without the header is not refused here; the
    schema's own length caps still apply, and no client of this API sends one.
    """
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > settings.max_request_bytes:
        return _error_response(
            status=413,
            code="payload_too_large",
            message=f"request body exceeds {settings.max_request_bytes} bytes",
        )

    response = await call_next(request)
    # This API serves JSON and an event stream to a first-party UI. Nothing it
    # returns should ever be sniffed as another type, framed, or leaked as a
    # referrer to a third party.
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    return response


@app.exception_handler(limits.LimitExceeded)
async def limit_exceeded(request: Request, exc: limits.LimitExceeded) -> JSONResponse:
    response = _error_response(status=429, code="rate_limited", message=exc.message)
    response.headers["Retry-After"] = str(exc.retry_after)
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """The client learns that it failed and how to ask about it. Nothing else.

    `str(exc)` used to go straight back over SSE, which meant a missing fixture
    answered with an absolute filesystem path. The exception goes to the log
    against a correlation id; the caller gets the id.
    """
    request_id = _log_failure(exc, request.url.path)
    return _error_response(
        status=500,
        code="internal_error",
        message="the request could not be completed",
        request_id=request_id,
    )


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    # Omitted on the first turn. The server mints the id and returns it on the
    # `done` event; the client echoes it back. Never client-chosen: a guessable
    # id would hand someone else's redaction vocabulary to whoever asked for it.
    session_id: str | None = Field(default=None, max_length=64)

    @field_validator("session_id")
    @classmethod
    def _must_be_a_minted_id(cls, value: str | None) -> str | None:
        """Reject anything that is not a UUID this server could have minted.

        With no auth, `session_id` is the only thing separating a caller from
        another conversation's PHI vocabulary - it is a bearer credential, so
        it gets a credential's validation. Checking the shape at the schema
        boundary also means an arbitrary caller-chosen string never becomes a
        key in the session map.
        """
        if value is None:
            return None
        try:
            uuid.UUID(value)
        except ValueError:
            raise ValueError("session_id must be a uuid issued by this server") from None
        return value


def resolve_active_model() -> str:
    """Name the model actually in use, whichever provider is selected.

    Health is the first thing checked after a deploy. Reporting the Ollama
    model from a container that talks to Groq sends you debugging the wrong
    half of the system, so this follows `llm_provider` rather than assuming.
    """
    return {
        "ollama": settings.ollama_model,
        "groq": settings.groq_model,
        "bedrock": settings.bedrock_model_id,
    }.get(settings.llm_provider.lower(), "unknown")


def resolve_live_provider() -> tuple[str, str]:
    """Provider and model as of the last call, not as of the configuration.

    With a fallback configured (`llm/fallback.py`) the answer is not a setting.
    It depends on whether the primary was reachable, which is the thing health
    is being asked. So the running provider is asked directly - but only if one
    has already been built. Health must never be what constructs the
    orchestrator: it is the endpoint you reach for when something is wrong, and
    it should answer then too. Before the first turn it reports the configured
    primary, which is what the next call will try.
    """
    provider = _provider_already_built()
    if provider is not None:
        return live_identity(provider)
    return settings.llm_provider, resolve_active_model()


def _provider_already_built() -> object | None:
    """The provider in use, or None if nothing has built one yet.

    Deliberately never constructs. `default_orchestrator` is memoised, so its
    cache size answers "has a turn happened" without causing one; tests replace
    it with a plain stub, which has no cache and no provider, and falls through
    to the configured answer.
    """
    cache_info = getattr(default_orchestrator, "cache_info", None)
    if cache_info is not None and not cache_info().currsize:
        return None
    return getattr(default_orchestrator(), "provider", None)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    provider, model = resolve_live_provider()
    return {
        "status": "ok",
        "provider": provider,
        "model": model,
        # Named so the answer to "why is this on Groq?" is visible rather than
        # inferred. Null when this deployment has only one provider.
        "fallback": settings.llm_fallback or None,
        "agents": registry.keys(),
        "phi_redaction": settings.phi_redaction_enabled,
    }


@app.get("/api/agents")
async def agents() -> dict[str, Any]:
    """The live roster. The UI renders its agent rail from this, so a newly
    registered specialist appears without a frontend change."""
    return {
        "agents": [
            {
                "key": spec.key,
                "display_name": spec.display_name,
                "data_plane": spec.data_plane,
                "description": spec.description,
                "tools": spec.tool_names,
            }
            for spec in registry.all()
        ]
    }


@app.get("/api/examples")
async def examples() -> dict[str, Any]:
    return {"examples": EXAMPLES}


@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    caller: str = Depends(limits.enforce_rate),
) -> dict[str, Any]:
    """Non-streaming. Used by the eval harness and by tests.

    A request without `session_id` is standalone: no memory is created for it,
    so the eval harness keeps getting one cold request per row.
    """
    async with limits.chat_gate().hold(caller):
        conversation = (
            default_conversations().get(request.session_id) if request.session_id else None
        )
        return await default_orchestrator().run(request.message, conversation=conversation)


@app.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    caller: str = Depends(limits.enforce_rate),
) -> StreamingResponse:
    # Resolved outside the generator so an unknown or expired id becomes a new
    # conversation before the first byte, not an error mid-stream.
    conversation = default_conversations().get(request.session_id)

    async def events() -> AsyncIterator[str]:
        # The hold is inside the generator because that is where the work
        # happens: a streaming response returns as soon as the headers are
        # written, so acquiring outside would release the slot while the model
        # was still running. The `async with` releases on the error path too.
        try:
            async with limits.chat_gate().hold(caller):
                trace = Trace()
                async for event in default_orchestrator().stream(
                    request.message, trace, conversation
                ):
                    yield _sse(event["event"], event["data"])
        except limits.LimitExceeded as exc:
            # The stream has already begun, so this cannot become a 429. Say so
            # in-band instead, with the same wording the status code carries.
            yield _sse("error", {"message": exc.message, "retry_after": exc.retry_after})
        except Exception as exc:  # noqa: BLE001 - the client needs to hear that it failed
            request_id = _log_failure(exc, "/api/chat/stream")
            yield _sse(
                "error",
                {"message": "the request could not be completed", "request_id": request_id},
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this, nginx and friends buffer the stream and the live
            # trace arrives all at once at the end, which defeats the point.
            "X-Accel-Buffering": "no",
        },
    )


def _log_failure(exc: Exception, path: str) -> str:
    """Record an exception server-side and return the id the client is told.

    One place, so the streaming and non-streaming paths cannot drift into
    telling the caller different amounts about a failure.
    """
    request_id = uuid.uuid4().hex[:12]
    log.exception("request %s failed at %s: %s", request_id, path, type(exc).__name__)
    return request_id


def _error_response(
    *,
    status: int,
    code: str,
    message: str,
    request_id: str | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {"code": code, "message": message}
    if request_id:
        body["request_id"] = request_id
    return JSONResponse(status_code=status, content={"error": body})


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# --- the built frontend ------------------------------------------------------
#
# In development the UI is served by Vite on another origin and reaches this API
# through its proxy. In a container there is no Vite, so the API serves the build
# itself and the whole app becomes one origin - which is also why the CORS policy
# above stops mattering in production rather than needing to be widened for it.
#
# Mounted last on purpose: the mount claims "/", and Starlette matches routes in
# registration order, so every /api/* route above must already be registered.


def _frontend_directory(path: str) -> Path | None:
    """Resolve a servable build directory, or None to skip mounting.

    An unset path means local development. A path that exists but holds no
    `index.html` means the frontend build failed and left an empty directory;
    mounting that would serve 404s that read like a routing bug, so it is
    refused here where the cause is still obvious.
    """
    if not path:
        return None
    directory = Path(path)
    return directory if (directory / "index.html").is_file() else None


_frontend = _frontend_directory(settings.static_dir)
if _frontend is not None:
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")
