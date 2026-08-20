"""HTTP surface.

Thin by design: it wires requests to `Orchestrator` and translates its event
stream to server-sent events. No orchestration logic lives here.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..bootstrap import default_orchestrator
from ..config import settings
from ..examples import EXAMPLES
from ..observability.trace import Trace
from ..orchestrator.registry import registry

app = FastAPI(
    title="OneMind",
    version="0.1.0",
    description=(
        "Healthcare multi-agent orchestrator. Routes a request to the "
        "specialists that own the relevant data, runs them in parallel, and "
        "synthesises one cited answer."
    ),
)

# The Vite dev server runs on a different origin during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "model": settings.ollama_model,
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
async def chat(request: ChatRequest) -> dict[str, Any]:
    """Non-streaming. Used by the eval harness and by tests."""
    outcome = await default_orchestrator().run(request.message)
    return outcome


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        trace = Trace()
        try:
            async for event in default_orchestrator().stream(request.message, trace):
                yield _sse(event["event"], event["data"])
        except Exception as exc:  # noqa: BLE001 - the client needs to hear about it
            yield _sse("error", {"message": str(exc)})

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


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
