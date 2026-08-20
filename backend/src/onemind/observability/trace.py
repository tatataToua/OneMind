"""Execution tracing.

Two jobs. It records what the orchestrator did so the answer can be explained
after the fact, and it emits those events live so the UI can show the system
working rather than a spinner.

Spans carry only redacted text. The audit-log rule in
`fixtures/policies/access-control-and-audit.md` says logs reference records by
identifier and never by content, and the trace is the closest thing this system
has to an audit log.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class SpanKind(StrEnum):
    GUARDRAIL = "guardrail"
    ROUTE = "route"
    AGENT = "agent"
    TOOL = "tool"
    SYNTHESIZE = "synthesize"


class SpanStatus(StrEnum):
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"


@dataclass
class Span:
    span_id: str
    kind: SpanKind
    name: str
    started_at: float
    parent_id: str | None = None
    ended_at: float | None = None
    status: SpanStatus = SpanStatus.RUNNING
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int | None:
        if self.ended_at is None:
            return None
        return int((self.ended_at - self.started_at) * 1000)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["status"] = self.status.value
        data["duration_ms"] = self.duration_ms
        return data


class Trace:
    """Collects spans for one request and publishes them to any listener.

    Parallel agent spans overlap by design - that overlap is the visual proof
    that fan-out is concurrent rather than a queue, so the UI renders spans on a
    timeline using wall-clock offsets rather than as a list.
    """

    def __init__(self, request_id: str | None = None) -> None:
        self.request_id = request_id or uuid.uuid4().hex[:12]
        self.created_at = time.perf_counter()
        self._spans: dict[str, Span] = {}
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    # -- span lifecycle ------------------------------------------------------

    def start(
        self,
        kind: SpanKind,
        name: str,
        *,
        parent_id: str | None = None,
        **detail: Any,
    ) -> str:
        span = Span(
            span_id=uuid.uuid4().hex[:10],
            kind=kind,
            name=name,
            started_at=time.perf_counter(),
            parent_id=parent_id,
            detail=detail,
        )
        self._spans[span.span_id] = span
        self._publish("span_start", span)
        return span.span_id

    def end(
        self,
        span_id: str,
        *,
        status: SpanStatus = SpanStatus.OK,
        **detail: Any,
    ) -> None:
        span = self._spans.get(span_id)
        if span is None or span.ended_at is not None:
            return
        span.ended_at = time.perf_counter()
        span.status = status
        span.detail.update(detail)
        self._publish("span_end", span)

    def fail(self, span_id: str, error: BaseException | str) -> None:
        self.end(span_id, status=SpanStatus.ERROR, error=str(error))

    # -- streaming -----------------------------------------------------------

    def emit(self, event: str, **payload: Any) -> None:
        """Publish a non-span event, such as an answer token."""
        self._queue.put_nowait({"event": event, "data": payload})

    def _publish(self, event: str, span: Span) -> None:
        data = span.to_dict()
        data["offset_ms"] = int((span.started_at - self.created_at) * 1000)
        self._queue.put_nowait({"event": event, "data": data})

    def close(self) -> None:
        self._queue.put_nowait(None)

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        """Consume until `close()`. Convenient, but it ends the stream - callers
        that still need to emit spans afterwards should use `next_event` and
        `pending` instead."""
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item

    async def next_event(self, timeout: float) -> dict[str, Any] | None:
        """Wait briefly for one event. Returns None on timeout or on close."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout)
        except TimeoutError:
            return None

    def pending(self) -> list[dict[str, Any]]:
        """Everything queued right now, without waiting."""
        drained: list[dict[str, Any]] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return drained
            if item is not None:
                drained.append(item)

    # -- inspection ----------------------------------------------------------

    def spans(self) -> list[Span]:
        return sorted(self._spans.values(), key=lambda s: s.started_at)

    def to_dict(self) -> dict[str, Any]:
        base = self.created_at
        out = []
        for span in self.spans():
            data = span.to_dict()
            data["offset_ms"] = int((span.started_at - base) * 1000)
            out.append(data)
        return {"request_id": self.request_id, "spans": out}
