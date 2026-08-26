"""Request limits: rate, concurrency, and body size.

## What is actually scarce

Not request handling - the GPU. One `/api/chat/stream` request occupies up to
`max_parallel_agents` inference slots for up to `agent_timeout_s`, so a handful
of concurrent callers is the whole machine. That makes this the highest-severity
gap on the HTTP surface: no credential is needed to exhaust it, and a `for` loop
is sufficient.

So there are two controls, and the second matters more:

    rate         how often a caller may start work      protects against volume
    concurrency  how much work a caller may hold open   protects against depth

Rate limiting alone does not help when each request costs ninety seconds; ten
requests a minute is already more inference than the box can serve. The
concurrency cap is what actually bounds a single caller's share of the GPU:
`max_concurrent_per_client` in-flight requests, each able to occupy up to
`max_parallel_agents` slots, so the two settings want reading together.

## Hand-rolled, and why

A token bucket is about thirty lines. `slowapi` is a dependency, a middleware
stack, and a Redis story for a project that has one process and no database.
This is readable end to end, which matters more here than being general.

## X-Forwarded-For is not trusted

Deliberately. Nothing terminates TLS in front of this, so an
attacker-controlled header would make the limiter opt-in - spoof a fresh value
per request and every bucket is new. Behind a real proxy this needs revisiting
*together with* the proxy's own header rewriting; trusting the header before
that exists is worse than not limiting at all, because it looks limited.

## Not distributed

State is per process. Two uvicorn workers means two buckets and twice the limit.
Correct for one process serving one Ollama instance; a multi-worker deployment
needs shared state, which is a different design and is called out in the README
rather than half-built here.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import Request

from ..config import settings


class LimitExceeded(Exception):
    """A caller asked for more than its share. Carries `retry_after` so the
    handler can tell them when to come back rather than just refusing."""

    def __init__(self, message: str, retry_after: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


@dataclass
class TokenBucket:
    """Capacity tokens, refilled at `refill_per_s`, never above capacity.

    The cap on refill is what stops an idle caller from banking an unbounded
    burst - an account dormant for a day should get one burst on return, not a
    day's worth of requests at once.
    """

    capacity: float
    refill_per_s: float
    tokens: float = field(default=0.0)
    updated: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.tokens = self.capacity

    def take(self, amount: float = 1.0) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.refill_per_s)
        self.updated = now
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def retry_after(self) -> int:
        """Whole seconds until one token is available, at least one."""
        if self.refill_per_s <= 0:
            return 60
        return max(1, int((1.0 - self.tokens) / self.refill_per_s) + 1)


@dataclass
class RateLimiter:
    """One bucket per caller, with a ceiling on how many buckets exist.

    The ceiling is not tidiness. A limiter that allocates per source address is
    itself a memory target, and the attacker chooses the addresses - so the map
    is bounded and the oldest entries are dropped, the same bargain
    `conversation.py` makes with `max_sessions`. Evicting an active caller only
    grants them a fresh bucket, which is the failure mode we can afford; the one
    we cannot is unbounded growth.
    """

    capacity: float
    refill_per_s: float
    max_keys: int = 1024
    buckets: dict[str, TokenBucket] = field(default_factory=dict)

    def allow(self, key: str) -> bool:
        bucket = self.buckets.get(key)
        if bucket is None:
            if len(self.buckets) >= self.max_keys:
                self._evict()
            bucket = self.buckets[key] = TokenBucket(self.capacity, self.refill_per_s)
        else:
            # Re-insert so iteration order tracks recency; dicts preserve
            # insertion order, which is all the LRU this needs.
            self.buckets[key] = self.buckets.pop(key)
        return bucket.take()

    def retry_after(self, key: str) -> int:
        bucket = self.buckets.get(key)
        return bucket.retry_after() if bucket else 1

    def _evict(self) -> None:
        for key in list(self.buckets)[: max(1, self.max_keys // 4)]:
            del self.buckets[key]


@dataclass
class ConcurrencyLimiter:
    """How many requests one caller may have in flight at once.

    Acquire and release are paired by the context manager rather than left to
    callers, because the release that matters most is the one after an
    exception - a streaming generator that raises mid-answer must not leak the
    slot. `test_concurrency_releases_when_the_body_raises` pins that.
    """

    limit: int
    active: dict[str, int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        async with self._lock:
            current = self.active.get(key, 0)
            if current >= self.limit:
                raise LimitExceeded(
                    f"too many requests in flight; at most {self.limit} at a time",
                    retry_after=5,
                )
            self.active[key] = current + 1
        try:
            yield
        finally:
            async with self._lock:
                remaining = self.active.get(key, 1) - 1
                if remaining > 0:
                    self.active[key] = remaining
                else:
                    self.active.pop(key, None)


# --- process-wide instances ------------------------------------------------
# Built lazily so tests can `reset()` between cases and so settings are read at
# first use rather than at import.

_chat_limiter: RateLimiter | None = None
_chat_gate: ConcurrencyLimiter | None = None


def chat_limiter() -> RateLimiter:
    global _chat_limiter
    if _chat_limiter is None:
        _chat_limiter = RateLimiter(
            capacity=float(settings.rate_limit_burst),
            refill_per_s=settings.rate_limit_per_minute / 60.0,
            max_keys=settings.rate_limit_max_keys,
        )
    return _chat_limiter


def chat_gate() -> ConcurrencyLimiter:
    global _chat_gate
    if _chat_gate is None:
        _chat_gate = ConcurrencyLimiter(limit=settings.max_concurrent_per_client)
    return _chat_gate


def reset() -> None:
    """Drop all limiter state. For tests; nothing in the app calls this."""
    global _chat_limiter, _chat_gate
    _chat_limiter = None
    _chat_gate = None


def client_key(request: Request) -> str:
    """Who to charge for this request.

    The peer address, never a forwarded header - see the module docstring.
    `request.client` is None for ASGI transports that do not report a peer, in
    which case everyone shares one bucket, which errs toward limiting.
    """
    return request.client.host if request.client else "unknown"


async def enforce_rate(request: Request) -> str:
    """FastAPI dependency for the expensive routes. Returns the caller key so
    the route can reuse it for the concurrency hold without recomputing it."""
    key = client_key(request)
    limiter = chat_limiter()
    if not limiter.allow(key):
        raise LimitExceeded(
            "rate limit exceeded; this endpoint runs model inference",
            retry_after=limiter.retry_after(key),
        )
    return key
