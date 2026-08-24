"""Composition root.

One place that knows how the pieces fit together, shared by the API, the eval
harness, and the tests. Nothing below this module constructs its own provider.
"""

from __future__ import annotations

from functools import lru_cache

from .agents import build_specialists
from .config import settings
from .guardrails.phi import PHIRedactor, load_known_names
from .llm.base import LLMProvider
from .orchestrator.conversation import ConversationStore
from .orchestrator.graph import Orchestrator
from .orchestrator.registry import registry
from .tools import store


def build_provider(name: str | None = None) -> LLMProvider:
    """Select a provider by name. The only place provider choice is decided."""
    choice = (name or settings.llm_provider).lower()

    if choice == "ollama":
        from .llm.ollama import OllamaProvider

        return OllamaProvider()

    if choice == "bedrock":
        from .llm.bedrock import BedrockProvider

        return BedrockProvider()

    raise ValueError(f"unknown provider {choice!r}; expected 'ollama' or 'bedrock'")


def build_redactor() -> PHIRedactor:
    return PHIRedactor(known_names=load_known_names(store.patients()))


def build_orchestrator(provider: LLMProvider | None = None) -> Orchestrator:
    provider = provider or build_provider()
    return Orchestrator(
        provider=provider,
        specialists=build_specialists(provider),
        redactor=build_redactor(),
        roster=registry,
    )


@lru_cache(maxsize=1)
def default_orchestrator() -> Orchestrator:
    """Process-wide instance for the API. Stateless per request."""
    return build_orchestrator()


@lru_cache(maxsize=1)
def default_conversations() -> ConversationStore:
    """Process-wide session memory for the API.

    The only stateful thing in the system, and deliberately the only one. It
    holds redaction vocabularies, so it is in-process, capped, and evicted when
    idle - never written to disk. See `orchestrator/conversation.py`.

    Separate from `default_orchestrator` because the orchestrator remains
    stateless: a conversation is passed in per request rather than looked up,
    which is what keeps the eval harness and the tests able to call it with no
    session at all. It shares that orchestrator's redactor so there is one
    known-name list in the process rather than two that could diverge.
    """
    return ConversationStore(default_orchestrator().redactor)
