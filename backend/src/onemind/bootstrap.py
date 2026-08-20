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
