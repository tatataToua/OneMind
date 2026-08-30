"""Composition root.

One place that knows how the pieces fit together, shared by the API, the eval
harness, and the tests. Nothing below this module constructs its own provider.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from .agents import build_specialists
from .config import settings
from .guardrails.phi import PHIRedactor, load_known_names
from .llm.base import LLMProvider
from .orchestrator.conversation import ConversationStore
from .orchestrator.graph import Orchestrator
from .orchestrator.registry import registry
from .tools import store

log = logging.getLogger(__name__)


def build_one_provider(name: str) -> LLMProvider:
    """Construct a single named provider. Knows every backend; chooses none."""
    choice = name.lower()

    if choice == "ollama":
        from .llm.ollama import OllamaProvider

        return OllamaProvider()

    if choice == "groq":
        from .llm.groq import GroqProvider

        return GroqProvider()

    if choice == "bedrock":
        from .llm.bedrock import BedrockProvider

        return BedrockProvider()

    raise ValueError(f"unknown provider {choice!r}; expected 'ollama', 'groq' or 'bedrock'")


def build_provider(name: str | None = None, fallback: str | None = None) -> LLMProvider:
    """Select a provider. The only place provider choice is decided.

    With `llm_fallback` set the result is a `FallbackProvider` rather than one
    backend, so the hosted build can prefer the demo laptop's GPU and still
    answer when that laptop is closed. Nothing above `llm/base.py` can tell the
    difference; both satisfy the same three-method protocol.
    """
    primary_name = (name or settings.llm_provider).lower()
    primary = build_one_provider(primary_name)

    secondary_name = (fallback if fallback is not None else settings.llm_fallback).lower()
    if not secondary_name or secondary_name == primary_name:
        return primary

    try:
        secondary = build_one_provider(secondary_name)
    except Exception as exc:  # noqa: BLE001 - a missing key must not be fatal
        # A fallback that cannot be constructed - usually `groq` with no key -
        # is not an error. It means this deployment has one provider, which is
        # the ordinary local case. Loud enough to find, quiet enough to run.
        log.warning(
            "no %r fallback available (%s); using %r alone", secondary_name, exc, primary_name
        )
        return primary

    from .llm.fallback import FallbackProvider

    return FallbackProvider(primary, secondary)


def build_redactor() -> PHIRedactor:
    return PHIRedactor(known_names=load_known_names(store.patients()))


def resolve_patient_name(name: str) -> list[str]:
    """Identifiers a patient name matches, for the subject identity check.

    Lives here rather than in `guardrails/identity.py` for the same reason
    `build_redactor` loads the name roster here: the composition root is what
    knows about stores. A list, because names are not unique - two patients
    genuinely share one in this corpus, and the check treats membership rather
    than equality as agreement.
    """
    return [patient["patient_id"] for patient in store.find_patients(name)]


def build_orchestrator(provider: LLMProvider | None = None) -> Orchestrator:
    provider = provider or build_provider()
    return Orchestrator(
        provider=provider,
        specialists=build_specialists(provider),
        redactor=build_redactor(),
        roster=registry,
        resolve_name=resolve_patient_name,
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
