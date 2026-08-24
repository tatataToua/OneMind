"""The blackboard: identifiers one specialist established, for another to use.

Specialists do not talk to each other and this module does not let them. They
write nothing here themselves. The orchestrator reads their tool output after
fan-in, extracts the identifiers it recognises, and hands the result to whoever
was blocked for want of exactly that. Clinical still cannot call
`claim_lookup`; Remote Monitoring still cannot read a chart.

That is the blackboard pattern, and the constraint that keeps it honest is that
nothing on the board is a claim. Only an identifier, and where it came from.

## Everything here is in redaction space

`Fact.value` is always a placeholder - `PHI_PATIENT_2`, never `12345`. This is
not decoration and it is the reason the module is safe.

`redact_json` cannot tokenise a structured identifier. Serialised FHIR output
reads `"patient_id": "12345"`, and `_PATIENT_ID` needs a patient-ish word
followed by whitespace; `_id": "` is not whitespace. The value survives
redaction intact - correct for output a specialist quotes back, wrong for a
value about to be written into a second specialist's prompt.

So extraction calls `PHISession.tokenize` first. Three things follow:

  - the model plans over a placeholder, as it does for every other identifier;
  - `rehydrate_args` restores the real value at the tool call, unchanged path;
  - `_is_grounded` needs no modification whatsoever. It already accepts any
    value rehydration alters, so a fact-derived identifier passes and an
    invented one is rejected exactly as before.

An earlier design widened the grounding guard to trust this module. Working in
redaction space instead means the guard never had to be relaxed, and a guard
whose whole worth is that it has not been weakened is worth not weakening.

## One subject at a time

A session discussing patient A at turn two and patient B at turn nine must not
answer turn ten with A's identifiers. `reconcile.py` already calls a confident
statement spanning two patients "the worst output this module could produce"
and guards its join key against it; leaving that open one layer up would be
inconsistent. So resolving a different patient switches the subject and drops
the previous one's facts rather than merging them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from ..guardrails.phi import PHISession

# Which PHI token kind each fact key is minted under. A key absent from this
# map is not extractable - deliberately, since a fact nobody can classify is a
# fact nobody can safely put in a prompt.
_KINDS: dict[str, str] = {
    "patient_id": "PATIENT",
    "mrn": "MRN",
}

# Facts that identify a person. Resolving a different one of these switches the
# subject; everything else is kept independently of who is being discussed.
SUBJECT_KEY = "patient_id"


@dataclass(frozen=True)
class Fact:
    """One established identifier and its provenance.

    `value` is always a redaction placeholder. `source` names the specialist
    and tool that produced it, so a reader can find the retrieval in the trace
    rather than taking the fact on trust.
    """

    key: str
    value: str
    source: str
    turn: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "value": self.value, "source": self.source, "turn": self.turn}


class Facts:
    """Established identifiers for the conversation's current subject."""

    def __init__(self) -> None:
        self._facts: dict[str, Fact] = {}
        self._subject: str = ""

    # -- reading -------------------------------------------------------------

    @property
    def subject(self) -> str:
        """Placeholder token of the patient these facts describe, if any."""
        return self._subject

    def get(self, key: str) -> Fact | None:
        return self._facts.get(key)

    def value(self, key: str) -> str:
        fact = self._facts.get(key)
        return fact.value if fact else ""

    def keys(self) -> set[str]:
        return set(self._facts)

    def all(self) -> list[Fact]:
        return sorted(self._facts.values(), key=lambda f: f.key)

    def __len__(self) -> int:
        return len(self._facts)

    def __bool__(self) -> bool:
        return bool(self._facts)

    def __iter__(self) -> Iterator[Fact]:
        return iter(self.all())

    def __contains__(self, key: object) -> bool:
        return key in self._facts

    # -- writing -------------------------------------------------------------

    def set(self, key: str, value: str, *, source: str, turn: int = 0) -> None:
        """Record a fact, switching subject if this names a different patient.

        Last write wins for a key already held. Re-resolving the same patient
        is not a switch, so a conversation that keeps returning to one chart
        keeps the claim ids it picked up along the way.
        """
        if not key or not value:
            return
        if key == SUBJECT_KEY and self._subject and value != self._subject:
            self._facts.clear()
        if key == SUBJECT_KEY:
            self._subject = value
        self._facts[key] = Fact(key=key, value=value, source=source, turn=turn)

    def as_prompt_block(self) -> str:
        """Render for a specialist's planning prompt.

        Values only - no provenance. A 4B model handed "from
        clinical.fhir_search_patient" will reproduce it in its answer as though
        it were evidence.
        """
        if not self._facts:
            return ""
        lines = "\n".join(f"- {f.key}: {f.value}" for f in self.all())
        return (
            "ESTABLISHED FACTS - identifiers already resolved in this "
            "conversation. Use them when the request does not supply one.\n"
            f"{lines}\n"
        )


# -- extraction ---------------------------------------------------------------

Extractor = Callable[[dict[str, Any]], dict[str, str]]


@dataclass(frozen=True)
class Provider:
    tool: str
    keys: tuple[str, ...]
    fn: Extractor


PROVIDERS: dict[str, Provider] = {}


def provides(tool: str, *, keys: tuple[str, ...]) -> Callable[[Extractor], Extractor]:
    """Declare which fact keys a tool's output can yield.

    On the tool, not on the specialist. "`fhir_search_patient` yields a
    `patient_id`" is a fact about the tool, and `SpecialistSpec.provides`
    derives from it - so the two cannot drift apart.

    Mirrors `@check(...)` in `reconcile.py` and `@tool(...)` in
    `tools/base.py`: declare a capability and its contract in one place.
    """

    def wrap(fn: Extractor) -> Extractor:
        PROVIDERS[tool] = Provider(tool=tool, keys=keys, fn=fn)
        return fn

    return wrap


def keys_for_tools(tool_names: list[str]) -> tuple[str, ...]:
    """Every fact key the given tools can produce, in declaration order."""
    found: list[str] = []
    for name in tool_names:
        provider = PROVIDERS.get(name)
        if provider is None:
            continue
        found.extend(k for k in provider.keys if k not in found)
    return tuple(found)


@provides("fhir_search_patient", keys=("patient_id", "mrn"))
def _from_patient_search(output: dict[str, Any]) -> dict[str, str]:
    """A resolved chart yields the identifiers that scope other planes.

    The `found` guard is load-bearing. This tool returns
    `found: False, ambiguous: True` when several patients share a name, which
    is exactly the case where taking an identifier would attach the wrong
    person to the conversation for the rest of the session.
    """
    if not output.get("found"):
        return {}
    return {key: str(output[key]) for key in ("patient_id", "mrn") if output.get(key)}


@provides("fhir_get_resource", keys=("patient_id",))
def _from_resource_fetch(output: dict[str, Any]) -> dict[str, str]:
    if not output.get("found"):
        return {}
    return {"patient_id": str(output["patient_id"])} if output.get("patient_id") else {}


@provides("claim_lookup", keys=("patient_id",))
def _from_claim_lookup(output: dict[str, Any]) -> dict[str, str]:
    """A claim names its patient, which is how billing unblocks the chart.

    Only when every returned row agrees. A lookup by patient id returns that
    patient's whole ledger and a lookup by claim id returns one row, so
    disagreement means something unexpected - and guessing which row is the
    subject is precisely the error this module exists to avoid.
    """
    if not output.get("found"):
        return {}
    ids = {
        str(row["patient_id"]).strip()
        for row in output.get("claims", [])
        if isinstance(row, dict) and row.get("patient_id")
    }
    if len(ids) != 1:
        return {}
    return {"patient_id": ids.pop()}


def extract(
    tool: str,
    output: Any,
    session: PHISession,
    *,
    source: str,
    turn: int = 0,
) -> list[Fact]:
    """Pull facts out of one redacted tool result, in redaction space.

    A tool with no registered extractor contributes nothing, and an extractor
    that raises contributes nothing - a broken extractor must not sink a
    request the specialists already answered.
    """
    provider = PROVIDERS.get(tool)
    if provider is None or not isinstance(output, dict) or "error" in output:
        return []

    try:
        raw = provider.fn(output)
    except Exception:  # noqa: BLE001 - a bad extractor is not a failed request
        return []

    found: list[Fact] = []
    for key, value in raw.items():
        kind = _KINDS.get(key)
        text = str(value).strip()
        if kind is None or not text:
            continue
        # Already a placeholder (the request supplied it, or an earlier turn
        # did): `tokenize` returns the token unchanged rather than minting a
        # token for a token.
        token = text if session.rehydrate(text) != text else session.tokenize(kind, text)
        found.append(Fact(key=key, value=token, source=source, turn=turn))
    return found


def collect(
    results: Any,
    session: PHISession,
    *,
    turn: int = 0,
    into: Facts | None = None,
) -> Facts:
    """Harvest every fact the given specialist results support.

    Reads `SpecialistResult.tool_calls`, which already holds redacted output
    merged at fan-in - a consumer of existing state rather than a new one.
    """
    facts = into if into is not None else Facts()
    found: list[Fact] = []
    for result in results:
        for call in getattr(result, "tool_calls", []) or []:
            tool = str(call.get("tool", ""))
            found.extend(
                extract(
                    tool,
                    call.get("result"),
                    session,
                    source=f"{getattr(result, 'agent', '?')}.{tool}",
                    turn=turn,
                )
            )

    # Subject first, always. `Facts.set` clears the board when the patient
    # changes, so writing an `mrn` before the `patient_id` that arrived with it
    # would file that mrn under the outgoing subject and then delete it. Within
    # one extractor the order is already right; across two specialists
    # completing in either order it is not, and a fan-in has no fixed order.
    found.sort(key=lambda f: f.key != SUBJECT_KEY)
    for fact in found:
        facts.set(fact.key, fact.value, source=fact.source, turn=fact.turn)
    return facts


__all__ = [
    "Fact",
    "Facts",
    "PROVIDERS",
    "SUBJECT_KEY",
    "collect",
    "extract",
    "keys_for_tools",
    "provides",
]
