"""Does the request agree with itself about who it is asking about?

A request may name a person, or supply a patient identifier, or both. When it
supplies both, nothing downstream compares them, and the reason is structural
rather than an oversight:

  - Remote Monitoring and Revenue Cycle declare `needs=("patient_id",)`. An id
    in the request satisfies that, so neither ever reports itself blocked.
  - Nothing blocked means no second wave, and the router - seeing a question
    about telemetry - has no reason to wake Clinical.
  - Clinical is the only plane that can turn a name into an identifier. The
    telemetry and claims planes hold no names at all and say so.

So no component is ever holding both values, and a component that never holds
two things cannot notice they disagree. Observed live: *"Look up Tobias Kaur
with patient id 13344"* was answered, confidently and with correct numbers,
about patient 13344 - Samuel Ferreira. The name was discarded in silence.

`reconcile.py` calls a confident statement spanning two patients "the worst
output this module could produce" and guards its join key against exactly this.
That guard sits after retrieval and compares two records. This one sits before
dispatch and compares the request against the patient index, because by the
time evidence exists the wrong patient's records have already been read.

## Why this is not a check the model performs

The model never sees either value. Redaction has already replaced the name with
`PHI_NAME_1` and the identifier with `PHI_PATIENT_1` - two opaque tokens with no
stated relationship, which is also why a request naming both sometimes produced
a hedge: asked about a name that appears in no tool result, the model correctly
reported it could not find one.

A safety property that holds only when a 4B model chooses to check it is not a
safety property. So the comparison is a table lookup over the same patient index
the FHIR plane resolves against, and its verdict is a fact about the request
rather than an opinion about it.

## What it reads

The tokens standing in *this request*, resolved through `PHISession.mapping`.
Redaction has already located every name and identifier in order to tokenise
them, so nothing here re-parses the request - a second detector would be free to
disagree with the first about what the request even contains.

Reading the mapping alone would be wrong, and subtly. A session spans the whole
conversation, because a token must keep meaning the same person for as long as
anyone can refer back to it - so by turn three the mapping holds every name and
identifier the conversation has ever mentioned. Asking about Tobias Kaur and
then about patient 12345 is two questions about two patients, which is ordinary.
Scoping to the request is what tells that apart from one question naming two
patients, which is not.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from .phi import PHISession

Verdict = Literal["conflict", "confirmed", "not_applicable"]

# Said to the asker, and deliberately barren. The same line `fhir._unresolved`
# draws: a refusal may say THAT the request is contradictory, never who either
# patient is. Naming the person behind the identifier would disclose a record
# the asker has established no business with - and would do it in the one
# response guaranteed to reach someone who supplied the wrong identifier.
CONFLICT_QUESTION = (
    "The name and the patient identifier in this request belong to two "
    "different patients. Re-send it with just one of them, whichever you meant."
)


@dataclass(frozen=True)
class SubjectCheck:
    """The verdict, and the values it was reached from.

    `named_patient_id` and `supplied_patient_id` are real identifiers, never
    placeholders - this runs beside the data plane, not inside the model's view
    of the world. They exist for the trace and for `confirmed`, which is worth
    stating positively: it is the only thing in the system that establishes that
    a name and an identifier denote one person, and the synthesiser reports it
    as a computed finding rather than leaving the model to wonder why the name
    it was asked about appears in no record.
    """

    verdict: Verdict
    name: str = ""
    named_patient_id: str = ""
    supplied_patient_id: str = ""
    candidates: tuple[str, ...] = field(default_factory=tuple)
    # The placeholders these values were tokenised as. A `confirmed` verdict is
    # stated back to the model as a finding, and the model is on the far side of
    # the trust boundary - so the sentence is built from tokens and rehydrated
    # for the reader, exactly like every other sentence it sees.
    name_token: str = ""
    patient_token: str = ""

    @property
    def question(self) -> str:
        return CONFLICT_QUESTION if self.verdict == "conflict" else ""


def _values(session: PHISession, kind: str, redacted: str) -> list[tuple[str, str]]:
    """(token, original) pairs of one kind standing in `redacted`, in mint order.

    Digits are what can extend a token, so `PHI_PATIENT_1` is required not to
    match inside `PHI_PATIENT_12` once a conversation has minted more than nine
    of them.
    """
    prefix = f"PHI_{kind}_"
    return [
        (token, original)
        for token, original in session.mapping.items()
        if token.startswith(prefix)
        and re.search(rf"(?<![0-9]){re.escape(token)}(?![0-9])", redacted)
    ]


def check_subject(
    session: PHISession,
    resolve: Callable[[str], list[str]],
    redacted: str,
) -> SubjectCheck:
    """Compare the name and the identifier a request supplied.

    `resolve` maps a lookup key - a name, a patient id, an MRN - to the patient
    identifiers it matches. Injected rather than imported so this module reaches
    no store of its own, the same way `PHIRedactor` is handed its roster of names
    instead of loading one.

    `redacted` is this turn's request after redaction, and it scopes the check to
    what was asked now rather than to everything the conversation has mentioned;
    see the module docstring.

    Both sides go through `resolve`, which is why an MRN is covered by the same
    three lines as a patient id: "MRN-861301" and "Tobias Kaur" are both just
    lookup keys, and the question is whether they land on the same person. A
    guard that caught the identifier shape but not the MRN shape would be a
    guard with a hole in it exactly where the demo happens to point.

    Abstains unless the request supplies exactly one name and exactly one
    identifier. Two of either is not a contradiction this can adjudicate, and
    guessing which pair was meant is how a guard starts refusing valid work.
    """
    names = _values(session, "NAME", redacted)
    supplied = _values(session, "PATIENT", redacted) + _values(session, "MRN", redacted)
    if len(names) != 1 or len(supplied) != 1:
        return SubjectCheck(verdict="not_applicable")

    (name_token, name), (patient_token, patient_id) = names[0], supplied[0]
    candidates = [str(c) for c in resolve(name)]
    identified = [str(c) for c in resolve(patient_id)]

    # A name the corpus cannot place is not evidence against the identifier, and
    # neither is an identifier it cannot place. Reporting a conflict from either
    # would refuse a valid request on the strength of a lookup that failed.
    if not candidates or not identified:
        return SubjectCheck(
            verdict="not_applicable",
            name=name,
            supplied_patient_id=patient_id,
            name_token=name_token,
            patient_token=patient_token,
        )

    # Overlap, not equality. A shared name matches several patients, and an
    # identifier that is one of them resolves the ambiguity rather than
    # contradicting it - which is the disambiguation reply `disambiguate.py`
    # already handles, arriving in a single turn.
    agreed = sorted(set(candidates) & set(identified))
    if agreed:
        return SubjectCheck(
            verdict="confirmed",
            name=name,
            named_patient_id=agreed[0],
            supplied_patient_id=agreed[0],
            candidates=tuple(candidates),
            name_token=name_token,
            patient_token=patient_token,
        )

    return SubjectCheck(
        verdict="conflict",
        name=name,
        # Only meaningful when the name resolves to one person. A shared name
        # that matches nobody supplied leaves this empty rather than naming
        # whichever record happened to sort first.
        named_patient_id=candidates[0] if len(candidates) == 1 else "",
        supplied_patient_id=patient_id,
        candidates=tuple(candidates),
        name_token=name_token,
        patient_token=patient_token,
    )


def confirmation_statement(check: SubjectCheck) -> str:
    """The `confirmed` verdict, written for the model rather than the trace.

    Built from placeholders, so it is safe on the model's side of the boundary
    and reads correctly once rehydrated. It exists because the model otherwise
    has no way to know the two tokens denote one person: the name appears in no
    tool result, since neither the telemetry nor the claims plane holds names.
    Left unsaid, that produced answers reporting "no data was found" for a name
    beside correct readings for the identifier.
    """
    return (
        f"{check.name_token} is patient {check.patient_token} - the name and the "
        f"identifier in the request denote the same patient, confirmed against "
        f"the patient index before any data plane was reached."
    )
