"""Resolving "which of these people did you mean".

Names are not unique. `fhir_search_patient` refuses a name that matches several
patients rather than picking one, and it says how many matched but never which -
a count is what the asker needs to understand the refusal; a list of candidates
is a disclosure about people they have established no business with.

That refusal used to be a full stop. The asker had to retype the entire question
with an MRN attached, and nothing carried over. Session memory makes the obvious
thing possible instead:

    "What medications is Samuel Ferreira taking?"
    -> 2 patients share that name. Supply an MRN, a patient id, or a date of birth.
    "MRN-672113"
    -> [answers the original question]

Two deterministic steps, neither of which is a model call.

`ambiguity` reads the tool results for a refusal that named a count. `resume`
recognises a turn that is nothing but an identifier and rebuilds the held
question around it.

## Why substitution rather than appending

A resumed question replaces the ambiguous name with the identifier that was
supplied - `PHI_NAME_1` becomes `PHI_MRN_2` - instead of carrying both. Carrying
both leaves the planner to decide which key to search on, and the whole point is
that one of them is exact and the other is the thing that failed.

A date of birth is the exception and is appended, because it does not identify
anyone by itself. It narrows the name, which is exactly what
`store.match_patients` uses it for.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

# Placeholders that identify one person. A NAME token is deliberately absent:
# supplying another name in answer to "which of these people" resolves nothing.
_EXACT = re.compile(r"PHI[_ ]?(?:PATIENT|MRN)[_ ]?\d+", re.IGNORECASE)
_DOB = re.compile(r"PHI[_ ]?DOB[_ ]?\d+", re.IGNORECASE)
_NAME = re.compile(r"PHI[_ ]?NAME[_ ]?\d+", re.IGNORECASE)

# Raw forms, for the case where redaction is disabled or a pattern did not fire.
_RAW_MRN = re.compile(r"\bMRN-\d{4,10}\b", re.IGNORECASE)
_RAW_ID = re.compile(r"\b\d{4,6}\b")
_RAW_DOB = re.compile(r"\b(?:19|20)\d{2}-\d{1,2}-\d{1,2}\b")

# A reply may carry a little politeness around the identifier and still be a
# reply. It may not carry a new question - that is a change of subject, not an
# answer to the one that was asked.
_MAX_REPLY_WORDS = 8


def ambiguity(results: Sequence[Any]) -> int:
    """Match count from an unresolved-by-name refusal, or 0.

    Reads the tool output the specialists already retrieved, exactly as
    `facts.py` and `reconcile.py` do - nothing here reaches a store, and nothing
    re-interprets prose. `ambiguous` is a flag the data plane sets; this is only
    the orchestrator noticing it.
    """
    for result in results:
        for call in getattr(result, "tool_calls", []) or []:
            output = call.get("result")
            if isinstance(output, dict) and output.get("ambiguous"):
                try:
                    return int(output.get("match_count", 0))
                except (TypeError, ValueError):
                    return 0
    return 0


def _identifier_in(request: str) -> tuple[str, str]:
    """The single identifier a reply carries, as (kind, token).

    Kind is "exact" for something that names one person, "dob" for something
    that only narrows. Returns ("", "") when the request carries no identifier,
    or more than one - two identifiers is not an answer to "which one", it is a
    new question, and guessing between them is the error this module exists to
    avoid.
    """
    for kind, pattern in (("exact", _EXACT), ("dob", _DOB)):
        found = pattern.findall(request)
        if len(found) == 1:
            return kind, found[0]
        if len(found) > 1:
            return "", ""

    # Unredacted fallbacks, in order of exactness.
    for kind, pattern in (("exact", _RAW_MRN), ("dob", _RAW_DOB), ("exact", _RAW_ID)):
        found = pattern.findall(request)
        if len(found) == 1:
            return kind, found[0]
        if len(found) > 1:
            return "", ""
    return "", ""


def is_reply(request: str) -> bool:
    """True when this turn is an answer to a disambiguation, not a new question.

    Deliberately strict. A turn that carries an identifier *and* a question of
    its own - "what about MRN-672113's labs?" - is a new request that stands on
    its own, and resuming a held question over the top of it would answer
    something nobody asked.
    """
    kind, _ = _identifier_in(request)
    if not kind:
        return False
    return len(request.split()) <= _MAX_REPLY_WORDS and "?" not in request


def resume(question: str, request: str) -> str:
    """Rebuild the held question around the identifier this turn supplied.

    An exact identifier replaces the ambiguous name; a date of birth is appended
    because it narrows rather than identifies. Returns the question unchanged if
    there is nothing usable, so the caller can fall back to normal handling.
    """
    kind, token = _identifier_in(request)
    if not kind:
        return question

    if kind == "dob":
        # Spelled out as a sentence rather than parenthesised. Given
        # "... taking? (date of birth PHI_DOB_1)" the planner treated the date
        # as the lookup key and searched for a patient called `1957-03-18`.
        # Naming the argument is what makes it land in `birth_date`.
        return (
            f"{question} Search by name and pass birth_date={token} "
            f"to identify which patient is meant."
        )

    if _NAME.search(question):
        return _NAME.sub(token, question, count=1)
    # No name token to replace - redaction may be off, or the ambiguity came
    # from somewhere else. Appending still scopes the question.
    return f"{question} ({token})"


__all__ = ["ambiguity", "is_reply", "resume"]
