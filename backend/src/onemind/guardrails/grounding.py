"""Answer grounding: does the prose only assert what the evidence contains?

The sibling guard in `agents/base.py` checks a lookup *key* before a tool runs.
This one checks the *claims* after the model has written them, and it exists
because the two failures are independent. A correctly-scoped lookup returns the
right record; the model can still state something about that record which the
record does not say. Observed live: handed a chart with birth date 1979-10-22,
the model wrote 1980-06-15.

## What gets checked, and what deliberately does not

The rule is one sentence: **a value specific enough that the model could not
have derived it by rounding or counting must appear in the evidence.**

That line is where the false positives live. Rewriting is what a language model
is *for*, and most of it is legitimate:

    tool says 318.38   ->  answer says "about $318"      fine, rounded
    tool says 142.3    ->  answer says "just over 142"   fine, rounded
    tool says 21 rows  ->  answer says "three of them"   fine, counted

So bare numbers are not checked at all. A wrong one is a lower-stakes error
than the false-refusal rate needed to catch it. What *is* checked are values
with a canonical form that must survive rewriting intact - clinical and billing
codes, record identifiers, redaction placeholders, and dates. Nobody rounds
`N18.3` to `N18`, and a CPT code is either the one the ledger holds or it is a
different procedure.

Dates are normalised before comparison, because reformatting one is legitimate
("October 22, 1979") while changing one is not.

## Why this is not the critic loop `docs/decisions.md` rejected

A critic asks a second model whether the first model did well, which at 4B is a
coin flip. This asks a string whether it occurs in another string. It is
deterministic, it returns the specific offending values rather than a verdict,
and it cannot hallucinate an objection.
"""

from __future__ import annotations

import re

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# 1979-10-22
_ISO = re.compile(r"\b((?:19|20)\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b")
# 10/22/1979, 10.22.1979, 10-22-1979
_US = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.]((?:19|20)\d{2})\b")
# October 22, 1979 / Oct. 22 1979
_MONTH_FIRST = re.compile(
    r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{2})\b"
)
# 22 October 1979
_DAY_FIRST = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+((?:19|20)\d{2})\b")

# A run of alphanumerics, optionally joined by the separators that appear inside
# codes and identifiers: N18.3, CO-197, CLM-8840, MRN-672113, PHI_PATIENT_1.
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")

# Values the model may legitimately have rewritten rather than quoted.
_BARE_DECIMAL = re.compile(r"^\d+\.\d+$")
_SMALL_INTEGER = re.compile(r"^\d{1,3}$")

# Names of coding systems and standards, in the folded form `_squash` produces.
#
# These have the shape of a code - letters, a separator, digits - but they name
# a *vocabulary*, not a value drawn from a record. "The ICD-10 code is N18.3"
# asserts one fact, not two: `N18.3` is checkable and must be, `ICD-10` is the
# noun it is being called.
#
# The module docstring's claim that separator folding already handles this only
# holds when the evidence happens to mention the field. It does not when the
# specialist's answer is "I cannot see the ICD-10 codes from my data source" -
# which is the honest thing for a specialist on the wrong data plane to say,
# and it was being reported as a fabrication.
_VOCABULARIES = frozenset(
    {
        "icd9",
        "icd10",
        "icd11",
        "cpt4",
        "hcpcs",
        "snomedct",
        "loinc",
        "rxnorm",
        "hl7",
        "fhir4",
        "iso27001",
        "soc2",
    }
)


def _iso(year: str, month: str, day: str) -> str | None:
    try:
        m, d = int(month), int(day)
    except ValueError:
        return None
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return None
    return f"{int(year):04d}-{m:02d}-{d:02d}"


def _named_month(name: str) -> str | None:
    key = name[:3].lower()
    return str(_MONTHS[key]) if key in _MONTHS else None


def dates(text: str) -> list[tuple[str, str]]:
    """Every date in `text`, as (as-written, normalised to YYYY-MM-DD)."""
    found: list[tuple[str, str]] = []

    for match in _ISO.finditer(text):
        iso = _iso(*match.groups())
        if iso:
            found.append((match.group(0), iso))

    for match in _US.finditer(text):
        month, day, year = match.groups()
        iso = _iso(year, month, day)
        if iso:
            found.append((match.group(0), iso))

    for match in _MONTH_FIRST.finditer(text):
        name, day, year = match.groups()
        month = _named_month(name)
        if month and (iso := _iso(year, month, day)):
            found.append((match.group(0), iso))

    for match in _DAY_FIRST.finditer(text):
        day, name, year = match.groups()
        month = _named_month(name)
        if month and (iso := _iso(year, month, day)):
            found.append((match.group(0), iso))

    return found


def _without_dates(text: str) -> str:
    """Dates are compared on their normalised form, so remove them before
    tokenising - otherwise `1979-10-22` is also read as a bare code."""
    out = text
    for pattern in (_ISO, _US, _MONTH_FIRST, _DAY_FIRST):
        out = pattern.sub(" ", out)
    return out


def hard_tokens(text: str) -> list[str]:
    """Codes, identifiers and placeholders - values that must survive rewriting.

    Bare numbers are excluded by design; see the module docstring.
    """
    out: list[str] = []
    for token in _TOKEN.findall(_without_dates(text)):
        if not any(ch.isdigit() for ch in token):
            continue
        if _BARE_DECIMAL.match(token) or _SMALL_INTEGER.match(token):
            continue
        if _squash(token) in _VOCABULARIES:
            continue
        out.append(token)
    return out


def _squash(text: str) -> str:
    """Fold away the separators inside a code, the way `phi.py` folds them
    inside a token.

    Prose and JSON keys punctuate the same value differently: an answer says
    `ICD-10` where the claims ledger has `icd10_code`. Comparing the punctuated
    forms reports a standard's name as a fabrication, which is how this guard
    loses the reader's trust - a flag nobody believes is worse than no flag.
    """
    return re.sub(r"[.\-_]", "", text).casefold()


def ungrounded_values(answer: str, evidence: str, request: str = "") -> list[str]:
    """Values asserted in `answer` that neither `evidence` nor `request` contains.

    Membership is a substring test over separator-folded text rather than a
    token match, deliberately: the permissive direction costs a missed
    fabrication, the strict one costs a refused answer that was correct all
    along.

    All three arguments are expected in redacted space - the specialist's answer
    before re-hydration, the tool results as the model was shown them, and the
    redacted request.

    ## Why the request counts as grounding

    This guard reports values the *model* produced from nowhere. A value the
    user typed is not one of those. A specialist that correctly reports "I have
    no data on CLM-8849" is quoting the question, and flagging that identifier
    tells the reader a true statement was fabricated - which is how a guard
    stops being believed. `_is_grounded` in `agents/base.py` already draws the
    line in the same place, for the same reason.

    The cost is deliberate and worth stating: if the request names a record
    that does not exist and the model then asserts things about it, that
    identifier is no longer flagged. Catching *that* is the job of the tool's
    `found: false` result and the answer prompt, not of this function.

    `request` is optional so existing callers - `evals/`, and any direct use of
    the pure function - keep their current behaviour.

    None of this makes the result PHI-free, and callers must not treat it as
    such. An identifier the inbound patterns failed to catch reaches the model
    intact, and if the model then re-types it, it lands here *because* nothing
    supports it. Redact before logging.
    """
    if not answer.strip() or not evidence.strip():
        return []

    haystack = _squash(evidence + "\n" + request)
    known = {iso for _, iso in dates(evidence)} | {iso for _, iso in dates(request)}
    flagged: list[str] = []

    for written, iso in dates(answer):
        if iso not in known and _squash(written) not in haystack:
            flagged.append(written)

    for token in hard_tokens(answer):
        if _squash(token) not in haystack:
            flagged.append(token)

    return list(dict.fromkeys(flagged))
