"""PHI redaction with a per-request token session.

## Where the trust boundary actually is

The naive design redacts the user's request and stops there. It does not work,
and the way it fails is instructive: redacting "patient 12345" to
`PHI_PATIENT_1` means the FHIR tool is handed a placeholder as a lookup key,
finds nothing, and the specialist confidently reports the patient does not
exist.

The boundary is not "identifiers must never appear anywhere". It is **the model
is untrusted; the data plane is not**. The tools already hold the records - they
are the system of record. Nothing is protected by hiding a patient id from the
FHIR server it came from.

So identifiers cross the boundary in both directions:

    user text ---redact--> [ MODEL ] ---rehydrate--> tool
                                     <---redact----  tool result
    final answer <--rehydrate-- [ MODEL ]

The model plans and writes prose over placeholders and never sees a real
identifier. Tools execute on real values inside the trust boundary. Traces and
logs keep the redacted form, satisfying the audit-log rule in
`fixtures/policies/access-control-and-audit.md`.

## Token stability

Tokens must be stable for the whole request, because a value first seen in a
tool result may be referenced later in the final answer. `PHISession` holds one
mapping across every redact call in a request and reuses the token for a value
it has already seen.

## Token mangling

Language models rewrite tokens. `PHI_MRN_1` came back as `PHI_MR_N_1` in
testing. Re-hydration therefore matches on a normalised form with separators
stripped, so a mangled token still resolves rather than silently leaking a
placeholder into the answer.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

# Ordered most-specific first. A phone pattern will happily eat an SSN, so SSN
# must win; likewise MRN before the bare-digit patient id.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Each digit is allowed at most one space/dot/dash before the next, so the
    # dashed form (541-63-1736) still matches but so does a spelled-out or
    # re-typed one (541 63 1736, 5 4 1 6 3 1 7 3 6). Found live: a spelled-out
    # SSN reached the model untouched because the original pattern only
    # recognised the dashed form.
    ("SSN", re.compile(r"\b\d[ .-]?\d[ .-]?\d[ .-]?\d[ .-]?\d[ .-]?\d[ .-]?\d[ .-]?\d[ .-]?\d\b")),
    ("MRN", re.compile(r"\bMRN-\d{4,10}\b", re.IGNORECASE)),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("PHONE", re.compile(r"\(?\b\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b")),
    # Both ISO (1979-10-22) and US slash/dot (10/22/1979, 10.22.1979) - found
    # live: a date re-typed as "10/22/1979" reached the model untouched
    # because the original pattern only recognised the ISO form.
    ("DOB", re.compile(
        r"\b(?:(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}"
        r"|\d{1,2}[-/.]\d{1,2}[-/.](?:19|20)\d{2})\b"
    )),
]

# Bare 4-6 digit run used as a patient identifier. Applied last and only when
# adjacent to a patient-ish word, so "500 mg" and "20 units" survive intact.
_PATIENT_ID = re.compile(
    r"\b(patient|member|subscriber|pt\.?)\s+(?:id\s*)?#?\s*(\d{4,6})\b",
    re.IGNORECASE,
)

# Any token this module could have produced, however the model re-spaced it.
_TOKEN = re.compile(r"\bPHI[_ ]?[A-Z][A-Z_ ]*?[_ ]?\d+\b")

# Deliberately NOT redacted: claim, device, and encounter ids. They identify
# records, not people, and carry no PHI on their own - 45 CFR 164.514(b) lists
# identifiers "of the individual". Redacting them broke claim lookups for no
# privacy gain. Patient linkage is what makes a record identifiable, and the
# patient identifier itself is redacted.


def _normalise(token: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", token.upper())


def _closest_token(norm: str, known: list[str]) -> str | None:
    """Fuzzy fallback for a token the model misspelled outright, e.g.
    PHI_PANTIENT_1 for PHI_PATIENT_1 - `_normalise` only fixes reformatting
    (spaces, dashes), not a wrong letter. Digits are far less likely to be
    mangled than letters, so only candidates with the exact same trailing
    digit run are considered, then the closest by string similarity wins.
    """
    digits = re.search(r"\d+$", norm)
    if not digits:
        return None
    candidates = [k for k in known if k.endswith(digits.group()) and k != norm]
    best = difflib.get_close_matches(norm, candidates, n=1, cutoff=0.75)
    return best[0] if best else None


class PHISession:
    """One redaction vocabulary for the lifetime of a single request."""

    def __init__(self, known_names: list[str] | None = None) -> None:
        self._names = sorted((n for n in (known_names or []) if n.strip()), key=len, reverse=True)
        self._to_token: dict[str, str] = {}
        self._from_token: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    # -- mapping -------------------------------------------------------------

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self._from_token)

    @property
    def count(self) -> int:
        return len(self._from_token)

    def kinds(self) -> list[str]:
        return sorted({t.split("_")[1] for t in self._from_token if "_" in t})

    def _token_for(self, kind: str, original: str) -> str:
        if original in self._to_token:
            return self._to_token[original]
        self._counts[kind] = self._counts.get(kind, 0) + 1
        token = f"PHI_{kind}_{self._counts[kind]}"
        self._to_token[original] = token
        self._from_token[token] = original
        return token

    # -- directions ----------------------------------------------------------

    def redact(self, text: str) -> str:
        """Replace identifiers with stable tokens. Safe to call repeatedly."""
        if not text:
            return text

        out = text
        for name in self._names:
            out = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE).sub(
                lambda m: self._token_for("NAME", m.group(0)), out
            )
        for kind, pattern in _PATTERNS:
            out = pattern.sub(lambda m, k=kind: self._token_for(k, m.group(0)), out)
        out = _PATIENT_ID.sub(
            lambda m: f"{m.group(1)} {self._token_for('PATIENT', m.group(2))}", out
        )
        return out

    def redact_json(self, payload: Any) -> Any:
        """Redact a tool result before the model is allowed to read it."""
        return json.loads(self.redact(json.dumps(payload, default=str)))

    def rehydrate(self, text: str) -> str:
        """Restore original values, tolerating tokens the model reformatted or
        outright misspelled."""
        if not text or not self._from_token:
            return text

        lookup = {_normalise(t): v for t, v in self._from_token.items()}
        known = list(lookup)

        def restore(match: re.Match[str]) -> str:
            norm = _normalise(match.group(0))
            if norm in lookup:
                return lookup[norm]
            closest = _closest_token(norm, known)
            return lookup[closest] if closest else match.group(0)

        return _TOKEN.sub(restore, text)

    def rehydrate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Turn placeholder tool arguments back into real lookup keys."""
        return {
            key: self.rehydrate(value) if isinstance(value, str) else value
            for key, value in args.items()
        }


class PHIRedactor:
    """Factory for per-request sessions.

    Scope, stated honestly: deterministic pattern matching plus a roster of
    known patient names from the fixture set. Right shape for the trust
    boundary, wrong tool for open-world text - production would put a trained
    NER model (Presidio, Comprehend Medical) behind this same interface.
    """

    def __init__(self, known_names: list[str] | None = None) -> None:
        self.known_names = known_names or []

    def session(self) -> PHISession:
        return PHISession(self.known_names)


def load_known_names(patients: list[dict]) -> list[str]:
    """Roster of names the redactor should recognise, from the patient corpus.

    Real deployments would not enumerate names; see `PHIRedactor`.
    """
    names: set[str] = set()
    for patient in patients:
        full = (patient.get("name") or "").strip()
        if full:
            names.add(full)
            names.update(part for part in full.split() if len(part) > 2)
    return sorted(names)
