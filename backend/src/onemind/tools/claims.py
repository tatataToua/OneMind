"""Revenue cycle data plane - claims ledger and code sets.

Only the Revenue Cycle specialist holds these tools.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from . import store
from .base import obj_schema, tool, tools


def _classify_denial(claim: dict[str, Any]) -> dict[str, Any]:
    """Attach what a denial code means to the claim that carries it.

    The classification lives with the code set, and the code set belongs to
    this data plane - so the enrichment happens here rather than downstream.
    Two consumers need it. The specialist sees it and stops guessing whether a
    denial is a coding problem; `orchestrator/reconcile.py` reads it out of the
    evidence, which keeps that module reading retrieved records only and never
    reaching into a store of its own.

    An unclassified code adds no keys at all, so a caller can tell "not a
    coding denial" from "nothing is known about this code".
    """
    code = claim.get("denial_code")
    if not code:
        return claim
    for entry in store.codesets().get("denial_codes", []):
        if entry["code"].upper() == str(code).upper() and "coding_related" in entry:
            return {
                **claim,
                "denial_category": entry["category"],
                "denial_coding_related": entry["coding_related"],
            }
    return claim


@tool(
    tools,
    name="claim_lookup",
    description=(
        "Look up claims by claim id, or list all claims for a patient id. "
        "Returns status, amounts, codes, and any denial reason."
    ),
    parameters=obj_schema(
        {
            "claim_id": {"type": "string", "description": "e.g. CLM-8842"},
            "patient_id": {"type": "string", "description": "Patient identifier"},
        }
    ),
)
def claim_lookup(claim_id: str = "", patient_id: str = "") -> dict[str, Any]:
    rows = store.claims()
    if claim_id:
        wanted = claim_id.strip().upper()
        matches = [_classify_denial(c) for c in rows if c["claim_id"].upper() == wanted]
        if not matches:
            # No claim inventory in the miss payload: a lookup that fails must
            # not hand back a sample of the ledger.
            return {
                "found": False,
                "claim_id": claim_id,
                "detail": "no claim matches that id",
            }
        return {"found": True, "count": len(matches), "claims": matches}

    if patient_id:
        wanted = str(patient_id).strip()
        matches = [_classify_denial(c) for c in rows if c["patient_id"] == wanted]
        return {"found": bool(matches), "count": len(matches), "claims": matches}

    return {"found": False, "error": "provide either claim_id or patient_id"}


@tool(
    tools,
    name="validate_code",
    description=(
        "Validate an ICD-10, CPT, or denial code against the active code sets "
        "and return its official description."
    ),
    parameters=obj_schema(
        {"code": {"type": "string", "description": "e.g. E11.9, 99214, or CO-197"}},
        required=["code"],
    ),
)
def validate_code(code: str) -> dict[str, Any]:
    wanted = code.strip().upper()
    for system, entries in store.codesets().items():
        for entry in entries:
            if entry["code"].upper() == wanted:
                return {
                    "valid": True,
                    "code": entry["code"],
                    "system": system,
                    "display": entry["display"],
                }
    return {
        "valid": False,
        "code": code,
        "checked_systems": sorted(store.codesets()),
    }


@tool(
    tools,
    name="denial_summary",
    description=(
        "Aggregate denial statistics across the claims ledger: denial rate, "
        "most frequent denial reasons, and dollars at risk. Optionally scoped "
        "to one payer."
    ),
    parameters=obj_schema(
        {"payer": {"type": "string", "description": "Optional payer name filter"}}
    ),
)
def denial_summary(payer: str = "") -> dict[str, Any]:
    rows = store.claims()
    if payer:
        needle = payer.strip().lower()
        rows = [c for c in rows if needle in c["payer"].lower()]

    if not rows:
        return {
            "count": 0,
            "payer": payer or "all",
            "known_payers": sorted({c["payer"] for c in store.claims()}),
        }

    denied = [c for c in rows if c["status"] == "denied"]
    reasons = Counter(
        f"{c['denial_code']} - {c['denial_reason']}" for c in denied if c["denial_code"]
    )
    return {
        "payer": payer or "all",
        "total_claims": len(rows),
        "denied_claims": len(denied),
        "denial_rate_pct": round(100 * len(denied) / len(rows), 1),
        "billed_at_risk": round(sum(c["billed_amount"] for c in denied), 2),
        "top_denial_reasons": [
            {"reason": reason, "count": count} for reason, count in reasons.most_common(5)
        ],
    }
