"""Revenue cycle data plane - claims ledger and code sets.

Only the Revenue Cycle specialist holds these tools.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from . import store
from .base import obj_schema, tool, tools


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
        matches = [c for c in rows if c["claim_id"].upper() == wanted]
        if not matches:
            return {
                "found": False,
                "claim_id": claim_id,
                "known_claim_ids": [c["claim_id"] for c in rows][:10],
            }
        return {"found": True, "count": len(matches), "claims": matches}

    if patient_id:
        wanted = str(patient_id).strip()
        matches = [c for c in rows if c["patient_id"] == wanted]
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
