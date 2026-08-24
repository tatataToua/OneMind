"""Clinical data plane - FHIR-shaped patient records.

Only the Clinical specialist holds these tools.
"""

from __future__ import annotations

from typing import Any

from . import store
from .base import obj_schema, tool, tools


def _unresolved(
    matches: list[dict[str, Any]],
    patient_id: str,
    mrn: str,
    name: str,
) -> dict[str, Any]:
    """The refusal payload when a lookup did not land on exactly one patient.

    How much may be said here is a policy question, not a technical one, and the
    line drawn is: **how many** people match, never **which**. A count is what
    the asker needs to understand why they were refused and what would fix it.
    A list of candidates - names, MRNs, dates of birth - is a disclosure about
    people the asker has established no business with, and enumerating them is
    the same disclosure as browsing the store. A real EHR shows that picker
    behind authentication; this system has none, so it does not.

    `needs` names the fields that would resolve it, so the caller can ask for
    one thing rather than guess.
    """
    if len(matches) > 1:
        return {
            "found": False,
            "ambiguous": True,
            "match_count": len(matches),
            "searched_by": "name",
            "needs": ["mrn", "patient_id", "birth_date"],
            "detail": (
                f"{len(matches)} patients share that name. Supply an MRN, a patient "
                "id, or a date of birth to identify which one. Do not guess, and do "
                "not report any details - none of these patients has been identified."
            ),
        }

    # What was *actually* searched, not what the argument was called.
    # `store.match_patients` routes a misplaced value by shape, so a name passed
    # as `patient_id` was searched as a name - and saying otherwise told the
    # planner to retry the lookup that had just run.
    searched = store.classify_key(patient_id) or ("mrn" if mrn else "name" if name else "")
    if not searched:
        return {
            "found": False,
            "searched_by": "nothing",
            "needs": ["patient_id", "mrn", "name"],
            "detail": "no lookup key was supplied",
        }
    # Still does not enumerate the store: listing every id we *do* hold turns a
    # miss into a disclosure, and the model will repeat that list into an answer.
    return {
        "found": False,
        "searched_by": searched,
        "detail": f"no patient matches that {searched}",
    }


@tool(
    tools,
    name="fhir_search_patient",
    description=(
        "Look up a patient and return their demographics, active conditions, "
        "and current medications. Identify them by patient_id or mrn where "
        "possible; name is a search and may match several people."
    ),
    parameters=obj_schema(
        {
            "patient_id": {
                "type": "string",
                "description": (
                    "Exact patient identifier, digits only - e.g. 12345. Never put a name here."
                ),
            },
            "mrn": {
                "type": "string",
                "description": (
                    "Exact medical record number - e.g. MRN-448120. Prefer this "
                    "or patient_id over a name: they identify one person."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "Full name, only when no id or MRN is available. Names are "
                    "not unique, so this may match several patients and return "
                    "no record at all."
                ),
            },
            "birth_date": {
                "type": "string",
                "description": (
                    "Date of birth, YYYY-MM-DD. Narrows a name that matches "
                    "more than one patient. If the request mentions one, always "
                    "pass it on the first attempt - the plan is fixed before any "
                    "result comes back, so there is no chance to add it later."
                ),
            },
        },
        required=[],
    ),
)
def fhir_search_patient(
    patient_id: str = "", mrn: str = "", name: str = "", birth_date: str = ""
) -> dict[str, Any]:
    matches = store.match_patients(patient_id, mrn, name, birth_date)

    # Names are not unique. Answering for whichever record sorted first would be
    # a confident answer about the wrong person, so anything other than exactly
    # one match is a refusal. See `_unresolved` for what may be said about it.
    if len(matches) != 1:
        return _unresolved(matches, patient_id, mrn, name)

    patient = matches[0]
    return {
        "found": True,
        "patient_id": patient["patient_id"],
        "name": patient["name"],
        "birth_date": patient["birth_date"],
        "gender": patient["gender"],
        "mrn": patient["mrn"],
        "conditions": patient["conditions"],
        "medications": patient["medications"],
    }


@tool(
    tools,
    name="fhir_get_resource",
    description=(
        "Fetch a specific clinical resource for a patient. resource must be one "
        "of: labs, encounters, medications, conditions."
    ),
    parameters=obj_schema(
        {
            "patient_id": {
                "type": "string",
                "description": (
                    "Exact patient identifier, digits only - e.g. 12345. Never put a name here."
                ),
            },
            "mrn": {
                "type": "string",
                "description": (
                    "Exact medical record number - e.g. MRN-448120. Prefer this "
                    "or patient_id over a name: they identify one person."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "Full name, only when no id or MRN is available. Names are "
                    "not unique, so this may match several patients and return "
                    "no record at all."
                ),
            },
            "birth_date": {
                "type": "string",
                "description": (
                    "Date of birth, YYYY-MM-DD. Narrows a name that matches more than one patient."
                ),
            },
            "resource": {
                "type": "string",
                "enum": ["labs", "encounters", "medications", "conditions"],
            },
        },
        required=["resource"],
    ),
)
def fhir_get_resource(
    resource: str,
    patient_id: str = "",
    mrn: str = "",
    name: str = "",
    birth_date: str = "",
) -> dict[str, Any]:
    matches = store.match_patients(patient_id, mrn, name, birth_date)
    if len(matches) != 1:
        return _unresolved(matches, patient_id, mrn, name)

    patient = matches[0]

    payload = patient.get(resource, [])
    if resource == "labs":
        # Newest first, and flag whichever side of the reference threshold is
        # clinically abnormal so the model does not have to do the comparison
        # itself. Direction matters: low eGFR is the bad direction, not high -
        # a single "above threshold" rule would flag healthy kidney function as
        # abnormal and miss the CKD case entirely.
        payload = sorted(payload, key=lambda r: r["effective_date"], reverse=True)
        payload = [
            {
                **row,
                "abnormal": (
                    row["value"] < row["reference_threshold"]
                    if row["reference_direction"] == "below"
                    else row["value"] > row["reference_threshold"]
                ),
            }
            for row in payload
        ]

    return {
        "found": True,
        "patient_id": patient["patient_id"],
        "resource": resource,
        "count": len(payload),
        "items": payload,
    }
