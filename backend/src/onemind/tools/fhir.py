"""Clinical data plane - FHIR-shaped patient records.

Only the Clinical specialist holds these tools.
"""

from __future__ import annotations

from typing import Any

from . import store
from .base import obj_schema, tool, tools


@tool(
    tools,
    name="fhir_search_patient",
    description=(
        "Look up a patient by patient id, MRN, or full name and return their "
        "demographics, active conditions, and current medications."
    ),
    parameters=obj_schema(
        {
            "patient_id": {
                "type": "string",
                "description": (
                    "Patient identifier, MRN, or full name - e.g. 12345, "
                    "MRN-448120, or Samuel Ferreira"
                ),
            },
            "birth_date": {
                "type": "string",
                "description": (
                    "Second identifier, YYYY-MM-DD. If the request mentions a "
                    "date of birth, always pass it here on the first attempt - "
                    "names are not unique, and the plan is fixed before any "
                    "result comes back, so there is no chance to add it later."
                ),
            },
        },
        required=["patient_id"],
    ),
)
def fhir_search_patient(patient_id: str, birth_date: str = "") -> dict[str, Any]:
    matches = store.resolve_patients(patient_id, birth_date)

    if len(matches) > 1:
        # Names are not unique. Answering for whichever record sorted first
        # would be a confident answer about the wrong person, so refuse and
        # make the specialist come back for a unique key. The candidates are
        # deliberately not listed - that would be the same disclosure as
        # enumerating the store.
        return {
            "found": False,
            "ambiguous": True,
            "match_count": len(matches),
            "patient_id": patient_id,
            "detail": (
                "several patients match that name; retry with the MRN or "
                "patient id if the request contains one, otherwise ask the "
                "user for a date of birth. Do not guess."
            ),
        }

    if not matches:
        # Deliberately does not enumerate the store. Listing every id we *do*
        # hold turns a miss into a disclosure, and the model will repeat that
        # list straight into the answer.
        return {
            "found": False,
            "patient_id": patient_id,
            "detail": "no patient matches that id, MRN, or name",
        }

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
                "description": "Patient identifier, MRN, or full name",
            },
            "resource": {
                "type": "string",
                "enum": ["labs", "encounters", "medications", "conditions"],
            },
            "birth_date": {
                "type": "string",
                "description": (
                    "Second identifier, YYYY-MM-DD. If the request mentions a "
                    "date of birth, always pass it here on the first attempt - "
                    "names are not unique, and the plan is fixed before any "
                    "result comes back, so there is no chance to add it later."
                ),
            },
        },
        required=["patient_id", "resource"],
    ),
)
def fhir_get_resource(patient_id: str, resource: str, birth_date: str = "") -> dict[str, Any]:
    matches = store.resolve_patients(patient_id, birth_date)
    if len(matches) > 1:
        return {
            "found": False,
            "ambiguous": True,
            "match_count": len(matches),
            "patient_id": patient_id,
            "detail": (
                "several patients match that name; retry with the MRN or "
                "patient id if the request contains one, otherwise ask the "
                "user for a date of birth. Do not guess."
            ),
        }
    if not matches:
        return {
            "found": False,
            "patient_id": patient_id,
            "detail": "no patient matches that id, MRN, or name",
        }

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
