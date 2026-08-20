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
        "Look up a patient by patient id or MRN and return their demographics, "
        "active conditions, and current medications."
    ),
    parameters=obj_schema(
        {
            "patient_id": {
                "type": "string",
                "description": "Patient identifier or MRN, e.g. 12345 or MRN-448120",
            }
        },
        required=["patient_id"],
    ),
)
def fhir_search_patient(patient_id: str) -> dict[str, Any]:
    patient = store.find_patient(patient_id)
    if patient is None:
        known = [p["patient_id"] for p in store.patients()]
        return {"found": False, "patient_id": patient_id, "known_patient_ids": known}
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
            "patient_id": {"type": "string", "description": "Patient identifier or MRN"},
            "resource": {
                "type": "string",
                "enum": ["labs", "encounters", "medications", "conditions"],
            },
        },
        required=["patient_id", "resource"],
    ),
)
def fhir_get_resource(patient_id: str, resource: str) -> dict[str, Any]:
    patient = store.find_patient(patient_id)
    if patient is None:
        return {"found": False, "patient_id": patient_id}

    payload = patient.get(resource, [])
    if resource == "labs":
        # Newest first, and flag anything above the reference range so the model
        # does not have to do the comparison itself.
        payload = sorted(payload, key=lambda r: r["effective_date"], reverse=True)
        payload = [{**row, "abnormal": row["value"] > row["reference_upper"]} for row in payload]

    return {
        "found": True,
        "patient_id": patient["patient_id"],
        "resource": resource,
        "count": len(payload),
        "items": payload,
    }
