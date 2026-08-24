"""The four specialists.

Descriptions here are not prose - they are the router's prompt, and they were
tuned against qwen3.5:4b until routing hit 11/11 on the probe set. Changing the
wording changes routing behaviour, so re-run `evals/run_eval.py` after edits.

Each specialist owns one data source nobody else touches, and the four data
sources have deliberately different shapes: documents, coded rows, prose, and
numeric time series.
"""

from __future__ import annotations

from ..orchestrator.registry import SpecialistSpec, registry

CLINICAL = registry.register(
    SpecialistSpec(
        key="clinical",
        display_name="Clinical",
        data_plane="FHIR server (synthetic patient bundles)",
        description=(
            "patient records via FHIR (diagnoses, medications, lab results, "
            "encounters, clinical history)"
        ),
        tool_names=["fhir_search_patient", "fhir_get_resource"],
        exemplars=[
            "What medications is patient 12345 currently taking?",
            "Show me recent lab results",
        ],
    )
)

REVENUE_CYCLE = registry.register(
    SpecialistSpec(
        key="revenue_cycle",
        display_name="Revenue Cycle",
        data_plane="claims ledger + ICD-10/CPT code sets",
        description=(
            "claims, denials, ICD-10/CPT billing codes, reimbursement and denial-rate analysis"
        ),
        tool_names=["claim_lookup", "validate_code", "denial_summary"],
        # A claim id is enough on its own, so this rarely fires. It matters for
        # "which of this patient's claims were denied" when the request names
        # the patient by name and Clinical is the one that resolves the id.
        needs=("patient_id",),
        exemplars=[
            "Why was claim CLM-8842 denied and what CPT code should we have used?",
            "What is our denial rate trend?",
        ],
    )
)

COMPLIANCE = registry.register(
    SpecialistSpec(
        key="compliance",
        display_name="Compliance",
        data_plane="HIPAA / SOC 2 / ISO 27001 policy corpus",
        description=(
            "HIPAA, SOC 2 and ISO 27001 policy and regulatory questions, "
            "including BAAs, de-identification and data retention"
        ),
        tool_names=["policy_search"],
        exemplars=[
            "Do we need a BAA with a vendor that only processes de-identified data?",
            "How long must we retain audit logs?",
        ],
    )
)

REMOTE_MONITORING = registry.register(
    SpecialistSpec(
        key="remote_monitoring",
        display_name="Remote Monitoring",
        data_plane="device telemetry time series",
        description=(
            "device telemetry time series, vitals readings, alert thresholds "
            "and trend detection for remote patient monitoring"
        ),
        tool_names=["telemetry_series", "evaluate_thresholds"],
        # Both tools scope by patient and this plane holds no way to resolve a
        # name, so this is the specialist the second wave exists for.
        needs=("patient_id",),
        exemplars=[
            "Patient's blood pressure cuff has been reading high for three days straight",
            "What is the current alert threshold for SpO2?",
        ],
    )
)

__all__ = [
    "CLINICAL",
    "REVENUE_CYCLE",
    "COMPLIANCE",
    "REMOTE_MONITORING",
    "registry",
]
