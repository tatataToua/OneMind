"""Curated demo prompts.

Every identifier here resolves against `fixtures/`, so the demo never depends on
someone typing a patient id that happens to exist. The set is chosen to exercise
each distinct behaviour once: single-agent routing for all four specialists, two
genuinely cross-agent requests, the clarifying-question path, and PHI redaction.
"""

from __future__ import annotations

from typing import TypedDict


class Example(TypedDict):
    label: str
    prompt: str
    expect: str


EXAMPLES: list[Example] = [
    {
        "label": "Clinical",
        "prompt": "What medications is patient 12345 currently taking?",
        "expect": "Routes to Clinical alone. One FHIR lookup.",
    },
    {
        "label": "Revenue Cycle",
        "prompt": "Why was claim CLM-8849 denied, and what CPT code was billed?",
        "expect": "Routes to Revenue Cycle alone. Claim lookup plus code validation.",
    },
    {
        "label": "Compliance",
        "prompt": "Do we need a BAA with a vendor that only processes de-identified data?",
        "expect": "Routes to Compliance alone. Answer cites the policy section.",
    },
    {
        "label": "Remote Monitoring",
        "prompt": "Has patient 12456 breached their SpO2 alert threshold recently?",
        "expect": "Routes to Remote Monitoring. A floor breach, not a ceiling breach.",
    },
    {
        "label": "Cross-agent: clinical + billing",
        "prompt": (
            "Claim CLM-8849 for patient 12345 was denied. Check their diagnosis "
            "history and tell me whether the billed code matches."
        ),
        "expect": "Fans out to Clinical and Revenue Cycle in parallel, then synthesises.",
    },
    {
        "label": "Cross-agent: policy + telemetry",
        "prompt": (
            "Is our device telemetry retention compliant, and has patient 12345 "
            "breached their blood pressure threshold?"
        ),
        "expect": "Fans out to Compliance and Remote Monitoring in parallel.",
    },
    {
        "label": "Ambiguous",
        "prompt": "check the numbers",
        "expect": "Not actionable. Router asks one clarifying question instead of guessing.",
    },
    {
        "label": "PHI redaction",
        "prompt": ("Samuel Ferreira, MRN-672113, SSN 412-88-2019 - what is he taking?"),
        "expect": "Name, MRN and SSN are replaced before the model sees them.",
    },
    {
        "label": "Two hops",
        "prompt": (
            "Look up Tobias Kaur and tell me whether their blood pressure has been trending high"
        ),
        "expect": (
            "Remote Monitoring blocks - the request names no patient id. Clinical "
            "resolves one, and a second wave runs Remote Monitoring with it."
        ),
    },
    {
        "label": "Follow-up (ask this second)",
        "prompt": "and what are they currently prescribed?",
        "expect": (
            "Only works as a follow-up. The subject comes from the established "
            "facts, so no second wave is needed."
        ),
    },
]
