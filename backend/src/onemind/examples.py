"""Curated demo prompts.

Every identifier here resolves against `fixtures/` **and carries the data the
prompt asks about** - the second half is the part that is easy to lose. An id
that exists but has no device attached still produces "no monitored device",
which reads on stage as a broken system rather than as an honest miss. So each
prompt below was replayed against the tools and kept only if the answer it
produces is the one the label promises.

The set is chosen to exercise each distinct behaviour once: single-agent
routing for all four specialists, two genuinely cross-agent requests, the
clarifying-question path, and PHI redaction.
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
        "expect": "Routes to Clinical alone. One FHIR lookup: lisinopril and acetaminophen.",
    },
    {
        "label": "Revenue Cycle",
        "prompt": "Why was claim CLM-8909 denied, and what CPT code was billed?",
        "expect": (
            "Routes to Revenue Cycle alone. CO-29, an expired filing deadline - "
            "so the check also says recoding would not fix it. CPT 85025."
        ),
    },
    {
        "label": "Compliance",
        "prompt": "Do we need a BAA with a vendor that only processes de-identified data?",
        "expect": "Routes to Compliance alone. Answer cites the policy section.",
    },
    {
        "label": "Remote Monitoring",
        "prompt": (
            "Has patient 13788's SpO2 dropped below their alert threshold in the last week?"
        ),
        "expect": (
            "Routes to Remote Monitoring. Four readings below the 92% floor, worst "
            "86% - a floor breach, not a ceiling breach."
        ),
    },
    {
        "label": "Cross-agent: clinical + billing",
        "prompt": (
            "Claim CLM-8972 for patient 12345 was denied. Check their diagnosis "
            "history and tell me whether the billed code matches."
        ),
        "expect": (
            "Fans out to Clinical and Revenue Cycle in parallel. The billed E11.9 "
            "is not on the chart, and the reconciler - not the model - computes that."
        ),
    },
    {
        "label": "Cross-agent: policy + telemetry",
        "prompt": (
            "How long must we retain device telemetry, and has patient 13344 "
            "breached their blood pressure threshold?"
        ),
        "expect": (
            "Fans out to Compliance and Remote Monitoring in parallel. Policy "
            "answers the retention half, telemetry the breach half."
        ),
    },
    {
        "label": "Ambiguous",
        "prompt": "check the numbers",
        "expect": "Not actionable. Router asks one clarifying question instead of guessing.",
    },
    {
        "label": "PHI redaction",
        "prompt": "Priya Okafor, MRN-943792, SSN 156-44-5517 - what are they taking?",
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
        "label": "Follow-up (after Two hops)",
        "prompt": "and what are they currently prescribed?",
        "expect": (
            "Only works after a turn that resolved a patient. The subject comes "
            "from the established facts, so no second wave is needed."
        ),
    },
]
