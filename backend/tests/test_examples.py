"""The demo chips, checked against the fixtures they claim to resolve against.

`examples.py` is the first thing anyone touches, and every prompt in it names a
record by identifier. Fixtures are regenerated from a seed; identifiers move
when they are. What broke before was not an id that had stopped existing - it
was an id that still existed and no longer carried the data its prompt asked
about, which produces a truthful "no monitored device for that patient" and
looks, on stage, exactly like a broken system.

So these tests assert the second half: the record is there *and* so is the
thing the prompt asks it for. They call the tools directly and need no model.
"""

from __future__ import annotations

import re

import pytest

from onemind.examples import EXAMPLES
from onemind.tools.claims import claim_lookup
from onemind.tools.fhir import fhir_search_patient
from onemind.tools.telemetry import evaluate_thresholds

# Every id a prompt mentions must resolve. Pulled out of the prompts themselves
# rather than listed here, so adding a chip extends the coverage automatically.
_PATIENT_ID = re.compile(r"\bpatient (\d{4,6})\b", re.IGNORECASE)
_CLAIM_ID = re.compile(r"\bCLM-\d+\b")
_MRN = re.compile(r"\bMRN-\d+\b")


def _prompts() -> list[str]:
    return [example["prompt"] for example in EXAMPLES]


def test_labels_and_prompts_are_unique() -> None:
    labels = [example["label"] for example in EXAMPLES]
    assert len(set(labels)) == len(labels)
    assert len(set(_prompts())) == len(EXAMPLES)


@pytest.mark.parametrize("prompt", _prompts())
def test_every_patient_id_resolves(prompt: str) -> None:
    for patient_id in _PATIENT_ID.findall(prompt):
        result = fhir_search_patient(patient_id=patient_id)
        assert result["found"], f"{prompt!r} names patient {patient_id}, which does not resolve"


@pytest.mark.parametrize("prompt", _prompts())
def test_every_mrn_resolves(prompt: str) -> None:
    for mrn in _MRN.findall(prompt):
        result = fhir_search_patient(mrn=mrn)
        assert result["found"], f"{prompt!r} names {mrn}, which does not resolve"


@pytest.mark.parametrize("prompt", _prompts())
def test_every_claim_id_resolves(prompt: str) -> None:
    for claim_id in _CLAIM_ID.findall(prompt):
        result = claim_lookup(claim_id=claim_id)
        assert result["found"], f"{prompt!r} names {claim_id}, which does not resolve"


def test_named_patients_are_unambiguous() -> None:
    """A prompt that identifies someone by name only works if the name is
    unique. Two Samuel Ferreiras in the store turn a demo of the two-hop
    dispatch into a demo of the disambiguation refusal - correct behaviour,
    wrong chip."""
    for name in ("Tobias Kaur", "Priya Okafor"):
        result = fhir_search_patient(name=name)
        assert result["found"], f"{name} no longer resolves to exactly one patient"


def test_denied_claims_are_actually_denied() -> None:
    """Both claim chips ask why a claim was *denied*. A paid claim answers the
    question honestly and makes the chip look wrong."""
    for claim_id in ("CLM-8909", "CLM-8972"):
        claim = claim_lookup(claim_id=claim_id)["claims"][0]
        assert claim["status"] == "denied", f"{claim_id} is {claim['status']}, not denied"


def test_the_two_claim_chips_show_opposite_verdicts() -> None:
    """The pair is the point: one denial recoding fixes, one it cannot. If a
    fixture change collapsed them onto the same verdict, both chips would still
    run and the contrast would be gone."""
    filing = claim_lookup(claim_id="CLM-8909")["claims"][0]
    coding = claim_lookup(claim_id="CLM-8972")["claims"][0]
    assert filing["denial_coding_related"] is False
    assert coding["denial_coding_related"] is True


def test_the_cross_plane_chip_still_mismatches() -> None:
    """The clinical + billing chip exists to show the reconciler computing a
    mismatch. A fixture regeneration that put E11.9 on this patient's chart
    would leave the chip running and the demonstration gone."""
    claim = claim_lookup(claim_id="CLM-8972")["claims"][0]
    patient = fhir_search_patient(patient_id=claim["patient_id"])
    on_chart = {condition["code"] for condition in patient["conditions"]}
    assert claim["icd10_code"] not in on_chart
    assert claim["denial_coding_related"] is True


def test_the_telemetry_chips_have_breaching_readings() -> None:
    """`found` is not enough: a monitored patient sitting inside their
    threshold answers "no breach", which is the opposite of what these chips
    are for."""
    for patient_id, metric in (("13788", "spo2"), ("13344", "blood_pressure_systolic")):
        result = evaluate_thresholds(patient_id=patient_id, metric=metric)
        assert result["found"], f"patient {patient_id} has no {metric} device"
        evaluation = result["evaluations"][0]
        assert evaluation["breach_count"] > 0, f"patient {patient_id} is not breaching {metric}"

    # The SpO2 chip's whole point is that a floor breach is not a ceiling breach.
    spo2 = evaluate_thresholds(patient_id="13788", metric="spo2")["evaluations"][0]
    assert spo2["alert_direction"] == "below"


def test_the_two_hop_chip_needs_a_second_wave() -> None:
    """The chip demonstrates one specialist unblocking another. That only
    happens if the telemetry plane genuinely cannot resolve the name itself."""
    blocked = evaluate_thresholds(patient_id="Tobias Kaur", metric="blood_pressure_systolic")
    assert blocked["invalid_key"] is True

    resolved = fhir_search_patient(name="Tobias Kaur")
    unblocked = evaluate_thresholds(
        patient_id=resolved["patient_id"], metric="blood_pressure_systolic"
    )
    assert unblocked["found"] and unblocked["evaluations"][0]["breach_count"] > 0
