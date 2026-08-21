"""PHI boundary tests.

The regression these exist to prevent: an earlier design redacted the user's
request and stopped, so a patient id reached the FHIR tool as `PHI_PATIENT_1`,
the lookup failed, and the specialist reported the patient did not exist. The
boundary is the model, not the data plane.
"""

from __future__ import annotations

from onemind.guardrails.phi import PHIRedactor, PHISession


def test_structured_identifiers_are_replaced(session: PHISession) -> None:
    text = "Patient 12345, SSN 412-88-2019, MRN-672113, call (555) 271-9004"
    out = session.redact(text)

    for secret in ("12345", "412-88-2019", "MRN-672113", "555"):
        assert secret not in out
    assert "PHI_PATIENT_1" in out
    assert "PHI_SSN_1" in out
    assert "PHI_MRN_1" in out


def test_known_names_are_replaced(redactor: PHIRedactor) -> None:
    session = redactor.session()
    out = session.redact("Samuel Ferreira came in Tuesday")
    assert "Samuel" not in out and "Ferreira" not in out
    assert "PHI_NAME" in out


def test_round_trip_restores_original(session: PHISession) -> None:
    original = "Patient 12345 with MRN-672113"
    assert session.rehydrate(session.redact(original)) == original


def test_same_value_reuses_one_token(session: PHISession) -> None:
    out = session.redact("patient 12345 and patient 12345 again")
    assert out.count("PHI_PATIENT_1") == 2
    assert session.count == 1


def test_tokens_are_stable_across_calls(session: PHISession) -> None:
    """A value first seen in a tool result may be referenced in the final
    answer, so the vocabulary must persist for the whole request."""
    first = session.redact("patient 12345")
    second = session.redact("follow-up for patient 12345")
    assert "PHI_PATIENT_1" in first
    assert "PHI_PATIENT_1" in second
    assert session.count == 1


def test_mangled_tokens_still_rehydrate(session: PHISession) -> None:
    """qwen3.5 rewrote PHI_MRN_1 as PHI_MR_N_1 in testing. A mangled token must
    not silently leak into the answer."""
    session.redact("MRN-672113")
    assert session.rehydrate("the record PHI_MR_N_1 shows") == "the record MRN-672113 shows"
    assert session.rehydrate("the record PHI MRN 1 shows") == "the record MRN-672113 shows"


def test_misspelled_token_still_rehydrates(session: PHISession) -> None:
    """Found by evals/edge_cases.py: qwen3.5 wrote PHI_PANTIENT_1 for
    PHI_PATIENT_1 - an inserted letter, not a separator change, so stripping
    separators alone (the fix above) does not catch it. The fuzzy fallback
    matches on exact digit suffix plus closest spelling."""
    session.redact("patient 12345")
    assert session.rehydrate("Patient PHI_PANTIENT_1 is stable") == "Patient 12345 is stable"


def test_misspelled_token_does_not_cross_kinds(session: PHISession) -> None:
    """The fuzzy fallback must not confuse two different placeholders that
    happen to share a trailing digit."""
    session.redact("patient 12345 with MRN-672113")
    assert session.rehydrate("chart PHI_PANTIENT_1, MRN PHI_MRN_1") == (
        "chart 12345, MRN MRN-672113"
    )


def test_dob_matches_us_slash_format(session: PHISession) -> None:
    """Found by evals/edge_cases.py: the DOB pattern only recognised the ISO
    form, so a user-typed "10/22/1979" reached the model untouched."""
    out = session.redact("DOB on file: 1979-10-22, patient says 10/22/1979")
    assert "1979-10-22" not in out
    assert "10/22/1979" not in out


def test_tool_arguments_are_rehydrated(session: PHISession) -> None:
    """The core of the trust boundary: tools receive real lookup keys."""
    session.redact("patient 12345")
    args = session.rehydrate_args({"patient_id": "PHI_PATIENT_1", "resource": "labs"})
    assert args == {"patient_id": "12345", "resource": "labs"}


def test_tool_results_are_redacted_before_the_model_sees_them(
    session: PHISession,
) -> None:
    payload = {"name": "Samuel Ferreira", "mrn": "MRN-672113", "dose": "500 mg"}
    safe = session.redact_json(payload)
    assert "Ferreira" not in str(safe)
    assert "MRN-672113" not in str(safe)
    assert safe["dose"] == "500 mg", "clinical values must survive redaction"


def test_record_identifiers_are_not_redacted(session: PHISession) -> None:
    """Claim, device, and encounter ids identify records, not people. Redacting
    them broke claim lookups for no privacy gain."""
    out = session.redact("Claim CLM-8849 on DEV-2012 during ENC-5803")
    assert "CLM-8849" in out and "DEV-2012" in out and "ENC-5803" in out


def test_clinical_numbers_survive(session: PHISession) -> None:
    """The patient-id pattern must not eat dosages or vitals."""
    text = "500 mg twice daily, 18 units at bedtime, systolic 172 mmHg"
    assert session.redact(text) == text


def test_sessions_do_not_share_vocabulary(redactor: PHIRedactor) -> None:
    a, b = redactor.session(), redactor.session()
    a.redact("patient 12345")
    assert b.rehydrate("PHI_PATIENT_1") == "PHI_PATIENT_1"


def test_rehydrate_is_safe_with_no_mapping(session: PHISession) -> None:
    assert session.rehydrate("nothing to restore") == "nothing to restore"
