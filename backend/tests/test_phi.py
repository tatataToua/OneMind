"""PHI boundary tests.

The regression these exist to prevent: an earlier design redacted the user's
request and stopped, so a patient id reached the FHIR tool as `PHI_PATIENT_1`,
the lookup failed, and the specialist reported the patient did not exist. The
boundary is the model, not the data plane.
"""

from __future__ import annotations

import pytest

from onemind.guardrails.phi import PHIRedactor, PHISession


def test_structured_identifiers_are_replaced(session: PHISession) -> None:
    text = "Patient 12345, SSN 412-88-2019, MRN-672113, call (555) 271-9004"
    out = session.redact(text)

    for secret in ("12345", "412-88-2019", "MRN-672113", "555"):
        assert secret not in out
    assert "PHI_PATIENT_1" in out
    assert "PHI_SSN_1" in out
    assert "PHI_MRN_1" in out


@pytest.mark.parametrize(
    "phrasing",
    [
        "patient 12678",
        "patient id 12678",
        "patient id of 12678",
        "patient ID: 12678",
        "patient #12678",
        "patient number 12678",
        "patient no. 12678",
    ],
)
def test_a_patient_id_is_redacted_however_it_is_introduced(
    session: PHISession, phrasing: str
) -> None:
    """Observed live: "Look up Tobias Kaur with patient id of 12678 ..." reached
    the model with 12678 intact - the pattern recognised "patient id 12678" but
    not the same thing with "of" or a colon between the label and the digits. A
    leaked identifier also makes the subject-identity check abstain, because that
    check reads the tokens redaction minted, so the hedge about a name that
    "could not be linked" came back too."""
    out = session.redact(f"Look up their chart, {phrasing}, and check the trend")
    assert "12678" not in out
    assert "PHI_PATIENT_1" in out


@pytest.mark.parametrize(
    "text",
    [
        "the patient took 500 mg",
        "give the patient 20 units",
        "patient now weighs 12345 grams",
    ],
)
def test_a_number_near_patient_that_is_not_an_id_survives(session: PHISession, text: str) -> None:
    """The widened label match still needs the digits to sit where an id would,
    not merely somewhere after the word "patient"."""
    assert session.redact(text) == text


@pytest.mark.parametrize(
    "written",
    [
        "541631736",
        "541-63-1736",
        "541 63 1736",
        "541.63.1736",
        "541-63 1736",
        "5 4 1 6 3 1 7 3 6",
        "5-4-1-6-3-1-7-3-6",
    ],
)
def test_every_written_ssn_form_is_redacted(session: PHISession, written: str) -> None:
    """Only the dashed form was ever pinned, which is how the spelled-out form
    in `evals/datasets/phi_leak.jsonl` (leak-09) reached the model untouched.
    Each grouping a person or a model actually types is now a case."""
    assert written not in session.redact(f"confirm this SSN is on file: {written}")


@pytest.mark.parametrize(
    "text",
    [
        "claim CLM-123456789 was denied",
        "device DEV-123456789 reading high",
        "during encounter ENC-123456789",
        "lives at zip 90210-1234",
    ],
)
def test_a_nine_digit_run_that_is_not_an_ssn_survives(session: PHISession, text: str) -> None:
    """The untested direction, and where the pattern was wrong.

    Matching nine digits with an optional separator at each gap accepted every
    partition of the run, so a zip+4 (5-4) and the numeric tail of any
    nine-digit record id both read as an SSN. Record ids surviving untouched is
    what decisions.md #4 rests on - redacting one breaks the lookup it scopes,
    for no privacy gain.
    """
    assert session.redact(text) == text


def test_a_nine_digit_mrn_is_redacted_as_an_mrn(session: PHISession) -> None:
    """SSN is tried before MRN, so a nine-digit MRN used to be consumed by the
    SSN pattern and filed under the wrong kind in the audit trace."""
    out = session.redact("chart MRN-123456789 pulled")
    assert "123456789" not in out
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
