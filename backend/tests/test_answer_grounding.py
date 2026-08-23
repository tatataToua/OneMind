"""Answer grounding: claims the tool results do not support.

Two layers. The pure functions in `guardrails/grounding.py` are tested directly,
because the whole design lives in what they choose *not* to flag - the
false-positive rate is the thing that decides whether this guard is usable.
Then the end-to-end path, proving a fabricated value survives to the caller
labelled rather than silent.
"""

from __future__ import annotations

import json

import pytest

from onemind.guardrails.grounding import dates, hard_tokens, ungrounded_values
from onemind.observability.trace import SpanKind, Trace

from .conftest import StubProvider

# The record behind patient 12345, as the model is shown it - dates and names
# already replaced by the redactor.
EVIDENCE = """
[{"tool": "fhir_search_patient", "result": {"found": true,
  "patient_id": "PHI_PATIENT_1", "name": "PHI_NAME_1", "birth_date": "PHI_DOB_1",
  "mrn": "PHI_MRN_1", "conditions": [{"code": "N18.3",
  "display": "Chronic kidney disease, stage 3"}],
  "medications": [{"display": "Metformin", "dose": "500 mg"}]}}]
"""

CLAIM_EVIDENCE = """
[{"tool": "claim_lookup", "result": {"claim_id": "CLM-8840",
  "date_of_service": "2026-04-29", "cpt_code": "95250", "icd10_code": "I10",
  "billed_amount": 318.38, "denial_code": "CO-197"}}]
"""


# -- date normalisation ------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "1979-10-22",
        "10/22/1979",
        "10.22.1979",
        "October 22, 1979",
        "Oct. 22 1979",
        "22 October 1979",
    ],
)
def test_every_written_date_form_normalises_to_one_value(text: str) -> None:
    """Reformatting a date is legitimate; changing it is not. Comparison
    therefore happens on the normalised form."""
    assert [iso for _, iso in dates(text)] == ["1979-10-22"]


def test_an_impossible_date_is_not_read_as_a_date() -> None:
    assert dates("code 99/99/1979") == []


# -- what is deliberately not flagged ----------------------------------------


def test_rounding_a_figure_is_not_a_fabrication() -> None:
    """The model is expected to write prose, not to quote JSON."""
    assert ungrounded_values("The claim was billed at about $318.", CLAIM_EVIDENCE) == []


def test_counting_is_not_a_fabrication() -> None:
    assert ungrounded_values("There are 2 active conditions listed.", EVIDENCE) == []


def test_a_dose_restated_in_prose_is_not_a_fabrication() -> None:
    assert ungrounded_values("Metformin, 500 mg daily.", EVIDENCE) == []


def test_bare_numbers_are_never_checked() -> None:
    assert hard_tokens("500 mg over 14 days, mean 142.3, 3 breaches") == []


def test_a_standard_named_in_prose_is_not_a_fabrication() -> None:
    """Found live: the model wrote "ICD-10" where the ledger field is spelled
    `icd10_code`, and the guard reported a standard's name as invented. Prose
    and JSON keys punctuate the same value differently, so comparison folds
    separators away first."""
    answer = "The ICD-10 code on the claim is I10, billed under CPT 95250."
    assert ungrounded_values(answer, CLAIM_EVIDENCE) == []


def test_a_standard_is_not_a_fabrication_even_off_its_own_data_plane() -> None:
    """Separator folding only rescues a vocabulary name when the evidence
    happens to mention the field. A specialist correctly reporting that it
    cannot see billing codes has no `icd10_code` in its evidence at all, and
    naming the standard it lacks was being read as inventing a value."""
    answer = "This data source holds no ICD-10 billing codes for that patient."
    assert ungrounded_values(answer, EVIDENCE) == []


def test_a_code_is_still_checked_when_its_vocabulary_is_named() -> None:
    """The vocabulary is the noun; the code is the claim. Only one is exempt."""
    answer = "The ICD-10 code on the claim is E11.9."
    assert ungrounded_values(answer, CLAIM_EVIDENCE) == ["E11.9"]


def test_prose_without_values_is_clean() -> None:
    assert ungrounded_values("The record does not say.", EVIDENCE) == []


def test_empty_inputs_are_clean() -> None:
    assert ungrounded_values("", EVIDENCE) == []
    assert ungrounded_values("Anything at all, N18.3.", "") == []


# -- what is flagged ---------------------------------------------------------


def test_the_observed_fabrication_is_caught() -> None:
    """The bug this file exists for: a birth date the chart does not contain.
    The model never saw a real date - the redactor replaced it with PHI_DOB_1 -
    so anything date-shaped in the answer was invented outright."""
    answer = "The patient, born 1980-06-15, has stage 3 chronic kidney disease."
    assert ungrounded_values(answer, EVIDENCE) == ["1980-06-15"]


def test_a_reformatted_real_date_is_not_flagged_but_a_wrong_one_is() -> None:
    answer = "Service was rendered on April 29, 2026, not on 2025-01-03."
    assert ungrounded_values(answer, CLAIM_EVIDENCE) == ["2025-01-03"]


def test_an_invented_clinical_code_is_caught() -> None:
    answer = "The patient is coded E11.9 for type 2 diabetes."
    assert ungrounded_values(answer, EVIDENCE) == ["E11.9"]


def test_a_real_clinical_code_is_not_caught() -> None:
    assert ungrounded_values("Coded N18.3, stage 3.", EVIDENCE) == []


def test_an_invented_cpt_code_is_caught() -> None:
    """A CPT code is a bare five-digit number, but it is never rounded - it is
    either the procedure the ledger holds or a different procedure entirely."""
    answer = "You should have billed 99214 instead."
    assert ungrounded_values(answer, CLAIM_EVIDENCE) == ["99214"]


def test_an_invented_record_id_is_caught() -> None:
    answer = "See claim CLM-9999 for the adjustment."
    assert ungrounded_values(answer, CLAIM_EVIDENCE) == ["CLM-9999"]


def test_a_placeholder_the_session_never_issued_is_caught() -> None:
    """Re-hydration would leave this one in the answer as raw text, or worse,
    fuzzy-match it onto somebody else's value."""
    answer = "PHI_PATIENT_7 is stable."
    assert ungrounded_values(answer, EVIDENCE) == ["PHI_PATIENT_7"]


def test_findings_are_deduplicated_in_order() -> None:
    answer = "E11.9 today, E11.9 last week, and code Z99.2."
    assert ungrounded_values(answer, EVIDENCE) == ["E11.9", "Z99.2"]


# -- values the request supplied ---------------------------------------------
#
# A specialist that says "I have no data on CLM-8849" is quoting the question,
# not inventing a record. Without the request in the haystack the guard reports
# the user's own words back as a fabrication, which is how it loses the reader.


def test_an_identifier_from_the_request_is_not_a_fabrication() -> None:
    """The observed false positive, verbatim.

    Clinical resolved patient 12345 correctly and said it could not see the
    claim. The live system flagged two values here - `CLM-8849` and `ICD-10` -
    and neither was invented. They fail for different reasons and are fixed
    separately: the standard's name by `_VOCABULARIES`, the claim id by having
    the request in the haystack. Only the second needs `request` to pass.
    """
    request = (
        "Claim CLM-8849 for patient 12345 was denied. Check their diagnosis "
        "history and tell me whether the billed code matches."
    )
    answer = (
        "A single condition (N18.3) is available for this patient. There is no "
        "information about claim CLM-8849 or its ICD-10 billing codes in the "
        "tool results."
    )

    assert ungrounded_values(answer, EVIDENCE) == ["CLM-8849"]
    assert ungrounded_values(answer, EVIDENCE, request) == []


def test_a_fabrication_is_still_caught_when_a_request_is_supplied() -> None:
    """The relaxation must not become a hole.

    Widening the haystack may only excuse values the request actually contains.
    """
    request = "Tell me about claim CLM-8849 for patient 12345."
    answer = "Claim CLM-8849 was billed under E11.9 against MRN-999999."

    assert ungrounded_values(answer, CLAIM_EVIDENCE, request) == ["E11.9", "MRN-999999"]


def test_a_request_naming_a_record_that_does_not_exist_is_still_excused() -> None:
    """Deliberate, and the cost of the rule above.

    The guard reports values the *model* produced from nowhere. A value the
    user typed is not one of those, even when the lookup missed.
    """
    request = "What happened to claim CLM-0001?"
    answer = "No claim matches CLM-0001."

    assert ungrounded_values(answer, CLAIM_EVIDENCE, request) == []


def test_an_absent_request_leaves_behaviour_unchanged() -> None:
    """The parameter is optional so `evals/` and older callers keep working."""
    answer = "Patient was billed under E11.9."
    assert ungrounded_values(answer, EVIDENCE) == ungrounded_values(answer, EVIDENCE, "")


# -- end to end --------------------------------------------------------------


def _fabricating(answer: str) -> StubProvider:
    return StubProvider(
        agents=["clinical"],
        plans={
            "clinical": [
                {"tool": "fhir_search_patient", "arguments": {"patient_id": "PHI_PATIENT_1"}}
            ]
        },
        answer=answer,
    )


async def test_a_fabricated_value_reaches_the_caller_labelled(make_orchestrator) -> None:
    provider = _fabricating("The patient was born 1980-06-15.")
    outcome = await make_orchestrator(provider).run("What is patient 12345 taking?")

    assert outcome["unverified"] == ["1980-06-15"]


async def test_a_flagged_answer_is_still_delivered(make_orchestrator) -> None:
    """Suppressing the answer would mean a rounded figure could delete a
    correct one. The claim ships labelled, not deleted."""
    provider = _fabricating("The patient was born 1980-06-15.")
    outcome = await make_orchestrator(provider).run("What is patient 12345 taking?")

    assert "1980-06-15" in outcome["answer"]


async def test_a_fabrication_is_visible_in_the_trace(make_orchestrator) -> None:
    trace = Trace()
    provider = _fabricating("Coded E11.9, born 1980-06-15.")
    await make_orchestrator(provider).run("What is patient 12345 taking?", trace)

    flagged = [s for s in trace.spans() if s.name == "Unverified values"]
    assert len(flagged) == 1
    assert flagged[0].kind is SpanKind.GUARDRAIL
    assert flagged[0].detail["count"] == 2
    # The invented date is date-of-birth shaped, so the audit record gets a
    # token rather than the value; the clinical code is not PHI and stays.
    # Not asserting the token's number - the chart's own date took PHI_DOB_1
    # on the way out of the tool, and which index this one lands on is not
    # what the test is about.
    date_token, code = flagged[0].detail["values"]
    assert date_token.startswith("PHI_DOB_")
    assert code == "E11.9"


async def test_the_trace_never_carries_a_flagged_identifier(make_orchestrator) -> None:
    """Found by `evals/edge_cases.py`, not by inspection.

    A phone number spelled out in words walks past the inbound patterns, so the
    model sees it intact and may re-type it as digits. It is then flagged
    precisely *because* no tool result contains it - which made this span the
    one place in the system that published raw PHI to the audit log.
    """
    provider = _fabricating("You gave 555-493-1882, which is not on file.")
    trace = Trace()
    outcome = await make_orchestrator(provider).run(
        "Their number is five five five, four nine three, one eight eight two - "
        "is that the one on file for patient 12345?",
        trace,
    )

    published = json.dumps(outcome["trace"], default=str)
    assert "555-493-1882" not in published
    assert "PHI_PHONE" in published, "the finding must still be legible to an auditor"


async def test_a_flagged_code_survives_into_the_trace_unredacted(make_orchestrator) -> None:
    """Redacting the trace must not cost the diagnostic. A clinical code
    matches no PHI pattern, so it reaches the audit record intact."""
    provider = _fabricating("The patient is coded E11.9.")
    trace = Trace()
    await make_orchestrator(provider).run("What is patient 12345 taking?", trace)

    flagged = next(s for s in trace.spans() if s.name == "Unverified values")
    assert flagged.detail["values"] == ["E11.9"]


async def test_a_supported_answer_raises_nothing(make_orchestrator) -> None:
    trace = Trace()
    provider = _fabricating("PHI_PATIENT_1 has chronic kidney disease, stage 3.")
    outcome = await make_orchestrator(provider).run("What is patient 12345 taking?", trace)

    assert outcome["unverified"] == []
    assert [s for s in trace.spans() if s.name == "Unverified values"] == []
