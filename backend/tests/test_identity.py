"""A request that names a person and gives an identifier must agree with itself.

The gap this closes, observed live: asked *"Look up Tobias Kaur with patient id
13344 and tell me whether their blood pressure has been trending high"*, the
system answered confidently about patient 13344 - Samuel Ferreira. The name was
discarded in silence.

Nothing was broken in isolation. Supplying an id satisfies Remote Monitoring's
declared `needs`, so it never blocked, the router never woke Clinical, and the
one data plane that can resolve a name never ran. No component was ever holding
both values, so no component could compare them.

`reconcile.py` calls a confident statement spanning two patients "the worst
output this module could produce" and guards its join key against it. This is
that same failure arriving one layer earlier, through the request itself.
"""

from __future__ import annotations

import pytest

from onemind.guardrails.identity import check_subject
from onemind.guardrails.phi import PHISession
from onemind.tools import store
from tests.conftest import StubProvider

NAMES = ["Tobias Kaur", "Samuel Ferreira", "Priya Okafor"]


def _resolve(name: str) -> list[str]:
    return [patient["patient_id"] for patient in store.find_patients(name)]


def _check(request: str):
    """Run the check the way the graph does: redact, then compare what the
    redacted request stands for."""
    session = PHISession(known_names=NAMES)
    return check_subject(session, _resolve, session.redact(request))


def test_name_and_id_naming_different_people_is_a_conflict() -> None:
    """12678 is Tobias Kaur. 13344 is Samuel Ferreira. The request names both
    and means one."""
    check = _check(
        "Look up Tobias Kaur with patient id 13344 and tell me about their blood pressure"
    )
    assert check.verdict == "conflict"
    assert check.supplied_patient_id == "13344"
    assert check.named_patient_id == "12678"


def test_the_conflict_question_names_neither_patient() -> None:
    """Same disclosure line `fhir._unresolved` draws: say that they disagree,
    never who either one is. The asker has identified nobody yet."""
    check = _check(
        "Look up Tobias Kaur with patient id 13344 and tell me about their blood pressure"
    )
    assert "13344" not in check.question
    assert "12678" not in check.question
    assert "Tobias" not in check.question


def test_name_and_id_naming_the_same_person_is_confirmed() -> None:
    check = _check(
        "Look up Tobias Kaur with patient id 12678 and tell me about their blood pressure"
    )
    assert check.verdict == "confirmed"
    assert check.named_patient_id == "12678"


def test_a_name_alone_is_not_this_check() -> None:
    """No identifier to disagree with. This is the two-hop path, and the check
    must leave it alone."""
    check = _check("Look up Tobias Kaur and tell me about their blood pressure")
    assert check.verdict == "not_applicable"


def test_an_id_alone_is_not_this_check() -> None:
    check = _check("What medications is patient 12345 currently taking?")
    assert check.verdict == "not_applicable"


def test_an_ambiguous_name_is_left_to_the_existing_refusal() -> None:
    """Two patients are called Samuel Ferreira and 13344 is one of them. The
    id resolves the ambiguity rather than contradicting it, so there is no
    conflict to report."""
    check = _check("Look up Samuel Ferreira with patient id 13344 and tell me about their labs")
    assert check.verdict == "confirmed"


def test_an_unknown_name_cannot_contradict_anything() -> None:
    """A name the corpus does not hold resolves to nobody. Reporting a conflict
    would be asserting that the id is wrong on the strength of a name we cannot
    place."""
    assert _check("Look up patient id 12678 for Winston Marlowe").verdict == "not_applicable"


@pytest.mark.parametrize(
    "request_text", ["", "check the numbers", "How long must we retain audit logs?"]
)
def test_requests_with_no_identity_in_them(request_text: str) -> None:
    assert _check(request_text).verdict == "not_applicable"


# -- end to end, through the graph -------------------------------------------


@pytest.mark.asyncio
async def test_a_contradictory_request_is_refused_before_any_data_plane(
    make_orchestrator,
) -> None:
    """The live failure, end to end. Nothing may be retrieved: checking before
    dispatch rather than after fan-in is the whole point, because by the time
    there is evidence to compare, the wrong patient's records have been read."""
    provider = StubProvider(
        agents=["remote_monitoring"],
        plans={
            "remote_monitoring": [
                {
                    "tool": "evaluate_thresholds",
                    "arguments": {
                        "patient_id": "PHI_PATIENT_1",
                        "metric": "blood_pressure_systolic",
                    },
                }
            ]
        },
        answer="Blood pressure is trending high.",
    )
    outcome = await make_orchestrator(provider).run(
        "Look up Tobias Kaur with patient id 13344 and tell me whether their "
        "blood pressure has been trending high"
    )

    assert outcome["is_actionable"] is False
    assert outcome["agents"] == []
    assert outcome["clarifying_question"]
    assert "13344" not in outcome["answer"]
    assert "12678" not in outcome["answer"]
    assert not any(span["kind"] == "tool" for span in outcome["trace"]["spans"])


@pytest.mark.asyncio
async def test_a_consistent_request_still_runs_and_states_the_link(
    make_orchestrator,
) -> None:
    """The other half. A name and an identifier that agree must not be refused,
    and the agreement is reported as a computed finding - which is what stops
    the model reporting "no data found" for a name that appears in no record."""
    provider = StubProvider(
        agents=["remote_monitoring"],
        plans={
            "remote_monitoring": [
                {
                    "tool": "evaluate_thresholds",
                    "arguments": {
                        "patient_id": "PHI_PATIENT_1",
                        "metric": "blood_pressure_systolic",
                    },
                }
            ]
        },
        answer="Blood pressure is trending high.",
    )
    outcome = await make_orchestrator(provider).run(
        "Look up Tobias Kaur with patient id 12678 and tell me whether their "
        "blood pressure has been trending high"
    )

    assert outcome["is_actionable"] is True
    findings = {f["check"]: f["statement"] for f in outcome["findings"]}
    assert "named_patient_matches_supplied_id" in findings
    statement = findings["named_patient_matches_supplied_id"]
    assert "Tobias Kaur" in statement and "12678" in statement


def test_a_name_paired_with_someone_elses_mrn_is_a_conflict() -> None:
    """Same defect, one identifier shape over. MRN-861301 is Tobias Kaur's, and
    the PHI-redaction demo prompt is exactly this shape - a name beside an MRN -
    so a guard that only understood bare patient ids would have a hole in it
    precisely where the demo points."""
    check = _check("Priya Okafor, MRN-861301, SSN 156-44-5517 - what are they taking?")
    assert check.verdict == "conflict"


def test_a_name_paired_with_their_own_mrn_is_confirmed() -> None:
    check = _check("Priya Okafor, MRN-943792, SSN 156-44-5517 - what are they taking?")
    assert check.verdict == "confirmed"
    assert check.named_patient_id == "13677"


def test_an_unknown_identifier_cannot_contradict_a_name() -> None:
    """A lookup that failed is not evidence. Refusing here would reject a valid
    request because a typo'd identifier matched nobody."""
    check = _check("Look up Tobias Kaur with patient id 99999")
    assert check.verdict == "not_applicable"


def test_a_name_from_an_earlier_turn_does_not_contradict_this_one() -> None:
    """`PHISession` spans the whole conversation - a token must keep meaning the
    same person for as long as anyone can refer back to it. So its mapping holds
    every name and identifier the conversation has ever mentioned, and reading
    the mapping alone would let turn one's patient contradict turn two's.

    Asking about Tobias Kaur and then about patient 12345 is two questions about
    two patients, which is ordinary. It is not a request that names two people.
    """
    session = PHISession(known_names=NAMES)
    session.redact("What medications is Tobias Kaur taking?")
    this_turn = session.redact("and what are patient 12345's recent labs?")

    assert check_subject(session, _resolve, this_turn).verdict == "not_applicable"
