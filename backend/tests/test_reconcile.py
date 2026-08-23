"""Cross-plane reconciliation: the comparison no single specialist can make.

Each specialist answers from one data plane and correctly reports the limits of
that plane. Asked whether a claim's billed diagnosis matches the patient's
history, Clinical holds the problem list and Revenue Cycle holds the claim, and
neither can see the other. Two accurate "I do not have that" statements compose
into a wrong answer.

These checks run over the merged evidence after fan-in. They are pure functions
over the tool results the specialists already retrieved - no model, no store, no
second lookup - so they are tested directly and exhaustively here.

The verdicts matter as much as the happy path: a check that guesses when the
evidence is thin is worse than no check, because its output is stated as fact.
"""

from __future__ import annotations

import json

from onemind.agents.base import SpecialistResult
from onemind.guardrails.grounding import ungrounded_values
from onemind.observability.trace import SpanKind, Trace
from onemind.orchestrator.reconcile import CHECKS, Evidence, reconcile

from .conftest import StubProvider

# -- evidence builders -------------------------------------------------------


def _result(agent: str, display: str, calls: list[dict]) -> SpecialistResult:
    return SpecialistResult(
        agent=agent,
        display_name=display,
        answer=f"{display} answered.",
        tool_calls=calls,
    )


def _claim(
    claim_id: str = "CLM-8840",
    patient_id: str = "12345",
    icd10: str = "N18.3",
    denial_code: str | None = "CO-197",
    category: str | None = "authorization",
    coding_related: bool | None = False,
) -> dict:
    row = {
        "claim_id": claim_id,
        "patient_id": patient_id,
        "cpt_code": "99214",
        "icd10_code": icd10,
        "status": "denied" if denial_code else "paid",
        "denial_code": denial_code,
        "denial_reason": "Precertification/authorization absent" if denial_code else None,
    }
    if category is not None:
        row["denial_category"] = category
        row["denial_coding_related"] = coding_related
    return row


def _revenue(*claims: dict, found: bool = True) -> SpecialistResult:
    return _result(
        "revenue_cycle",
        "Revenue Cycle",
        [
            {
                "tool": "claim_lookup",
                "arguments": {"claim_id": claims[0]["claim_id"] if claims else "CLM-0000"},
                "result": {"found": found, "count": len(claims), "claims": list(claims)},
            }
        ],
    )


def _clinical(patient_id: str = "12345", codes: tuple[str, ...] = ("N18.3",)) -> SpecialistResult:
    return _result(
        "clinical",
        "Clinical",
        [
            {
                "tool": "fhir_search_patient",
                "arguments": {"patient_id": patient_id},
                "result": {
                    "found": True,
                    "patient_id": patient_id,
                    "name": "PHI_NAME_1",
                    "conditions": [{"code": c, "display": f"display for {c}"} for c in codes],
                },
            }
        ],
    )


def _by_check(findings, name):
    return [f for f in findings if f.check == name]


# -- billed diagnosis vs problem list ----------------------------------------

DX = "billed_diagnosis_matches_problem_list"


def test_a_billed_code_on_the_problem_list_matches() -> None:
    findings = _by_check(reconcile([_clinical(), _revenue(_claim(icd10="N18.3"))]), DX)

    assert len(findings) == 1
    assert findings[0].verdict == "match"
    assert findings[0].compared["billed_icd10"] == "N18.3"
    assert "CLM-8840" in findings[0].provenance


def test_a_billed_code_absent_from_the_problem_list_mismatches() -> None:
    """The verdict the fixture corpus could not reach until `generate.py`
    appended a claim for it."""
    findings = _by_check(
        reconcile([_clinical(codes=("M17.9", "N18.3")), _revenue(_claim(icd10="E11.9"))]), DX
    )

    assert len(findings) == 1
    assert findings[0].verdict == "mismatch"
    assert findings[0].compared["billed_icd10"] == "E11.9"
    assert "M17.9" in findings[0].compared["problem_list"]


def test_the_statement_is_not_produced_by_a_model() -> None:
    """The whole point of the design: this sentence is a format string."""
    findings = _by_check(reconcile([_clinical(), _revenue(_claim(icd10="N18.3"))]), DX)

    assert findings[0].statement == (
        "Billed diagnosis N18.3 on claim CLM-8840 is on the patient's active problem list."
    )


# -- the join-key guard ------------------------------------------------------


def test_records_for_different_patients_are_never_compared() -> None:
    """The worst thing this component could do is compare one person's claim
    against another person's chart. An ambiguous name match upstream is exactly
    how that would happen."""
    findings = _by_check(
        reconcile([_clinical(patient_id="12345"), _revenue(_claim(patient_id="12900"))]), DX
    )

    assert len(findings) == 1
    assert findings[0].verdict == "insufficient_evidence"
    assert "different patient" in findings[0].statement


def test_a_mismatched_patient_finding_states_no_comparison_was_made() -> None:
    findings = _by_check(
        reconcile([_clinical(patient_id="12345"), _revenue(_claim(patient_id="12900"))]), DX
    )

    assert findings[0].compared == {}


# -- thin evidence yields nothing, not a guess -------------------------------


def test_a_missing_claim_produces_no_finding() -> None:
    findings = _by_check(reconcile([_clinical(), _revenue(found=False)]), DX)
    assert findings == []


def test_a_patient_that_did_not_resolve_produces_no_finding() -> None:
    clinical = _result(
        "clinical",
        "Clinical",
        [
            {
                "tool": "fhir_search_patient",
                "arguments": {"patient_id": "99999"},
                "result": {"found": False, "detail": "no patient matches that id"},
            }
        ],
    )
    assert _by_check(reconcile([clinical, _revenue(_claim())]), DX) == []


def test_an_empty_problem_list_produces_no_finding() -> None:
    """Absence of recorded conditions is not evidence the billed code is wrong."""
    assert _by_check(reconcile([_clinical(codes=()), _revenue(_claim())]), DX) == []


def test_a_check_whose_tools_are_absent_does_not_run() -> None:
    """Clinical alone cannot answer a billing question, and must not try."""
    assert _by_check(reconcile([_clinical()]), DX) == []


def test_no_specialists_yields_no_findings() -> None:
    assert reconcile([]) == []


def test_every_claim_for_the_patient_is_checked() -> None:
    findings = _by_check(
        reconcile(
            [
                _clinical(codes=("N18.3",)),
                _revenue(
                    _claim(claim_id="CLM-8840", icd10="N18.3"),
                    _claim(claim_id="CLM-8843", icd10="E11.9"),
                ),
            ]
        ),
        DX,
    )

    assert [f.verdict for f in findings] == ["match", "mismatch"]


# -- denial classification ---------------------------------------------------

DENIAL = "denial_is_coding_related"


def test_an_authorization_denial_is_not_a_coding_problem() -> None:
    """The finding that makes the answer useful rather than merely correct: no
    amount of recoding fixes a missing precertification."""
    findings = _by_check(reconcile([_revenue(_claim(denial_code="CO-197"))]), DENIAL)

    assert len(findings) == 1
    assert findings[0].verdict == "not_applicable"
    assert "not a coding denial" in findings[0].statement


def test_a_coding_denial_is_flagged_as_one() -> None:
    findings = _by_check(
        reconcile(
            [
                _revenue(
                    _claim(denial_code="CO-11", category="coding", coding_related=True),
                )
            ]
        ),
        DENIAL,
    )

    assert len(findings) == 1
    assert findings[0].verdict == "applicable"


def test_a_paid_claim_produces_no_denial_finding() -> None:
    assert _by_check(reconcile([_revenue(_claim(denial_code=None))]), DENIAL) == []


def test_an_unclassified_denial_code_produces_no_finding() -> None:
    """A code the reference set does not carry yields silence, not a guess."""
    claim = _claim(denial_code="CO-999", category=None)
    assert _by_check(reconcile([_revenue(claim)]), DENIAL) == []


def test_the_denial_check_needs_no_clinical_evidence() -> None:
    findings = _by_check(reconcile([_revenue(_claim(denial_code="CO-197"))]), DENIAL)
    assert len(findings) == 1


# -- scoping to what the request named ---------------------------------------


def test_findings_are_scoped_to_the_claim_the_request_names() -> None:
    """`claim_lookup` by patient id returns every claim, so a question about one
    claim can produce findings about four. Observed live: handed all of them,
    the model mixed the claims up and contradicted its own finding."""
    results = [
        _clinical(codes=("N18.3",)),
        _revenue(
            _claim(claim_id="CLM-8840", icd10="N18.3"),
            _claim(claim_id="CLM-8843", icd10="E11.9"),
            _claim(claim_id="CLM-8972", icd10="E11.9"),
        ),
    ]
    findings = reconcile(results, "Why was claim CLM-8972 denied?")

    assert {f.subject for f in findings} == {"CLM-8972"}


def test_a_request_naming_no_record_reports_everything() -> None:
    """A question about the patient rather than one claim should get the lot."""
    results = [
        _clinical(codes=("N18.3",)),
        _revenue(
            _claim(claim_id="CLM-8840", icd10="N18.3"),
            _claim(claim_id="CLM-8843", icd10="E11.9"),
        ),
    ]
    findings = reconcile(results, "Are any of this patient's claims miscoded?")

    assert {f.subject for f in findings} == {"CLM-8840", "CLM-8843"}


def test_scoping_is_skipped_when_no_request_is_given() -> None:
    results = [_clinical(), _revenue(_claim(claim_id="CLM-8840"))]
    assert reconcile(results) == reconcile(results, "")


def test_a_request_naming_an_unrelated_claim_narrows_nothing() -> None:
    """Scoping may only ever narrow to records the evidence has findings for.
    A claim id the evidence never returned must not blank the whole set."""
    results = [_clinical(), _revenue(_claim(claim_id="CLM-8840"))]
    findings = reconcile(results, "What about claim CLM-9999?")

    assert [f.subject for f in findings] == ["CLM-8840", "CLM-8840"]


# -- properties the rest of the system relies on -----------------------------


def test_findings_never_trip_the_answer_grounding_guard() -> None:
    """Every value in a statement is copied out of the evidence, so a finding
    is grounded by construction. Asserted rather than assumed: if it stopped
    holding, the synthesiser would start flagging its own verified findings."""
    results = [
        _clinical(codes=("M17.9", "N18.3")),
        _revenue(
            _claim(claim_id="CLM-8972", icd10="E11.9", denial_code="CO-197"),
        ),
    ]
    evidence = json.dumps([c for r in results for c in r.tool_calls], default=str)

    findings = reconcile(results)
    assert findings, "expected findings to check"
    for finding in findings:
        assert ungrounded_values(finding.statement, evidence) == [], finding.statement


def test_every_registered_check_declares_what_it_needs() -> None:
    assert CHECKS, "no checks registered"
    for check in CHECKS:
        assert check.requires, f"{check.name} declares no required tools"
        assert check.name


def test_evidence_reads_across_specialists() -> None:
    ev = Evidence([_clinical(), _revenue(_claim())])

    assert ev.has("claim_lookup", "fhir_search_patient")
    assert not ev.has("telemetry_series")
    assert ev.first("claim_lookup")["found"] is True
    assert len(ev.outputs("fhir_search_patient")) == 1


def test_evidence_ignores_failed_tool_results() -> None:
    broken = _result(
        "revenue_cycle",
        "Revenue Cycle",
        [{"tool": "claim_lookup", "arguments": {}, "result": {"error": "boom"}}],
    )
    assert reconcile([_clinical(), broken]) == []


# -- end to end --------------------------------------------------------------
#
# The failure this module was built for, driven through the real graph with
# only the model stubbed. The tools, the fixture store and the guardrails are
# the real ones - the point is that the evidence reaches the reconciler, not
# that a prompt was reworded.


def _ask(claim_id: str) -> str:
    """The request must name the claim it asks about.

    Not cosmetic: the identifier guard grounds every lookup key against the
    request, so a question about one claim cannot open another.
    """
    return (
        f"Claim {claim_id} for patient 12345 was denied. Check their diagnosis "
        f"history and tell me whether the billed code matches."
    )


CROSS_PLANE = _ask("CLM-8972")


def _cross_plane(claim_id: str) -> StubProvider:
    """Both specialists plan a correct lookup on their own plane, and both
    answer that they cannot see the other's - which is what they really do.

    Clinical plans against `PHI_PATIENT_1`, not the raw id: the specialist is
    handed the redacted request, so the placeholder is what a real planner sees
    and the only form the identifier guard will accept. Claim ids are never
    redacted - they identify a record, not a person - so that one is literal.
    """
    return StubProvider(
        agents=["revenue_cycle", "clinical"],
        plans={
            "revenue_cycle": [{"tool": "claim_lookup", "arguments": {"claim_id": claim_id}}],
            "clinical": [
                {"tool": "fhir_search_patient", "arguments": {"patient_id": "PHI_PATIENT_1"}}
            ],
        },
        answer="I can only see my own data source.",
    )


async def test_the_cross_plane_question_now_produces_a_verdict(make_orchestrator) -> None:
    """The regression this module exists for.

    CLM-8972 bills E11.9 for a patient whose problem list is M17.9 and N18.3,
    and it was denied CO-11. Neither specialist can reach that conclusion; the
    reconciler reads both their results and does.
    """
    outcome = await make_orchestrator(_cross_plane("CLM-8972")).run(CROSS_PLANE)

    verdicts = {f["check"]: f["verdict"] for f in outcome["findings"]}
    assert verdicts["billed_diagnosis_matches_problem_list"] == "mismatch"
    assert verdicts["denial_is_coding_related"] == "applicable"


async def test_a_matching_claim_reports_a_match_end_to_end(make_orchestrator) -> None:
    """CLM-8975 bills a diagnosis that is on the chart, denied for timely
    filing - so the codes match and recoding would change nothing."""
    outcome = await make_orchestrator(_cross_plane("CLM-8975")).run(_ask("CLM-8975"))

    verdicts = {f["check"]: f["verdict"] for f in outcome["findings"]}
    assert verdicts["billed_diagnosis_matches_problem_list"] == "match"
    assert verdicts["denial_is_coding_related"] == "not_applicable"


async def test_findings_survive_phi_rehydration(make_orchestrator) -> None:
    """Findings are computed in redacted space and leave through the same door
    as the answer, so no placeholder may reach the caller."""
    outcome = await make_orchestrator(_cross_plane("CLM-8972")).run(CROSS_PLANE)

    assert outcome["findings"]
    for finding in outcome["findings"]:
        assert "PHI_" not in finding["statement"], finding
        assert "PHI_" not in finding["provenance"], finding


async def test_the_reconcile_span_is_in_the_trace(make_orchestrator) -> None:
    """An audit reader has to be able to see the comparison happen."""
    trace = Trace()
    await make_orchestrator(_cross_plane("CLM-8972")).run(CROSS_PLANE, trace)

    spans = [s for s in trace.spans() if s.kind is SpanKind.RECONCILE]
    assert len(spans) == 1
    assert spans[0].detail["checks"] == 2
    assert spans[0].ended_at is not None


async def test_a_single_plane_request_still_reconciles_nothing(make_orchestrator) -> None:
    """Clinical alone has no claim to compare against, and the request must
    survive that without a finding and without an error."""
    provider = StubProvider(
        agents=["clinical"],
        plans={
            "clinical": [
                {"tool": "fhir_search_patient", "arguments": {"patient_id": "PHI_PATIENT_1"}}
            ]
        },
    )
    outcome = await make_orchestrator(provider).run("What is patient 12345 taking?")

    assert outcome["findings"] == []
    assert outcome["answer"]
