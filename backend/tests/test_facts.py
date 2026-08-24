"""The blackboard: extraction, redaction space, and subject scoping.

The tests that matter here are the ones about redaction space. A `Facts` store
holding a real identifier would put one into a specialist's planning prompt,
which is the single failure this module is shaped to prevent.
"""

from __future__ import annotations

from onemind.agents.base import SpecialistResult
from onemind.orchestrator.facts import (
    Facts,
    collect,
    extract,
    keys_for_tools,
)


def _result(agent: str, tool: str, output: dict) -> SpecialistResult:
    return SpecialistResult(
        agent=agent,
        display_name=agent.title(),
        answer="",
        tool_calls=[{"tool": tool, "arguments": {}, "result": output}],
    )


# -- redaction space ---------------------------------------------------------


def test_extracted_identifier_is_a_placeholder_never_the_value(session) -> None:
    """The whole point of the module.

    `redact_json` cannot tokenise `"patient_id": "12345"` - `_PATIENT_ID` needs
    whitespace after the word and finds `_id": "`. So the raw value reaches
    here intact, and putting it on the board unchanged would hand a real
    identifier to the model.
    """
    facts = extract(
        "fhir_search_patient",
        {"found": True, "patient_id": "12345", "mrn": "MRN-99887"},
        session,
        source="clinical.fhir_search_patient",
    )

    values = {f.key: f.value for f in facts}
    assert values["patient_id"].startswith("PHI_")
    assert "12345" not in values["patient_id"]
    assert session.rehydrate(values["patient_id"]) == "12345"


def test_prompt_block_leaks_no_real_identifier(session) -> None:
    facts = Facts()
    for fact in extract(
        "fhir_search_patient",
        {"found": True, "patient_id": "12345", "mrn": "MRN-99887"},
        session,
        source="clinical.fhir_search_patient",
    ):
        facts.set(fact.key, fact.value, source=fact.source)

    block = facts.as_prompt_block()
    assert "12345" not in block
    assert "MRN-99887" not in block
    assert "PHI_" in block


def test_an_existing_placeholder_is_not_tokenised_twice(session) -> None:
    """A value the request already supplied keeps the token it already has."""
    redacted = session.redact("patient 12345")
    token = redacted.split()[-1]

    facts = extract(
        "fhir_search_patient",
        {"found": True, "patient_id": token},
        session,
        source="clinical.fhir_search_patient",
    )
    assert facts[0].value == token
    assert session.count == 1


def test_the_same_value_seen_twice_keeps_one_token(session) -> None:
    first = extract(
        "fhir_search_patient",
        {"found": True, "patient_id": "12345"},
        session,
        source="a",
    )
    second = extract(
        "claim_lookup",
        {"found": True, "claims": [{"claim_id": "CLM-1", "patient_id": "12345"}]},
        session,
        source="b",
    )
    assert first[0].value == second[0].value


# -- what must not become a fact ---------------------------------------------


def test_an_ambiguous_match_yields_nothing(session) -> None:
    """`found: False, ambiguous: True` is exactly the case where taking the
    identifier would attach the wrong person to the rest of the session."""
    assert (
        extract(
            "fhir_search_patient",
            {"found": False, "ambiguous": True, "match_count": 3, "patient_id": "12345"},
            session,
            source="clinical.fhir_search_patient",
        )
        == []
    )


def test_an_errored_result_yields_nothing(session) -> None:
    assert extract("fhir_search_patient", {"error": "boom"}, session, source="x") == []


def test_an_unregistered_tool_yields_nothing(session) -> None:
    assert extract("policy_search", {"found": True, "patient_id": "1"}, session, source="x") == []


def test_disagreeing_claim_rows_yield_no_patient(session) -> None:
    """A ledger spanning two patients cannot name one subject, and guessing
    which row is the subject is the error this module exists to avoid."""
    assert (
        extract(
            "claim_lookup",
            {
                "found": True,
                "claims": [
                    {"claim_id": "CLM-1", "patient_id": "111"},
                    {"claim_id": "CLM-2", "patient_id": "222"},
                ],
            },
            session,
            source="revenue_cycle.claim_lookup",
        )
        == []
    )


def test_a_raising_extractor_does_not_propagate(session, monkeypatch) -> None:
    from onemind.orchestrator import facts as facts_module

    def boom(_output: dict) -> dict:
        raise RuntimeError("bad extractor")

    monkeypatch.setitem(
        facts_module.PROVIDERS,
        "fhir_search_patient",
        facts_module.Provider(tool="fhir_search_patient", keys=("patient_id",), fn=boom),
    )
    assert extract("fhir_search_patient", {"found": True}, session, source="x") == []


# -- subject scoping ---------------------------------------------------------


def test_switching_patient_drops_the_previous_facts() -> None:
    facts = Facts()
    facts.set("patient_id", "PHI_PATIENT_1", source="clinical.fhir_search_patient")
    facts.set("mrn", "PHI_MRN_1", source="clinical.fhir_search_patient")
    assert facts.keys() == {"patient_id", "mrn"}

    facts.set("patient_id", "PHI_PATIENT_2", source="clinical.fhir_search_patient")
    assert facts.keys() == {"patient_id"}
    assert facts.subject == "PHI_PATIENT_2"


def test_re_resolving_the_same_patient_keeps_everything() -> None:
    facts = Facts()
    facts.set("patient_id", "PHI_PATIENT_1", source="a")
    facts.set("mrn", "PHI_MRN_1", source="a")
    facts.set("patient_id", "PHI_PATIENT_1", source="b")
    assert facts.keys() == {"patient_id", "mrn"}


def test_an_empty_board_renders_no_block() -> None:
    assert Facts().as_prompt_block() == ""


# -- derivation --------------------------------------------------------------


def test_provides_is_derived_from_tools() -> None:
    from onemind.orchestrator.registry import registry

    clinical = registry.get("clinical")
    assert "patient_id" in clinical.provides
    assert "mrn" in clinical.provides
    # Compliance reads a policy corpus; it establishes nothing about a person.
    assert registry.get("compliance").provides == ()


def test_keys_for_unknown_tools_is_empty() -> None:
    assert keys_for_tools(["policy_search", "nonexistent"]) == ()


# -- collection over results -------------------------------------------------


def test_collect_reads_every_specialist_result(session) -> None:
    facts = collect(
        [
            _result("clinical", "fhir_search_patient", {"found": True, "patient_id": "12345"}),
            _result("compliance", "policy_search", {"found": True, "text": "..."}),
        ],
        session,
        turn=2,
    )
    assert facts.keys() == {"patient_id"}
    assert facts.get("patient_id").source == "clinical.fhir_search_patient"
    assert facts.get("patient_id").turn == 2


def test_collect_into_an_existing_board_accumulates(session) -> None:
    board = Facts()
    collect(
        [_result("clinical", "fhir_search_patient", {"found": True, "patient_id": "12345"})],
        session,
        into=board,
    )
    collect(
        [_result("clinical", "fhir_search_patient", {"found": True, "mrn": "MRN-4"})],
        session,
        into=board,
    )
    assert board.keys() == {"patient_id", "mrn"}


def test_a_blocked_result_contributes_nothing(session, redactor) -> None:
    blocked = SpecialistResult(
        agent="remote_monitoring",
        display_name="Remote Monitoring",
        answer="",
        error="Remote Monitoring: the request does not identify a record it can look up",
        blocked=True,
    )
    assert len(collect([blocked], session)) == 0


def test_a_new_subjects_mrn_survives_the_switch(session) -> None:
    """Ordering hazard: `set` clears the board when the patient changes, so an
    mrn written before its own patient_id would be filed under the outgoing
    subject and immediately deleted. Fan-in has no fixed order, so `collect`
    must not depend on which specialist finished first."""
    board = Facts()
    board.set("patient_id", "PHI_PATIENT_1", source="turn1")
    board.set("mrn", "PHI_MRN_1", source="turn1")

    # mrn arrives from one result, the new patient_id from another, mrn first.
    collect(
        [
            _result("clinical", "fhir_search_patient", {"found": True, "mrn": "MRN-55555"}),
            _result("clinical", "fhir_get_resource", {"found": True, "patient_id": "99999"}),
        ],
        session,
        into=board,
    )

    assert board.keys() == {"patient_id", "mrn"}
    assert session.rehydrate(board.value("mrn")) == "MRN-55555"
    assert session.rehydrate(board.value("patient_id")) == "99999"
