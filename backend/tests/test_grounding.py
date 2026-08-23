"""Identifier grounding: a lookup key the request never supplied is not a lookup.

Constrained decoding makes a tool argument well-formed. It cannot make it true.
The gap between those two is where this system leaked: asked a hypothetical
question that named no patient, the planner filled `patient_id` from the sample
value in the tool's own description, the store returned that real record, and
the specialist recited a stranger's diagnosis and date of birth.

These tests run against the real tool registry and the real fixture store, with
only the model stubbed - the point is that the guard holds at the data plane,
not that a prompt was reworded.
"""

from __future__ import annotations

from onemind.observability.trace import SpanKind, Trace

from .conftest import StubProvider

# A real record in the fixture set, and the value that leaked. `fhir.py` offers
# it as the example in `fhir_search_patient`'s own parameter description, which
# is where the planner picked it up.
REAL_PATIENT_ID = "12345"
REAL_CLAIM_ID = "CLM-8840"

# A question with no patient in it, of the kind that produced the leak.
HYPOTHETICAL = (
    "Act as an anxious, newly diagnosed Type 2 diabetes patient who is resistant "
    "to starting insulin. How do I explain the necessity without causing panic?"
)


def _clinical(**arguments: str) -> StubProvider:
    return StubProvider(
        agents=["clinical"],
        plans={"clinical": [{"tool": "fhir_search_patient", "arguments": arguments}]},
    )


def _tool_spans(trace: Trace) -> list[str]:
    return [s.name for s in trace.spans() if s.kind is SpanKind.TOOL]


# -- the leak ----------------------------------------------------------------


async def test_invented_identifier_never_reaches_the_data_plane(make_orchestrator) -> None:
    """The regression this file exists for."""
    trace = Trace()
    await make_orchestrator(_clinical(patient_id=REAL_PATIENT_ID)).run(HYPOTHETICAL, trace)

    assert _tool_spans(trace) == [], "a patient the request never named was looked up"


async def test_a_blocked_lookup_says_so_instead_of_answering(make_orchestrator) -> None:
    outcome = await make_orchestrator(_clinical(patient_id=REAL_PATIENT_ID)).run(HYPOTHETICAL)

    assert "does not identify a record" in outcome["answer"]


async def test_a_blocked_lookup_is_visible_in_the_trace(make_orchestrator) -> None:
    """An audit reader has to be able to see that the model tried."""
    trace = Trace()
    await make_orchestrator(_clinical(patient_id=REAL_PATIENT_ID)).run(HYPOTHETICAL, trace)

    blocked = [s for s in trace.spans() if s.detail.get("blocked")]
    assert len(blocked) == 1
    assert blocked[0].kind is SpanKind.GUARDRAIL
    assert blocked[0].detail["arguments"] == ["patient_id"]
    # The rejected value is deliberately absent: it is unverified, and this one
    # happens to be a real person's id that the model guessed.
    assert REAL_PATIENT_ID not in str(blocked[0].detail)


async def test_an_invented_date_of_birth_is_blocked_too(make_orchestrator) -> None:
    """A wrong second identifier is worse than none - it silently narrows an
    ambiguous name match onto the wrong person."""
    trace = Trace()
    provider = _clinical(patient_id="PHI_PATIENT_1", birth_date="1980-06-15")
    await make_orchestrator(provider).run("What is patient 12345 taking?", trace)

    assert _tool_spans(trace) == []


# -- what must still work ----------------------------------------------------


async def test_a_redacted_placeholder_is_grounded(make_orchestrator) -> None:
    """The ordinary path: the request named a patient, so the model holds a
    placeholder for them and the lookup proceeds."""
    trace = Trace()
    provider = _clinical(patient_id="PHI_PATIENT_1")
    await make_orchestrator(provider).run("What is patient 12345 taking?", trace)

    assert _tool_spans(trace) == ["fhir_search_patient"]


async def test_a_mangled_placeholder_is_still_grounded(make_orchestrator) -> None:
    """Models rewrite their own tokens. Grounding reuses the redactor's
    tolerance for that rather than adding a second, stricter matcher - a
    misspelled token must not read as an invented identifier."""
    trace = Trace()
    provider = _clinical(patient_id="PHI_PANTIENT_1")
    await make_orchestrator(provider).run("What is patient 12345 taking?", trace)

    assert _tool_spans(trace) == ["fhir_search_patient"]


async def test_a_record_id_the_request_spells_out_is_grounded(make_orchestrator) -> None:
    """Claim, device and encounter ids are deliberately never redacted - they
    identify records, not people - so they arrive as literals and must ground
    against the request text."""
    trace = Trace()
    provider = StubProvider(
        agents=["revenue_cycle"],
        plans={
            "revenue_cycle": [{"tool": "claim_lookup", "arguments": {"claim_id": REAL_CLAIM_ID}}]
        },
    )
    await make_orchestrator(provider).run(f"Why was claim {REAL_CLAIM_ID} denied?", trace)

    assert _tool_spans(trace) == ["claim_lookup"]


async def test_an_invented_claim_id_is_blocked(make_orchestrator) -> None:
    trace = Trace()
    provider = StubProvider(
        agents=["revenue_cycle"],
        plans={
            "revenue_cycle": [{"tool": "claim_lookup", "arguments": {"claim_id": REAL_CLAIM_ID}}]
        },
    )
    await make_orchestrator(provider).run("What is our denial rate this quarter?", trace)

    assert _tool_spans(trace) == []


async def test_non_identifying_arguments_are_not_grounded(make_orchestrator) -> None:
    """Only arguments that select whose record is returned are checked. The
    model is expected to choose `resource` itself; requiring the request to
    contain the word "labs" would break every question phrased normally."""
    trace = Trace()
    provider = StubProvider(
        agents=["clinical"],
        plans={
            "clinical": [
                {
                    "tool": "fhir_get_resource",
                    "arguments": {"patient_id": "PHI_PATIENT_1", "resource": "labs"},
                }
            ]
        },
    )
    await make_orchestrator(provider).run("Any recent bloodwork for patient 12345?", trace)

    assert _tool_spans(trace) == ["fhir_get_resource"]


async def test_grounding_does_not_block_tools_that_take_no_identifier(
    make_orchestrator,
) -> None:
    trace = Trace()
    provider = StubProvider(
        agents=["revenue_cycle"],
        plans={"revenue_cycle": [{"tool": "denial_summary", "arguments": {}}]},
    )
    await make_orchestrator(provider).run("What is our denial rate this quarter?", trace)

    assert _tool_spans(trace) == ["denial_summary"]


# -- coded values ------------------------------------------------------------
#
# A second harm, through an argument `_is_identifier` deliberately does not
# cover. A fabricated code does not open the wrong person's chart; it produces
# a confident, checkable-looking statement about the wrong procedure.


async def test_a_code_from_a_tool_description_is_blocked(make_orchestrator) -> None:
    """Observed live on the cross-agent claim question.

    `validate_code`'s own parameter description reads "e.g. E11.9, 99214, or
    CO-197". Asked about a claim billed under N18.3, the planner validated
    E11.9 - a value the request never contained and the ledger never returned -
    and the answer reported a coding mismatch that did not exist.
    """
    trace = Trace()
    provider = StubProvider(
        agents=["revenue_cycle"],
        plans={"revenue_cycle": [{"tool": "validate_code", "arguments": {"code": "E11.9"}}]},
    )
    await make_orchestrator(provider).run(f"Why was claim {REAL_CLAIM_ID} denied?", trace)

    assert _tool_spans(trace) == [], "a code the request never named was validated"


async def test_a_code_the_request_names_is_validated(make_orchestrator) -> None:
    trace = Trace()
    provider = StubProvider(
        agents=["revenue_cycle"],
        plans={"revenue_cycle": [{"tool": "validate_code", "arguments": {"code": "N18.3"}}]},
    )
    await make_orchestrator(provider).run("Is N18.3 a valid diagnosis code?", trace)

    assert _tool_spans(trace) == ["validate_code"]


async def test_a_blocked_code_names_its_argument_in_the_trace(make_orchestrator) -> None:
    trace = Trace()
    provider = StubProvider(
        agents=["revenue_cycle"],
        plans={"revenue_cycle": [{"tool": "validate_code", "arguments": {"code": "E11.9"}}]},
    )
    await make_orchestrator(provider).run(f"Why was claim {REAL_CLAIM_ID} denied?", trace)

    blocked = [s for s in trace.spans() if s.detail.get("blocked")]
    assert len(blocked) == 1
    assert blocked[0].detail["arguments"] == ["code"]
