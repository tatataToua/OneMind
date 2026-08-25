"""The comparison's bookkeeping.

`evals/run_eval.py --arm both` claims the orchestrator beats a single agent
holding every tool. The first thing worth attacking in a claim like that is not
the model - it is whether the two arms were measured on equal terms. These
tests pin the parts that decide that, and they run offline, because a fairness
property should not depend on what a 4B model happened to say.

Decision #16 draws the line: tests stub the model, evals measure it. The stub
here answers the monolith's schema, which `conftest.StubProvider` deliberately
refuses to guess at.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from onemind.llm.base import Message
from onemind.orchestrator.registry import registry

# `evals/` is a script directory, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))

from arms import MonolithArm, RouterArm, build_arm  # noqa: E402
from run_eval import ArmResult, Observation, score  # noqa: E402


class ToolSelectingStub:
    """Returns a scripted tool selection, whatever it is asked."""

    name = "stub"

    def __init__(self, tools: list[str], *, is_actionable: bool = True, question: str = "") -> None:
        self.tools = tools
        self.is_actionable = is_actionable
        self.question = question

    async def complete(self, messages: Sequence[Message], *, temperature: float = 0.0) -> str:
        raise AssertionError("the monolith arm should only ever make a structured call")

    async def stream(self, messages: Sequence[Message], *, temperature: float = 0.0):
        raise AssertionError("the monolith arm should only ever make a structured call")
        yield ""  # pragma: no cover - makes this an async generator

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[BaseModel],
        *,
        temperature: float = 0.0,
    ) -> BaseModel:
        return schema.model_validate(
            {
                "is_actionable": self.is_actionable,
                "clarifying_question": self.question,
                "tools": self.tools if self.is_actionable else [],
                "rationale": "stub rationale",
            }
        )


def _observations(rows: list[tuple[dict, set[str]]]) -> list[Observation]:
    return [Observation(row=row, attempt=0, got=got, latency_ms=1.0) for row, got in rows]


def test_the_monolith_is_offered_exactly_what_the_specialists_own():
    """The fairness invariant, and the analogue of
    `test_specialists_do_not_share_tools`.

    Offer the monolith a tool no specialist has and the arms no longer have the
    same reach, so a win could be capability rather than architecture. Offer it
    fewer and it is a strawman. Both failures are silent in the output table,
    which is why they are asserted here.
    """
    arm = MonolithArm(ToolSelectingStub([]), registry)
    owned = {name for spec in registry.all() for name in spec.tool_names}

    assert set(arm.tool_names) == owned
    assert len(arm.tool_names) == len(owned), "a tool was offered twice"


async def test_tool_choices_map_back_to_the_specialists_that_own_them():
    single = MonolithArm(ToolSelectingStub(["fhir_search_patient", "fhir_get_resource"]), registry)
    assert await single.select("anything") == ({"clinical"}, "")

    across = MonolithArm(ToolSelectingStub(["fhir_get_resource", "claim_lookup"]), registry)
    got, question = await across.select("anything")
    assert got == {"clinical", "revenue_cycle"}
    assert question == ""


async def test_selecting_nothing_while_claiming_actionable_is_an_abstention():
    """The same charity `Router._normalise` extends to the router.

    A model that says "yes I can help" and names no tool has not routed
    anywhere. Scoring that as anything but an abstention would hand the
    monolith free credit on the ten vague prompts - or free blame, depending on
    which way the bookkeeping leaned. It has to lean the same way for both.
    """
    arm = MonolithArm(ToolSelectingStub([], is_actionable=True), registry)
    got, question = await arm.select("check the numbers")

    assert got == set()
    assert question, "an abstention must carry a question, as the router's does"


async def test_an_unknown_tool_name_is_unrepresentable_not_merely_filtered():
    """Both arms build their schema from the live roster, so an off-roster name
    fails validation rather than being quietly dropped.

    This mirrors `router._decision_schema`, and it is the reason the comparison
    can be read as a difference in judgement rather than in error handling: the
    monolith cannot lose points for naming a tool that does not exist, because
    it cannot name one.
    """
    arm = MonolithArm(ToolSelectingStub(["not_a_real_tool"]), registry)
    with pytest.raises(ValidationError, match="literal_error"):
        await arm.select("anything")


def test_both_arms_are_scored_by_identical_bookkeeping():
    """The claim the whole comparison rests on: a difference in the table is a
    difference in behaviour, never in how the two arms were counted."""
    rows: list[tuple[dict, set[str]]] = [
        ({"id": "a", "prompt": "p", "expect": ["clinical"]}, {"clinical"}),
        ({"id": "b", "prompt": "p", "expect": ["clinical"]}, {"clinical", "compliance"}),
        ({"id": "c", "prompt": "p", "expect": []}, set()),
        ({"id": "d", "prompt": "p", "expect": ["clinical", "revenue_cycle"]}, {"clinical"}),
    ]
    kwargs: dict[str, Any] = {"model": "stub", "cases": len(rows), "repeat": 1}

    a = score(ArmResult("router", _observations(rows)), **kwargs)
    b = score(ArmResult("monolith", _observations(rows)), **kwargs)

    assert a.pop("arm") == "router"
    assert b.pop("arm") == "monolith"
    assert a == b

    # And the numbers are the ones those four rows actually imply.
    assert a["single_agent_accuracy"] == 50.0  # b woke a spare specialist
    assert a["abstain_accuracy"] == 100.0
    assert a["label_recall"] == 75.0  # d missed revenue_cycle
    assert a["label_precision"] == 75.0  # b woke compliance


def test_stability_is_only_reported_when_there_is_repetition():
    """A single pass cannot say whether a result is stable, so it does not
    claim to."""
    row = {"id": "a", "prompt": "p", "expect": ["clinical"]}
    once = score(
        ArmResult("router", _observations([(row, {"clinical"})])),
        model="stub",
        cases=1,
        repeat=1,
    )
    assert "stability" not in once

    flipped = ArmResult(
        "router",
        [
            Observation(row=row, attempt=0, got={"clinical"}, latency_ms=1.0),
            Observation(row=row, attempt=1, got={"compliance"}, latency_ms=1.0),
        ],
    )
    twice = score(flipped, model="stub", cases=1, repeat=2)
    assert twice["stability"] == 0.0


def test_build_arm_rejects_an_unknown_arm():
    with pytest.raises(ValueError, match="unknown arm"):
        build_arm("supervisor", ToolSelectingStub([]), registry)

    assert isinstance(build_arm("monolith", ToolSelectingStub([]), registry), MonolithArm)
    assert isinstance(build_arm("router", ToolSelectingStub([]), registry), RouterArm)
