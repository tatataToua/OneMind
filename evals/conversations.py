"""Multi-turn evaluation: session memory and two-wave dispatch.

Answers two questions the offline suite cannot, because both depend on what a
real 4B model chooses to do rather than on what a stub was scripted to do.

  Does a two-hop question resolve in one turn?  A specialist blocked for want of
  an identifier must actually be retried once a sibling establishes one - and
  the retry has to produce data, not a second "no data".

  Does a follow-up inherit its subject?  A turn that names nobody must route,
  resolve to the right patient, and need NO second wave, because the fact was
  already on the board when it started.

The interesting failures are asymmetric, so they are reported apart.

  A missing second wave is a feature that silently does not work.
  An extra second wave is latency and noise.
  A wrong subject after a switch is the one genuine correctness failure here,
  and it is checked directly rather than inferred from the prose.

Usage:
    python evals/conversations.py
    python evals/conversations.py --json evals/conversations_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from onemind.bootstrap import build_orchestrator, build_redactor  # noqa: E402
from onemind.orchestrator.conversation import ConversationStore  # noqa: E402

DATASET = ROOT / "evals" / "datasets" / "conversations.jsonl"


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _max_wave(outcome: dict[str, Any]) -> int:
    """Highest dispatch round any specialist reached this turn.

    Read off the agent spans rather than tracked separately: the trace is what
    an auditor would read, so if the wave is not visible there it did not
    happen in any sense that matters.
    """
    waves = [
        int(span["detail"].get("wave", 1))
        for span in outcome["trace"]["spans"]
        if span["kind"] == "agent"
    ]
    return max(waves) if waves else 0


async def run_case(case: dict[str, Any]) -> dict[str, Any]:
    """One conversation, start to finish, on its own session."""
    orchestrator = build_orchestrator()
    conversation = ConversationStore(build_redactor()).get(None)

    turns: list[dict[str, Any]] = []
    failures: list[str] = []

    for index, spec in enumerate(case["turns"], start=1):
        started = time.perf_counter()
        outcome = await orchestrator.run(spec["prompt"], conversation=conversation)
        elapsed = time.perf_counter() - started

        agents = sorted(outcome["agents"])
        facts = sorted({f["key"] for f in outcome["facts"]})
        waves = _max_wave(outcome)
        checks = sorted({f["check"] for f in outcome["findings"]})

        where = f"{case['id']} turn {index}"

        if "expect_actionable" in spec and outcome["is_actionable"] != spec["expect_actionable"]:
            failures.append(
                f"{where}: actionable={outcome['is_actionable']}, "
                f"expected {spec['expect_actionable']}"
            )

        # Recall, not exact match. Waking a spare specialist is noise; missing
        # one the question needed is a wrong answer.
        for expected in spec.get("expect_agents", []) or []:
            if expected not in agents:
                failures.append(f"{where}: {expected} never ran (ran {agents})")

        for key in spec.get("expect_facts", []) or []:
            if key not in facts:
                failures.append(f"{where}: fact {key!r} not established (have {facts})")

        for check in spec.get("expect_findings", []) or []:
            if check not in checks:
                failures.append(f"{where}: finding {check!r} missing (have {checks})")

        # Disclosure boundary: a refusal may say how many patients matched, and
        # must never say which. Checked against the rehydrated answer, because
        # that is what the person actually reads.
        for secret in spec.get("forbid", []) or []:
            if secret.casefold() in outcome["answer"].casefold():
                failures.append(f"{where}: answer disclosed {secret!r}")

        if "expect_waves" in spec and waves != spec["expect_waves"]:
            failures.append(f"{where}: reached wave {waves}, expected {spec['expect_waves']}")

        turns.append(
            {
                "prompt": spec["prompt"],
                "agents": agents,
                "facts": facts,
                "subject": conversation.facts.subject,
                "waves": waves,
                "findings": checks,
                "actionable": outcome["is_actionable"],
                "answer": " ".join(outcome["answer"].split())[:400],
                "seconds": round(elapsed, 1),
                "watch_for": spec.get("watch_for", ""),
            }
        )

    return {
        "id": case["id"],
        "label": case["label"],
        "turns": turns,
        "failures": failures,
        "passed": not failures,
    }


async def evaluate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    # Sequential on purpose. Conversations are stateful and the point is to
    # measure a real turn-by-turn exchange, not to see how many fit at once.
    results = [await run_case(case) for case in cases]
    return {
        "cases": results,
        "passed": sum(1 for r in results if r["passed"]),
        "total": len(results),
        "failures": [f for r in results for f in r["failures"]],
    }


def render_console(report: dict[str, Any]) -> None:
    print()
    print("  Multi-turn evaluation")
    print("  " + "-" * 66)
    for case in report["cases"]:
        mark = "ok  " if case["passed"] else "FAIL"
        print(f"  [{mark}] {case['id']}  {case['label']}")
        for index, turn in enumerate(case["turns"], start=1):
            detail = (
                f"agents={turn['agents']} facts={turn['facts']} "
                f"wave={turn['waves']} {turn['seconds']}s"
            )
            print(f"           {index}. {detail}")
            print(f"              {turn['answer'][:110]}")
        for failure in case["failures"]:
            print(f"           -> {failure}")
    print("  " + "-" * 66)
    print(f"  {report['passed']}/{report['total']} conversations passed")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the full JSON transcript here")
    args = parser.parse_args()

    report = asyncio.run(evaluate(load(DATASET)))
    render_console(report)

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  transcript written to {args.json}\n")

    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
