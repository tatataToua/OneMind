"""Routing evaluation.

Answers one question: given a labelled prompt, does the system reach the right
specialists? Everything downstream depends on that decision, and it is the one
part of the pipeline whose correctness can be measured rather than eyeballed.

Reported separately, because they fail differently and matter differently:

  single-agent accuracy  exact match on one-specialist prompts
  multi-agent recall     did we reach every specialist a prompt needed
  multi-agent precision  did we wake specialists the prompt did not need
  abstain accuracy       vague prompts correctly refused rather than guessed

Precision and recall are split because their costs are asymmetric. Missing a
specialist means an incomplete answer. Waking a spare one costs a few seconds
and some noise. Optimising a single blended number hides that trade.

Two arms can answer that question, and `--arm both` runs them side by side:

  router     the system as built - a routing call over the specialist roster
  monolith   the obvious alternative - one agent holding every tool

Scoring is deliberately shared. Both arms produce a set of specialist keys and
both are handed to the same `score`, so a difference in the table is a
difference in architecture rather than in bookkeeping. See `arms.py` for what
the monolith is given, and why it is not a strawman.

Usage:
    python evals/run_eval.py
    python evals/run_eval.py --arm both --repeat 3 --json report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arms import Arm, build_arm  # noqa: E402

from onemind.agents import catalog  # noqa: E402,F401  (registers the roster)
from onemind.bootstrap import build_provider  # noqa: E402
from onemind.orchestrator.registry import registry  # noqa: E402

DATASET = Path(__file__).parent / "datasets" / "routing.jsonl"
ABSTAIN = "(abstain)"
ARMS = ("router", "monolith")


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@dataclass
class Observation:
    """One arm's answer to one prompt, on one attempt."""

    row: dict
    attempt: int
    got: set[str]
    latency_ms: float
    question: str = ""

    @property
    def expected(self) -> set[str]:
        return set(self.row["expect"])

    @property
    def bucket(self) -> str:
        n = len(self.expected)
        return "abstain" if not n else "single" if n == 1 else "multi"


@dataclass
class ArmResult:
    name: str
    observations: list[Observation] = field(default_factory=list)


async def run_arm(arm: Arm, rows: list[dict], repeat: int) -> ArmResult:
    """Collect raw observations. No scoring happens here - that is the point."""
    result = ArmResult(name=arm.name)
    for row in rows:
        for attempt in range(repeat):
            started = time.perf_counter()
            got, question = await arm.select(row["prompt"])
            elapsed = (time.perf_counter() - started) * 1000
            result.observations.append(
                Observation(
                    row=row,
                    attempt=attempt,
                    got=got,
                    latency_ms=elapsed,
                    question=question,
                )
            )
    return result


def score(result: ArmResult, *, model: str, cases: int, repeat: int) -> dict:
    """Turn observations into the report. Shared by both arms, on purpose."""
    exact: Counter = Counter()
    totals: Counter = Counter()
    latencies: list[float] = []
    confusion: dict[str, Counter] = defaultdict(Counter)
    misses: list[dict] = []
    by_case: dict[str, list[frozenset]] = defaultdict(list)

    tp = fp = fn = 0

    for obs in result.observations:
        expected, got = obs.expected, obs.got
        latencies.append(obs.latency_ms)
        totals[obs.bucket] += 1
        by_case[obs.row["id"]].append(frozenset(got))

        if got == expected:
            exact[obs.bucket] += 1
        else:
            misses.append(
                {
                    "id": obs.row["id"],
                    "attempt": obs.attempt,
                    "prompt": obs.row["prompt"],
                    "expected": sorted(expected),
                    "got": sorted(got),
                    "question": obs.question,
                }
            )

        tp += len(expected & got)
        fp += len(got - expected)
        fn += len(expected - got)

        # Confusion is charged per expected label, so a multi-agent prompt
        # contributes one row per specialist it should have reached.
        for label in expected or {ABSTAIN}:
            for prediction in got or {ABSTAIN}:
                confusion[label][prediction] += 1

    def pct(hit: int, total: int) -> float:
        return round(100 * hit / total, 1) if total else 0.0

    latencies.sort()
    report = {
        "arm": result.name,
        "model": model,
        "cases": cases,
        "runs": sum(totals.values()),
        "repeat": repeat,
        "single_agent_accuracy": pct(exact["single"], totals["single"]),
        "multi_agent_exact_match": pct(exact["multi"], totals["multi"]),
        "abstain_accuracy": pct(exact["abstain"], totals["abstain"]),
        "overall_exact_match": pct(sum(exact.values()), sum(totals.values())),
        "label_precision": pct(tp, tp + fp),
        "label_recall": pct(tp, tp + fn),
        "latency_ms": {
            "p50": round(latencies[len(latencies) // 2], 1),
            "p95": round(latencies[int(len(latencies) * 0.95)], 1),
            "max": round(latencies[-1], 1),
        },
        "counts": dict(totals),
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "misses": misses,
    }

    # Stability answers the question a single-pass number cannot: is 97% a
    # property of the system or of the run? A case is stable when every attempt
    # returned the same set - right or wrong. Only meaningful above repeat 1.
    if repeat > 1:
        stable = sum(1 for answers in by_case.values() if len(set(answers)) == 1)
        report["stability"] = pct(stable, len(by_case))

    return report


METRICS = [
    ("single-agent accuracy", "single_agent_accuracy", "single"),
    ("multi-agent exact match", "multi_agent_exact_match", "multi"),
    ("abstain accuracy", "abstain_accuracy", "abstain"),
    ("overall exact match", "overall_exact_match", None),
    (None, None, None),
    ("label precision", "label_precision", None),
    ("label recall", "label_recall", None),
]


def render(report: dict) -> None:
    counts = report["counts"]
    notes = {
        "single": f"{counts.get('single', 0)} runs",
        "multi": f"{counts.get('multi', 0)} runs",
        "abstain": f"{counts.get('abstain', 0)} runs",
    }
    extra = {
        "label_precision": "woken unnecessarily",
        "label_recall": "that should have run",
    }

    header = (
        f"  ROUTING EVALUATION [{report['arm']}] - {report['cases']} cases "
        f"x{report['repeat']} = {report['runs']} runs"
    )
    print()
    print("=" * 74)
    print(header)
    print("=" * 74)
    print()

    for label, key, bucket in METRICS:
        if key is None:
            print()
            continue
        note = notes.get(bucket or "", "") or extra.get(key, "")
        print(f"  {label:<26}{report[key]:>6}%   {note}")

    lat = report["latency_ms"]
    p50, p95, mx = lat["p50"], lat["p95"], lat["max"]
    print()
    print(f"  {'selection latency':<26}p50 {p50}ms   p95 {p95}ms   max {mx}ms")
    if "stability" in report:
        note = "identical across all attempts"
        print(f"  {'stability':<26}{report['stability']:>6}%   {note}")
    print()

    labels = registry.keys() + [ABSTAIN]
    width = max(len(label) for label in labels) + 2
    print("  CONFUSION  (rows = expected, columns = predicted)")
    print("  " + " " * width + "".join(label[:9].rjust(11) for label in labels))
    for label in labels:
        row = report["confusion"].get(label, {})
        cells = "".join(str(row.get(pred, 0)).rjust(11) for pred in labels)
        print("  " + label.ljust(width) + cells)
    print()

    if report["misses"]:
        print(f"  MISSES ({len(report['misses'])})")
        for miss in report["misses"][:15]:
            print(f"    {miss['id']:<8} expected {miss['expected']}  got {miss['got']}")
            print(f"             {miss['prompt'][:66]}")
        if len(report["misses"]) > 15:
            print(f"    ... and {len(report['misses']) - 15} more")
        print()
    else:
        print("  no misses")
        print()


def render_comparison(reports: dict[str, dict]) -> None:
    """Side by side. The only table that answers 'did you need all this?'."""
    router, monolith = reports["router"], reports["monolith"]

    print()
    print("=" * 74)
    print(
        f"  ARCHITECTURE COMPARISON - {router['cases']} cases "
        f"x{router['repeat']} = {router['runs']} runs per arm"
    )
    print("=" * 74)
    print()
    print(f"  {'':<26}{'router':>10}{'monolith':>12}{'delta':>10}")
    print(f"  {'':<26}{'-' * 8:>10}{'-' * 8:>12}{'-' * 8:>10}")

    for label, key, _ in METRICS:
        if key is None:
            print()
            continue
        a, b = router[key], monolith[key]
        print(f"  {label:<26}{a:>9}%{b:>11}%{a - b:>+10.1f}")

    a_lat = router["latency_ms"]["p50"]
    b_lat = monolith["latency_ms"]["p50"]
    print()
    print(f"  {'selection latency p50':<26}{a_lat:>8}ms{b_lat:>10}ms{a_lat - b_lat:>+10.1f}")
    if "stability" in router:
        a_s, b_s = router["stability"], monolith["stability"]
        print(f"  {'stability':<26}{a_s:>9}%{b_s:>11}%{a_s - b_s:>+10.1f}")
    print()
    print("  Positive delta = the router arm is ahead on that metric.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=1, help="runs per case")
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument(
        "--arm",
        choices=[*ARMS, "both"],
        default="router",
        help="which architecture to evaluate (default: router)",
    )
    parser.add_argument(
        "--min-single",
        type=float,
        default=90.0,
        help="fail the run below this single-agent accuracy",
    )
    args = parser.parse_args()

    rows = load(DATASET)
    provider = build_provider()
    model = getattr(provider, "name", "unknown")
    wanted = ARMS if args.arm == "both" else (args.arm,)

    reports: dict[str, dict] = {}
    for name in wanted:
        arm = build_arm(name, provider, registry)
        result = asyncio.run(run_arm(arm, rows, args.repeat))
        reports[name] = score(result, model=model, cases=len(rows), repeat=args.repeat)
        render(reports[name])

    if len(reports) > 1:
        render_comparison(reports)

    if args.json:
        # A single arm writes its report at the top level, so the file keeps the
        # shape everything downstream already reads.
        payload = reports[wanted[0]] if len(reports) == 1 else {"arms": reports}
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  full report written to {args.json}\n")

    gate = reports.get("router")
    if gate and gate["single_agent_accuracy"] < args.min_single:
        print(
            f"  FAIL: single-agent accuracy {gate['single_agent_accuracy']}% "
            f"is below the {args.min_single}% gate\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
