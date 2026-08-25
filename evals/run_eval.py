"""Routing evaluation.

Answers one question: given a labelled prompt, does the router select the right
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

Usage:
    uv run python evals/run_eval.py
    uv run python evals/run_eval.py --repeat 3 --json report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from onemind.agents import catalog  # noqa: E402,F401  (registers the roster)
from onemind.bootstrap import build_provider  # noqa: E402
from onemind.orchestrator.registry import registry  # noqa: E402
from onemind.orchestrator.router import Router  # noqa: E402

DATASET = Path(__file__).parent / "datasets" / "routing.jsonl"
ABSTAIN = "(abstain)"


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


async def evaluate(rows: list[dict], repeat: int) -> dict:
    router = Router(build_provider())

    exact = Counter()
    totals = Counter()
    latencies: list[float] = []
    confusion: dict[str, Counter] = defaultdict(Counter)
    misses: list[dict] = []

    tp = fp = fn = 0

    for row in rows:
        expected = set(row["expect"])
        bucket = "abstain" if not expected else "single" if len(expected) == 1 else "multi"

        for attempt in range(repeat):
            started = time.perf_counter()
            decision = await router.route(row["prompt"])
            latencies.append((time.perf_counter() - started) * 1000)
            got = set(decision.agents)

            totals[bucket] += 1
            if got == expected:
                exact[bucket] += 1
            else:
                misses.append(
                    {
                        "id": row["id"],
                        "attempt": attempt,
                        "prompt": row["prompt"],
                        "expected": sorted(expected),
                        "got": sorted(got),
                        "question": decision.clarifying_question,
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
    return {
        "model": getattr(build_provider(), "name", "unknown"),
        "cases": len(rows),
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


def render(report: dict) -> None:
    counts = report["counts"]
    single = f"{counts.get('single', 0)} runs"
    multi = f"{counts.get('multi', 0)} runs"
    abstain = f"{counts.get('abstain', 0)} runs"

    rows = [
        ("single-agent accuracy", report["single_agent_accuracy"], single),
        ("multi-agent exact match", report["multi_agent_exact_match"], multi),
        ("abstain accuracy", report["abstain_accuracy"], abstain),
        ("overall exact match", report["overall_exact_match"], ""),
        ("", None, ""),
        ("label precision", report["label_precision"], "woken unnecessarily"),
        ("label recall", report["label_recall"], "that should have run"),
    ]

    header = (
        f"  ROUTING EVALUATION - {report['cases']} cases "
        f"x{report['repeat']} = {report['runs']} runs"
    )
    print()
    print("=" * 74)
    print(header)
    print("=" * 74)
    print()

    for label, value, note in rows:
        if value is None:
            print()
            continue
        print(f"  {label:<26}{value:>6}%   {note}")

    lat = report["latency_ms"]
    print()
    print(f"  {'router latency':<26}p50 {lat['p50']}ms   p95 {lat['p95']}ms   max {lat['max']}ms")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=1, help="runs per case")
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument(
        "--min-single",
        type=float,
        default=90.0,
        help="fail the run below this single-agent accuracy",
    )
    args = parser.parse_args()

    rows = load(DATASET)
    report = asyncio.run(evaluate(rows, args.repeat))
    render(report)

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  full report written to {args.json}\n")

    if report["single_agent_accuracy"] < args.min_single:
        print(
            f"  FAIL: single-agent accuracy {report['single_agent_accuracy']}% "
            f"is below the {args.min_single}% gate\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
