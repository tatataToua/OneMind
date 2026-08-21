"""Edge-case probe set.

Thirty hand-picked prompts, each targeting one specific way this system could
plausibly fail - grounded either in a correction already recorded in
`docs/decisions.md` (turned into a permanent regression check) or in a code
path that has never been exercised live (identity resolution, out-of-scope
requests, prompt injection).

This is deliberately not scored like `run_eval.py`. Routing accuracy is a
distribution worth gating on; most of what these 30 cases probe is not -
whether an answer states a gap instead of a false conflict, or refuses to
hallucinate a booked appointment, is a judgment call a human has to read to
verify. So each case gets one live run and a full transcript, not a pass/fail
bit, with two exceptions that ARE cheap and reliable to automate:

  routing hint   when the case names an expected `is_actionable`/`agents`,
                  flag a mismatch - a hint for review, not a hard failure,
                  since these are exploratory cases rather than a curated
                  golden set
  phi leak        every case, not just the phi_adversarial ones, is scanned
                  with the same detector `phi_leak.py` uses. A prompt about
                  claims or telemetry has no reason to leak a patient
                  identifier either, so this is free additional coverage

Output is two files: a JSON transcript (`--json`) for tooling, and a Markdown
report (`--md`) meant to be read - grouped by category, one case per entry,
written for someone deciding whether to trust this system with real data.

Usage:
    uv run python evals/edge_cases.py --json evals/edge_cases_report.json --md docs/edge-cases.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from onemind.bootstrap import build_orchestrator  # noqa: E402
from onemind.observability.trace import Trace  # noqa: E402

from phi_leak import _PLACEHOLDER, find_leaks, known_identifiers  # noqa: E402

DATASET = Path(__file__).parent / "datasets" / "edge_cases.jsonl"

CATEGORY_TITLES = {
    "abstain": "Abstain on ambiguity",
    "missing_parameter": "Missing parameters",
    "multi_specialist": "Multi-specialist fan-out",
    "phi_adversarial": "PHI adversarial formats",
    "threshold_direction": "Threshold direction",
    "arithmetic": "Arithmetic in tools",
    "attribution": "Attribution and gap-vs-conflict",
    "policy_retrieval": "Policy retrieval",
    "out_of_scope": "Out-of-scope requests",
    "prompt_injection": "Prompt injection",
    "identity_resolution": "Identity resolution",
    "formatting_stress": "Formatting stress",
    "full_fanout": "Full fan-out",
}


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


async def run_case(orchestrator, identifiers, row: dict) -> dict:
    trace = Trace()
    outcome = await orchestrator.run(row["prompt"], trace)

    got_agents = outcome.get("agents", [])
    got_actionable = outcome.get("is_actionable")

    routing_hint = None
    if row.get("expect_actionable") is not None and got_actionable != row["expect_actionable"]:
        routing_hint = f"expected is_actionable={row['expect_actionable']}, got {got_actionable}"
    elif row.get("expect_agents") is not None and set(got_agents) != set(row["expect_agents"]):
        routing_hint = f"expected agents {row['expect_agents']}, got {got_agents}"

    inbound_hits = find_leaks(outcome.get("redacted_request", ""), identifiers)
    audit_hits = find_leaks(json.dumps(outcome.get("trace", {}), default=str), identifiers)
    token_hits = _PLACEHOLDER.findall(outcome.get("answer", ""))

    return {
        "id": row["id"],
        "category": row["category"],
        "prompt": row["prompt"],
        "watch_for": row["watch_for"],
        "is_actionable": got_actionable,
        "agents": got_agents,
        "clarifying_question": outcome.get("clarifying_question", ""),
        "answer": outcome.get("answer", ""),
        "citations": outcome.get("citations", []),
        "routing_hint": routing_hint,
        "phi_leak": bool(inbound_hits or audit_hits or token_hits),
        "phi_leak_detail": {
            "inbound": inbound_hits,
            "audit": audit_hits,
            "token": len(token_hits),
        },
    }


async def evaluate(rows: list[dict]) -> dict:
    orchestrator = build_orchestrator()
    identifiers = known_identifiers()

    results = [await run_case(orchestrator, identifiers, row) for row in rows]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": len(results),
        "phi_leaks": sum(1 for r in results if r["phi_leak"]),
        "routing_hints": sum(1 for r in results if r["routing_hint"]),
        "results": results,
    }


def render_console(report: dict) -> None:
    print()
    print("=" * 74)
    print(f"  EDGE CASE PROBE SET - {report['cases']} cases, live model")
    print("=" * 74)
    print()
    print(f"  phi leaks       {report['phi_leaks']}")
    print(f"  routing hints   {report['routing_hints']}  (exploratory, not hard failures)")
    print()
    if report["phi_leaks"]:
        print("  PHI LEAKS")
        for r in report["results"]:
            if r["phi_leak"]:
                print(f"    {r['id']:<10} {r['category']}: {r['phi_leak_detail']}")
        print()
    if report["routing_hints"]:
        print("  ROUTING HINTS")
        for r in report["results"]:
            if r["routing_hint"]:
                print(f"    {r['id']:<10} {r['routing_hint']}")
        print()


def render_markdown(report: dict) -> str:
    lines = [
        "# Edge case probe set",
        "",
        f"Generated {report['generated_at']}, {report['cases']} cases against the live model.",
        "",
        "Each case targets one specific way this system could fail - most tied directly to a",
        "correction already recorded in [`decisions.md`](decisions.md), turned into a permanent,",
        "re-runnable check. Regenerate with `uv run python evals/edge_cases.py`.",
        "",
        f"**PHI leaks: {report['phi_leaks']}/{report['cases']}.** Every case is scanned with the",
        "same detector as `evals/phi_leak.py`, not just the PHI-adversarial category - a claims or",
        "telemetry question has no reason to leak a patient identifier either.",
        "",
        f"**Routing hints: {report['routing_hints']}/{report['cases']}.** These are exploratory",
        "cases, not a curated golden set like `evals/datasets/routing.jsonl`, so a hint here is a",
        "prompt to go read the transcript, not a gate failure.",
        "",
        "---",
        "",
    ]

    by_category: dict[str, list[dict]] = {}
    for r in report["results"]:
        by_category.setdefault(r["category"], []).append(r)

    for category, rows in by_category.items():
        title = CATEGORY_TITLES.get(category, category)
        lines.append(f"## {title}")
        lines.append("")
        for r in rows:
            flags = []
            if r["phi_leak"]:
                flags.append("**PHI LEAK**")
            if r["routing_hint"]:
                flags.append("routing hint")
            flag_str = f" — {', '.join(flags)}" if flags else ""

            lines.append(f"### `{r['id']}`{flag_str}")
            lines.append("")
            lines.append(f"**Prompt:** {r['prompt']}")
            lines.append("")
            lines.append(f"**Watching for:** {r['watch_for']}")
            lines.append("")
            lines.append(
                f"**Routed:** actionable={r['is_actionable']}, agents={r['agents'] or '(none)'}"
            )
            if r["routing_hint"]:
                lines.append(f" ⚠️ {r['routing_hint']}")
            lines.append("")
            if r["clarifying_question"]:
                lines.append(f"**Clarifying question:** {r['clarifying_question']}")
                lines.append("")
            if r["answer"]:
                lines.append("**Answer:**")
                lines.append("")
                lines.append("> " + r["answer"].replace("\n", "\n> "))
                lines.append("")
            if r["citations"]:
                lines.append(f"**Citations:** {', '.join(r['citations'])}")
                lines.append("")
            if r["phi_leak"]:
                lines.append(f"**PHI leak detail:** `{r['phi_leak_detail']}`")
                lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the full JSON transcript here")
    parser.add_argument("--md", type=Path, help="write the Markdown report here")
    args = parser.parse_args()

    rows = load(DATASET)
    report = asyncio.run(evaluate(rows))
    render_console(report)

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  transcript written to {args.json}\n")

    if args.md:
        args.md.write_text(render_markdown(report), encoding="utf-8")
        print(f"  report written to {args.md}\n")

    return 1 if report["phi_leaks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
