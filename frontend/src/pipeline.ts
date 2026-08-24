/**
 * Projects the raw span stream onto the nine stages the README draws.
 *
 * The trace is a flat list of spans in wall-clock order; the pipeline is the
 * shape of the system. Keeping the projection here - pure, no React - means
 * the developer view renders a model it did not have to infer inline, and the
 * one place that knows "a second wave is memory spans named `Second wave: ...`
 * plus the agent spans they caused" is a single function.
 *
 * A stage with no spans is `skipped`, not `pending`, once the run has ended.
 * That distinction is the point of the view: "synthesis did not run" is a real
 * fact about a single-specialist request, not missing data.
 */

import type { Span, SpanKind } from "./api";

export type StageStatus = "pending" | "running" | "ok" | "error" | "skipped";

export interface Stage {
  id: string;
  label: string;
  sub: string;
  /** Drives the accent colour; matches the span kind legend. */
  kind: SpanKind;
  status: StageStatus;
  /** Wall-clock envelope of the stage, not the sum of its parts. */
  durationMs: number;
  /** One short line of evidence, e.g. "3 calls · 1 guardrail stop". */
  metric: string;
  spans: Span[];
}

const isRedactIn = (s: Span) => s.kind === "guardrail" && s.name.startsWith("PHI redaction");
const isRedactOut = (s: Span) => s.kind === "guardrail" && s.name.startsWith("PHI re-hydration");

/** Everything the guardrail refused or flagged mid-flight. */
export const isGuardBlock = (s: Span) =>
  s.kind === "guardrail" && !isRedactIn(s) && !isRedactOut(s);

const isFacts = (s: Span) => s.kind === "memory" && s.name.startsWith("Facts");
const isSecondWave = (s: Span) => s.kind === "memory" && s.name.startsWith("Second wave");
const waveOf = (s: Span) => Number(s.detail.wave ?? 1);

function num(value: unknown, fallback = 0): number {
  return typeof value === "number" ? value : fallback;
}

function list(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

function plural(n: number, one: string, many = one + "s"): string {
  return n + " " + (n === 1 ? one : many);
}

/** Wall-clock envelope of a group, so parallel agents read as one stage. */
function envelope(spans: Span[]): number {
  if (spans.length === 0) return 0;
  let start = Infinity;
  let end = 0;
  for (const s of spans) {
    start = Math.min(start, s.offset_ms);
    end = Math.max(end, s.offset_ms + (s.duration_ms ?? 0));
  }
  return Math.max(0, end - start);
}

function sum(spans: Span[]): number {
  return spans.reduce((acc, s) => acc + (s.duration_ms ?? 0), 0);
}

function statusOf(spans: Span[], ended: boolean): StageStatus {
  if (spans.length === 0) return ended ? "skipped" : "pending";
  if (spans.some((s) => s.status === "running")) return "running";
  if (spans.some((s) => s.status === "error")) return "error";
  return "ok";
}

export interface Pipeline {
  stages: Stage[];
  /** Longest wall-clock point reached by any span; the timeline's full width. */
  totalMs: number;
  blocks: Span[];
}

export function buildPipeline(spans: Span[], ended: boolean): Pipeline {
  const redactIn = spans.filter(isRedactIn);
  const redactOut = spans.filter(isRedactOut);
  const blocks = spans.filter(isGuardBlock);
  const route = spans.filter((s) => s.kind === "route");
  const agents = spans.filter((s) => s.kind === "agent");
  const wave1 = agents.filter((s) => waveOf(s) <= 1);
  const later = agents.filter((s) => waveOf(s) > 1);
  const tools = spans.filter((s) => s.kind === "tool");
  const facts = spans.filter(isFacts);
  const reconcile = spans.filter((s) => s.kind === "reconcile");
  const retries = spans.filter(isSecondWave);
  const synth = spans.filter((s) => s.kind === "synthesize");

  const lastRoute = route.at(-1);
  const routedAgents = list(lastRoute?.detail.agents);
  const actionable = lastRoute?.detail.is_actionable !== false;

  const lastFacts = facts.at(-1);
  const newFacts = list(lastFacts?.detail.new);
  const knownFacts = Object.keys(
    (lastFacts?.detail.known as Record<string, string> | undefined) ?? {},
  );

  const verdicts = reconcile.flatMap((s) => list(s.detail.verdicts));
  const mismatches = verdicts.filter((v) => v === "mismatch").length;

  const inbound = num(redactIn.at(-1)?.detail.redacted_count);
  const inboundKinds = list(redactIn.at(-1)?.detail.kinds);

  const dispatchWall = envelope(wave1);
  const dispatchSerial = sum(wave1);

  const stages: Stage[] = [
    {
      id: "redact_in",
      label: "PHI redaction",
      sub: "inbound · before the model sees anything",
      kind: "guardrail",
      status: statusOf(redactIn, ended),
      durationMs: envelope(redactIn),
      metric: inbound
        ? plural(inbound, "identifier") +
          " masked" +
          (inboundKinds.length ? " · " + inboundKinds.join(", ") : "")
        : "nothing matched",
      spans: redactIn,
    },
    {
      id: "route",
      label: "Router",
      sub: "which planes own this request?",
      kind: "route",
      status: statusOf(route, ended),
      durationMs: envelope(route),
      metric: !actionable
        ? "not answerable → clarifying question"
        : routedAgents.length
          ? routedAgents.join(", ")
          : "—",
      spans: route,
    },
    {
      id: "dispatch",
      label: "Parallel dispatch",
      sub: "wave 1 · specialists run concurrently",
      kind: "agent",
      status: statusOf(wave1, ended),
      durationMs: dispatchWall,
      metric: wave1.length
        ? plural(wave1.length, "specialist") +
          " · " +
          dispatchWall +
          " ms wall clock vs " +
          dispatchSerial +
          " ms serial"
        : "no specialist ran",
      spans: wave1,
    },
    {
      id: "tools",
      label: "Tool calls",
      sub: "grounded lookups against the data planes",
      kind: "tool",
      status: blocks.length ? "error" : statusOf(tools, ended),
      durationMs: envelope(tools),
      metric: [
        tools.length ? plural(tools.length, "call") : "no calls",
        blocks.length ? plural(blocks.length, "guardrail stop") : null,
      ]
        .filter(Boolean)
        .join(" · "),
      spans: [...tools, ...blocks].sort((a, b) => a.offset_ms - b.offset_ms),
    },
    {
      id: "facts",
      label: "Facts",
      sub: "identifiers harvested into session memory",
      kind: "memory",
      status: statusOf(facts, ended),
      durationMs: envelope(facts),
      metric: newFacts.length
        ? plural(newFacts.length, "new fact") + ": " + newFacts.join(", ")
        : knownFacts.length
          ? "nothing new · " + plural(knownFacts.length, "fact") + " carried"
          : "nothing to harvest",
      spans: facts,
    },
    {
      id: "reconcile",
      label: "Reconcile",
      sub: "cross-plane comparisons computed in code",
      kind: "reconcile",
      status: statusOf(reconcile, ended),
      durationMs: envelope(reconcile),
      metric: verdicts.length
        ? plural(verdicts.length, "check") + (mismatches ? " · " + mismatches + " mismatch" : "")
        : "nothing comparable",
      spans: reconcile,
    },
    {
      id: "wave2",
      label: "Second wave",
      sub: "specialists a sibling unblocked",
      kind: "memory",
      status: statusOf([...retries, ...later], ended),
      durationMs: envelope([...retries, ...later]),
      metric: retries.length
        ? retries.map((s) => String(s.detail.agent ?? s.name)).join(", ")
        : "not triggered",
      spans: [...retries, ...later].sort((a, b) => a.offset_ms - b.offset_ms),
    },
    {
      id: "synthesize",
      label: "Synthesize",
      sub: "merge + attribute",
      kind: "synthesize",
      status: statusOf(synth, ended),
      durationMs: envelope(synth),
      metric: synth.length
        ? num(synth.at(-1)?.detail.answer_chars) + " chars merged"
        : "skipped · a single specialist answered",
      spans: synth,
    },
    {
      id: "redact_out",
      label: "PHI re-hydration",
      sub: "outbound · placeholders become real values",
      kind: "guardrail",
      status: statusOf(redactOut, ended),
      durationMs: envelope(redactOut),
      metric: redactOut.length
        ? plural(num(redactOut.at(-1)?.detail.restored), "value") + " restored"
        : "nothing to restore",
      spans: redactOut,
    },
  ];

  let totalMs = 1;
  for (const s of spans) totalMs = Math.max(totalMs, s.offset_ms + (s.duration_ms ?? 0));

  return { stages, totalMs, blocks };
}

/**
 * Depth of each span in the parent chain, so the waterfall can indent tool
 * calls under the agent that made them without a recursive render.
 */
export function depths(spans: Span[]): Map<string, number> {
  const byId = new Map(spans.map((s) => [s.span_id, s]));
  const out = new Map<string, number>();
  for (const span of spans) {
    let depth = 0;
    let cursor: Span | undefined = span;
    // Bounded by the real nesting (agent → tool); the guard is for cycles.
    while (cursor?.parent_id && depth < 8) {
      cursor = byId.get(cursor.parent_id);
      if (!cursor) break;
      depth += 1;
    }
    out.set(span.span_id, depth);
  }
  return out;
}
