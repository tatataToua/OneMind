/**
 * Projects the raw span stream onto the orchestration graph the backend runs.
 *
 * The trace is a flat list of spans in wall-clock order. The graph is the shape
 * of the system: a gate, a router, a fan-out to specialists that each own one
 * data plane, a fan-in that harvests a blackboard, and a gate on the way out.
 * Keeping the projection here - pure, no React - means the diagram renders a
 * model it did not have to infer inline, and the one place that knows "a second
 * wave is a memory span named `Second wave: ...`" is a single function.
 *
 * Two things this deliberately does *not* do. It does not invent nodes: every
 * node below is a node, edge or store in `orchestrator/graph.py`. And it does
 * not hide the parts that did not run - a specialist the router did not pick is
 * drawn dimmed rather than omitted, because "three of the four planes were
 * never touched" is the routing claim the diagram exists to show.
 */

import type { AgentInfo, Span, SpanKind } from "./api";

export type NodeStatus = "pending" | "running" | "ok" | "error" | "skipped";

/** `io` is the two terminators - what the user typed, what they get back. */
export type NodeRole = SpanKind | "io";

export interface FlowNode {
  id: string;
  label: string;
  /** What this node is, in the abstract. Constant across runs. */
  sub: string;
  role: NodeRole;
  status: NodeStatus;
  /**
   * Time this node itself spent, summed over its own spans - not the envelope
   * from its first to its last. Reconcile, the blackboard and any re-dispatched
   * specialist run once per wave, and the gap between two waves belongs to the
   * specialist running in it, not to the node waiting either side.
   */
  durationMs: number;
  /** One short line of evidence from *this* run. */
  metric: string;
  spans: Span[];
  /** Specialist nodes only. */
  agent?: { key: string; plane: string; tools: string[]; wave: number };
}

/** One re-dispatch the trace recorded, and the identifier that caused it. */
export interface Handoff {
  agent: string;
  keys: string[];
  source: string;
}

export interface FlowGraph {
  byId: Map<string, FlowNode>;
  /** Every registered specialist, in registry order - routed or not. */
  specialists: FlowNode[];
  /** Longest wall-clock point reached by any span; the timeline's full width. */
  totalMs: number;
  blocks: Span[];
  handoffs: Handoff[];
  waves: number;
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

/**
 * Time spent in a node's own spans. Callers pass only top-level spans - a
 * specialist's agent span already contains the time of the tools nested under
 * it, so adding the tool spans in would double-count them.
 */
function spent(spans: Span[]): number {
  return spans.reduce((total, s) => total + (s.duration_ms ?? 0), 0);
}

function statusOf(spans: Span[], ended: boolean): NodeStatus {
  if (spans.length === 0) return ended ? "skipped" : "pending";
  if (spans.some((s) => s.status === "running")) return "running";
  if (spans.some((s) => s.status === "error")) return "error";
  return "ok";
}

/** True once the node has produced something, whatever the verdict. */
export const hasRun = (n: FlowNode | undefined) =>
  n !== undefined && (n.status === "ok" || n.status === "error" || n.status === "running");

/**
 * How an edge should read, given its endpoints.
 *
 * `live` while either end is in flight - that is the glow. `done` once both
 * ends have produced spans, which is the only honest way to say an edge carried
 * something: the trace records nodes, not the arrows between them.
 */
export function edgeState(from?: FlowNode, to?: FlowNode): "live" | "done" | "idle" {
  if (from?.status === "running" || to?.status === "running") return "live";
  return hasRun(from) && hasRun(to) ? "done" : "idle";
}

function clip(text: string, max = 78): string {
  return text.length > max ? text.slice(0, max - 1).trimEnd() + "…" : text;
}

export function buildGraph(spans: Span[], agents: AgentInfo[], ended: boolean): FlowGraph {
  const redactIn = spans.filter(isRedactIn);
  const redactOut = spans.filter(isRedactOut);
  const blocks = spans.filter(isGuardBlock);
  const route = spans.filter((s) => s.kind === "route");
  const agentSpans = spans.filter((s) => s.kind === "agent");
  const tools = spans.filter((s) => s.kind === "tool");
  const facts = spans.filter(isFacts);
  const reconcile = spans.filter((s) => s.kind === "reconcile");
  const retries = spans.filter(isSecondWave);
  const synth = spans.filter((s) => s.kind === "synthesize");

  const lastRoute = route.at(-1);
  const routed = list(lastRoute?.detail.agents);
  const actionable = lastRoute?.detail.is_actionable !== false;
  const routeDecided = route.length > 0 && lastRoute?.status !== "running";

  const lastFacts = facts.at(-1);
  // Union across rounds, not the last round's. A second wave means `reconcile`
  // harvests twice, and the round that established the identifier is the first
  // one - reading only the last would report "nothing new" on exactly the runs
  // where a fact did the most work.
  const newFacts = [...new Set(facts.flatMap((s) => list(s.detail.new)))];
  const known = Object.keys((lastFacts?.detail.known as Record<string, string>) ?? {});

  const verdicts = reconcile.flatMap((s) => list(s.detail.verdicts));
  const mismatches = verdicts.filter((v) => v === "mismatch").length;

  const inbound = num(redactIn.at(-1)?.detail.redacted_count);
  const inboundKinds = list(redactIn.at(-1)?.detail.kinds);

  // Guardrail stops and tool calls are children of the agent span that made
  // them, so both can be attributed to the specialist that earned them.
  const parentOf = new Map(spans.map((s) => [s.span_id, s.parent_id]));
  const ownerOf = (span: Span): string | null => {
    let cursor: string | null = span.parent_id;
    for (let i = 0; i < 8 && cursor; i += 1) {
      const owner = agentSpans.find((a) => a.span_id === cursor);
      if (owner) return String(owner.detail.agent ?? "");
      cursor = parentOf.get(cursor) ?? null;
    }
    return null;
  };

  // The roster comes from `/api/agents`, so a newly registered specialist gets
  // a column with no frontend change. If the API never answered, fall back to
  // whatever the trace itself names.
  const roster: AgentInfo[] = agents.length
    ? agents
    : [
        ...new Map(
          agentSpans.map((s): [string, AgentInfo] => {
            const key = String(s.detail.agent ?? s.name);
            return [
              key,
              {
                key,
                display_name: s.name,
                data_plane: String(s.detail.data_plane ?? ""),
                description: "",
                tools: [],
              },
            ];
          }),
        ).values(),
      ];

  const specialists: FlowNode[] = roster.map((info) => {
    const mine = agentSpans.filter((s) => String(s.detail.agent) === info.key);
    const myTools = tools.filter((s) => ownerOf(s) === info.key);
    const myStops = blocks.filter((s) => ownerOf(s) === info.key);
    const called = [...new Set(myTools.map((s) => String(s.detail.tool ?? s.name)))];
    const errored = mine.find((s) => s.status === "error");
    const picked = routed.includes(info.key);

    let metric: string;
    if (mine.length === 0) {
      metric = picked
        ? "dispatched · waiting on the model"
        : routeDecided
          ? "not routed · plane untouched"
          : "waiting on the router";
    } else if (errored) {
      metric = clip(String(errored.detail.error ?? "failed"));
    } else if (called.length === 0) {
      metric = mine.some((s) => s.status === "running")
        ? "planning its tool calls"
        : "no call reached the plane · needs an identifier";
    } else {
      metric = plural(called.length, "call") + " · " + called.join(", ");
    }
    if (myStops.length) metric += " · " + plural(myStops.length, "stop");

    return {
      id: "spec:" + info.key,
      label: info.display_name,
      sub: info.tools.length ? plural(info.tools.length, "tool") : "specialist",
      role: "agent",
      // A specialist the router picked is *waiting* until its span opens, not
      // skipped. Only one the router passed over is a path not taken - and it
      // is only that once the router has actually decided.
      status: mine.length ? statusOf(mine, ended) : picked && !ended ? "pending" : statusOf([], routeDecided),
      durationMs: spent(mine),
      metric,
      spans: [...mine, ...myTools, ...myStops].sort((a, b) => a.offset_ms - b.offset_ms),
      agent: {
        key: info.key,
        plane: info.data_plane,
        tools: info.tools,
        wave: Math.max(1, ...mine.map(waveOf)),
      },
    };
  });

  const nodes: FlowNode[] = [
    {
      id: "request",
      label: "Request",
      sub: "real identifiers, in the clear",
      role: "io",
      status: spans.length ? "ok" : "pending",
      durationMs: 0,
      metric: "",
      spans: [],
    },
    {
      id: "redact_in",
      label: "PHI redaction",
      sub: "inbound gate · runs before any model call",
      role: "guardrail",
      status: statusOf(redactIn, ended),
      durationMs: spent(redactIn),
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
      sub: "one JSON-constrained call · picks the owning planes",
      role: "route",
      status: statusOf(route, ended),
      durationMs: spent(route),
      metric: !actionable ? "is_actionable: false" : routed.length ? routed.join(", ") : "—",
      spans: route,
    },
    {
      id: "clarify",
      label: "Ask back",
      sub: "END · no specialist runs",
      role: "route",
      status: actionable ? (routeDecided || ended ? "skipped" : "pending") : "ok",
      durationMs: 0,
      metric: actionable
        ? "the request was answerable"
        : clip(String(lastRoute?.detail.clarifying_question ?? "clarifying question returned")),
      spans: [],
    },
    {
      id: "reconcile",
      label: "Reconcile",
      sub: "fan-in · cross-plane checks computed in code",
      role: "reconcile",
      status: statusOf(reconcile, ended),
      durationMs: spent(reconcile),
      metric: verdicts.length
        ? plural(verdicts.length, "check") +
          (mismatches ? " · " + mismatches + " mismatch" : "")
        : "nothing comparable",
      spans: reconcile,
    },
    {
      id: "facts",
      label: "Facts board",
      sub: "identifiers only · never a claim",
      role: "memory",
      status: statusOf([...facts, ...retries], ended),
      durationMs: spent(facts),
      metric: newFacts.length
        ? plural(newFacts.length, "new fact") + ": " + newFacts.join(", ")
        : known.length
          ? "nothing new · " + plural(known.length, "fact") + " carried"
          : "nothing to harvest",
      spans: [...facts, ...retries].sort((a, b) => a.offset_ms - b.offset_ms),
    },
    {
      id: "synthesize",
      label: "Synthesize",
      sub: "merge + attribute · streamed, outside the graph",
      role: "synthesize",
      status: statusOf(synth, ended),
      durationMs: spent(synth),
      // `answer_chars` only exists once the span closes, so mid-stream this
      // would otherwise report a confident "0 chars merged".
      metric: !synth.length
        ? "not reached"
        : synth.at(-1)?.status === "running"
          ? "streaming to the client"
          : synth.at(-1)?.detail.synthesised === false
            ? "one specialist answered · passed through"
            : num(synth.at(-1)?.detail.answer_chars) + " chars merged",
      spans: synth,
    },
    {
      id: "redact_out",
      label: "PHI re-hydration",
      sub: "outbound gate · placeholders become values again",
      role: "guardrail",
      status: statusOf(redactOut, ended),
      durationMs: spent(redactOut),
      metric: redactOut.length
        ? plural(num(redactOut.at(-1)?.detail.restored), "value") + " restored"
        : "nothing to restore",
      spans: redactOut,
    },
    {
      id: "answer",
      label: "Answer",
      sub: "cited, re-hydrated, one voice",
      role: "io",
      status: redactOut.length ? "ok" : "pending",
      durationMs: 0,
      metric: "",
      spans: [],
    },
    ...specialists,
  ];

  const handoffs: Handoff[] = retries.map((s) => ({
    agent: String(s.detail.agent ?? ""),
    keys: Object.keys((s.detail.unblocked_by as Record<string, string>) ?? {}),
    source: String(s.detail.source ?? ""),
  }));

  let totalMs = 1;
  for (const s of spans) totalMs = Math.max(totalMs, s.offset_ms + (s.duration_ms ?? 0));

  return {
    byId: new Map(nodes.map((n) => [n.id, n])),
    specialists,
    totalMs,
    blocks,
    handoffs,
    waves: Math.max(1, ...spans.map(waveOf)),
  };
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
