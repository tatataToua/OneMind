/**
 * The orchestration graph, drawn.
 *
 * This is the same graph `orchestrator/graph.py` compiles - the gate, the
 * router, the `Send` fan-out, the fan-in, the blackboard, the second wave - laid
 * out once and lit from the live span stream. A node glows while its spans are
 * running, so the picture is the progress indicator rather than a spinner
 * beside one.
 *
 * ## Why the layout is hand-placed
 *
 * The graph's shape is fixed in Python: there is no state in which the router
 * feeds anything but the fan-out, and no run that grows a node. Only the
 * *columns* vary, with the specialist roster from `/api/agents`. So the
 * coordinates below are constants and the columns are computed from a count -
 * no auto-layout pass, no measuring the DOM, nothing that can reflow into a
 * different diagram than the one being explained.
 *
 * Nodes are real HTML buttons on an absolute grid; the edges are one SVG layer
 * behind them using the identical pixel coordinates. That split is the whole
 * trick: text wraps, truncates and takes focus the way text should, while the
 * arrows get to be arrows. A fixed canvas in a scroll container keeps the two
 * coordinate systems the same at every viewport width.
 */

import { useState } from "react";
import type { FlowGraph as Graph, FlowNode, NodeStatus } from "../graph";
import { edgeState, hasRun } from "../graph";
import { IconAlert, IconChevron } from "../icons";

// ---------------------------------------------------------------- geometry

const W = 940;
const H = 900;

/** Specialist columns: fixed width, centred as a block, count-driven. */
const CARD = { w: 192, gap: 18, y: 316, h: 124 };

const BUS_OUT = 296; // `Send` fan-out rail
const BUS_IN = 470; // fan-in rail, where `operator.add` merges the results
const LANE_L = 32; // return lane: the blackboard back into the fan-out
const LANE_R = 908; // bypass lane: a clarifying question skips every specialist

/** The trust boundary, drawn through the middle of both gate nodes. */
const SAFE = { x: 8, y: 116, w: 924, h: 654 };

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

const BOX: Record<string, Box> = {
  request: { x: 330, y: 0, w: 280, h: 52 },
  redact_in: { x: 250, y: 76, w: 440, h: 80 },
  route: { x: 310, y: 186, w: 320, h: 84 },
  clarify: { x: 690, y: 190, w: 200, h: 92 },
  facts: { x: 40, y: 502, w: 200, h: 84 },
  reconcile: { x: 290, y: 498, w: 360, h: 92 },
  synthesize: { x: 310, y: 618, w: 320, h: 84 },
  redact_out: { x: 250, y: 730, w: 440, h: 80 },
  answer: { x: 330, y: 838, w: 280, h: 52 },
};

function cardBox(i: number, n: number): Box {
  const span = n * CARD.w + Math.max(0, n - 1) * CARD.gap;
  return { x: (W - span) / 2 + i * (CARD.w + CARD.gap), y: CARD.y, w: CARD.w, h: CARD.h };
}

const cx = (b: Box) => b.x + b.w / 2;

// ------------------------------------------------------------------- edges

interface Edge {
  id: string;
  /** Endpoints, used only to decide how the edge reads. */
  from: string;
  to: string;
  d: string;
  label?: string;
  /** Where the label sits, in canvas coordinates. */
  at?: [number, number];
  /** Degrees, for the two labels that ride a vertical lane. */
  turn?: number;
  dashed?: boolean;
  /** A rail carries no arrowhead - the stubs off it do. */
  rail?: boolean;
  /**
   * For an edge whose endpoints both run on every request even when the edge
   * itself is never traversed. Without it the second-wave lane would read as
   * taken on any run that reached the blackboard, which is every run.
   */
  taken?: boolean;
}

function buildEdges(specialists: FlowNode[], handoffs: number): Edge[] {
  const n = specialists.length;
  const boxes = specialists.map((_, i) => cardBox(i, n));
  const first = boxes.length ? cx(boxes[0]) : W / 2;
  const last = boxes.length ? cx(boxes[boxes.length - 1]) : W / 2;

  const edges: Edge[] = [
    { id: "in", from: "request", to: "redact_in", d: "M470,52 L470,72" },
    { id: "gate-route", from: "redact_in", to: "route", d: "M470,156 L470,182" },
    { id: "clarify", from: "route", to: "clarify", d: "M630,229 L686,229", dashed: true },
    {
      id: "bypass",
      from: "clarify",
      to: "redact_out",
      d: `M890,229 H${LANE_R} V770 H694`,
      label: "not answerable · asks back, no plane is touched",
      at: [LANE_R + 12, 500],
      turn: 90,
      dashed: true,
    },
    {
      id: "fanout",
      from: "route",
      to: "route",
      d: `M470,270 L470,${BUS_OUT}`,
      label: "Send · one node invocation per specialist",
      at: [470, 288],
    },
    { id: "rail-out", from: "route", to: "route", d: `M${first},${BUS_OUT} H${last}`, rail: true },
    {
      id: "rail-in",
      from: "reconcile",
      to: "reconcile",
      d: `M${first},${BUS_IN} H${last}`,
      rail: true,
    },
    {
      id: "fanin",
      from: "reconcile",
      to: "reconcile",
      d: `M470,${BUS_IN} L470,494`,
      label: "fan-in · results merge through operator.add",
      at: [470, 488],
    },
    {
      id: "harvest",
      from: "reconcile",
      to: "facts",
      d: "M290,544 L244,544",
      label: "harvest",
      at: [265, 534],
    },
    {
      id: "wave2",
      from: "facts",
      to: "route",
      d: `M40,544 H${LANE_L} V${BUS_OUT} H${first - 6}`,
      label: "second wave · needs ∩ new facts",
      at: [LANE_L - 12, 420],
      turn: -90,
      taken: handoffs > 0,
    },
    { id: "synth", from: "reconcile", to: "synthesize", d: "M470,590 L470,614" },
    { id: "out", from: "synthesize", to: "redact_out", d: "M470,702 L470,726" },
    { id: "answer", from: "redact_out", to: "answer", d: "M470,810 L470,834" },
  ];

  for (const [i, box] of boxes.entries()) {
    const key = specialists[i].id;
    const x = cx(box);
    edges.push({ id: "out:" + key, from: "route", to: key, d: `M${x},${BUS_OUT} L${x},${box.y - 4}` });
    edges.push({
      id: "in:" + key,
      from: key,
      to: "reconcile",
      d: `M${x},${box.y + box.h} L${x},${BUS_IN}`,
    });
  }

  return edges;
}

const stateOf = (edge: Edge, graph: Graph) =>
  edge.taken === false ? "idle" : edgeState(graph.byId.get(edge.from), graph.byId.get(edge.to));

// ------------------------------------------------------------------ render

const TAG: Partial<Record<NodeStatus, string>> = {
  running: "live",
  pending: "waiting",
  skipped: "not taken",
  error: "flagged",
};

/** A specialist is not "not taken" - it is a plane the router passed over. */
const AGENT_TAG: Partial<Record<NodeStatus, string>> = { ...TAG, skipped: "not routed" };

export default function FlowGraph({
  graph,
  onSelectSpan,
  selectedSpanId,
}: {
  graph: Graph;
  onSelectSpan(id: string): void;
  selectedSpanId: string | null;
}) {
  const [openId, setOpenId] = useState<string | null>(null);
  const edges = buildEdges(graph.specialists, graph.handoffs.length);
  const open = openId ? graph.byId.get(openId) : undefined;

  // Execution order, not insertion order: tab order and the screen-reader list
  // both have to walk the graph the way the graph runs.
  const ordered = [
    "request",
    "redact_in",
    "route",
    "clarify",
    ...graph.specialists.map((s) => s.id),
    "reconcile",
    "facts",
    "synthesize",
    "redact_out",
    "answer",
  ]
    .map((id) => graph.byId.get(id))
    .filter((n): n is FlowNode => n !== undefined);

  const boxOf = (node: FlowNode): Box => {
    const i = graph.specialists.indexOf(node);
    return i >= 0 ? cardBox(i, graph.specialists.length) : BOX[node.id];
  };

  return (
    <div className="graph-wrap">
      <div className="graph-scroll">
        <div className="graph" style={{ width: W, height: H }}>
          <svg className="edges" width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden>
            <defs>
              {["idle", "done", "live"].map((state) => (
                <marker
                  key={state}
                  id={"arw-" + state}
                  markerWidth="9"
                  markerHeight="9"
                  refX="7.5"
                  refY="3"
                  orient="auto"
                  markerUnits="userSpaceOnUse"
                >
                  <path d="M0,0 L8,3 L0,6 z" />
                </marker>
              ))}
            </defs>

            {/* Everything inside this frame has been through redaction. */}
            <rect
              className="safe"
              x={SAFE.x}
              y={SAFE.y}
              width={SAFE.w}
              height={SAFE.h}
              rx="14"
            />
            <text className="safe-label" x={SAFE.x + 18} y={SAFE.y + 20}>
              REDACTION SPACE
            </text>

            {edges.map((edge) => {
              const state = stateOf(edge, graph);
              return (
                <path
                  key={edge.id}
                  d={edge.d}
                  className={"edge is-" + state + (edge.dashed ? " is-dashed" : "")}
                  markerEnd={edge.rail ? undefined : `url(#arw-${state})`}
                />
              );
            })}

            {edges
              .filter((e) => e.label && e.at)
              .map((edge) => {
                const state = stateOf(edge, graph);
                const [x, y] = edge.at as [number, number];
                return (
                  <text
                    key={edge.id}
                    className={"edge-label is-" + state}
                    x={x}
                    y={y}
                    textAnchor="middle"
                    transform={edge.turn ? `rotate(${edge.turn} ${x} ${y})` : undefined}
                  >
                    {edge.label}
                  </text>
                );
              })}
          </svg>

          {ordered.map((node) => {
            const box = boxOf(node);
            if (!box) return null;
            return (
              <Node
                key={node.id}
                node={node}
                box={box}
                open={openId === node.id}
                onToggle={() => setOpenId(openId === node.id ? null : node.id)}
              />
            );
          })}
        </div>
      </div>

      <ol className="sr-only">
        {ordered.map((node) => (
          <li key={node.id}>
            {node.label} — {node.status}. {node.metric}
          </li>
        ))}
      </ol>

      <div className="graph-key">
        <span>
          <i className="swatch is-live" aria-hidden />
          <span>A glowing box has an open span right now.</span>
        </span>
        <span>
          <i className="swatch is-dim" aria-hidden />
          <span>A dim box is a path this request did not take.</span>
        </span>
        <span>
          <i className="swatch is-safe" aria-hidden />
          <span>
            Inside the dashed frame every identifier is a placeholder like{" "}
            <code>PHI_PATIENT_1</code>.
          </span>
        </span>
        <span>
          <i className="swatch is-memory" aria-hidden />
          <span>
            Specialists never address each other. One writes an identifier to the board through the
            orchestrator; the orchestrator re-dispatches whoever that unblocks.
          </span>
        </span>
      </div>

      {graph.handoffs.length > 0 && (
        <p className="graph-handoff">
          <IconAlert size={14} />
          {graph.handoffs.map((h) => (
            <span key={h.agent}>
              <code>{h.source}</code> established <code>{h.keys.join(", ")}</code>, so the
              orchestrator re-dispatched <strong>{h.agent}</strong>. The two specialists never
              addressed each other.
            </span>
          ))}
        </p>
      )}

      {open && (
        <div className="graph-drill">
          <div className="graph-drill-head">
            <IconChevron size={12} />
            <strong>{open.label}</strong>
            <span className="fine">{open.sub}</span>
            {hasRun(open) && <span className="tabular fine">{open.durationMs} ms</span>}
          </div>
          <p className="step-metric">{open.metric}</p>
          {open.spans.length > 0 ? (
            <ul className="step-spans">
              {open.spans.map((s) => (
                <li key={s.span_id}>
                  <button
                    type="button"
                    className={
                      "step-span st-" +
                      s.status +
                      (s.span_id === selectedSpanId ? " is-selected" : "")
                    }
                    onClick={() => onSelectSpan(s.span_id)}
                  >
                    <span>{s.name}</span>
                    <span className="tabular">{s.duration_ms ?? "…"} ms</span>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="fine">This node emits no span of its own — it is a boundary, not work.</p>
          )}
        </div>
      )}
    </div>
  );
}

function Node({
  node,
  box,
  open,
  onToggle,
}: {
  node: FlowNode;
  box: Box;
  open: boolean;
  onToggle(): void;
}) {
  const plane = node.agent ? " plane-" + node.agent.key : "";
  // On a specialist the tag rides the sub row: "Remote Monitoring" plus a badge
  // does not fit one 192px line, and the name is the half that must survive.
  const tag = (node.agent ? AGENT_TAG : TAG)[node.status];
  // No elapsed figure while a node is open: the only duration available mid-run
  // is the envelope of its *finished* children, which reads as a total and is
  // not one. The tag says "live"; the timeline below says when.
  const ms =
    node.durationMs > 0 && node.status !== "running" ? (
      <span className="gnode-ms tabular">{node.durationMs} ms</span>
    ) : null;
  return (
    <button
      type="button"
      className={
        "gnode kind-" + node.role + " st-" + node.status + plane + (open ? " is-open" : "")
      }
      style={{ left: box.x, top: box.y, width: box.w, height: box.h }}
      aria-pressed={open}
      onClick={onToggle}
      title={node.metric || node.sub}
    >
      <span className="gnode-head">
        <span className="gnode-label">{node.label}</span>
        {node.agent ? ms : tag ? <span className="gnode-tag">{tag}</span> : ms}
      </span>
      <span className="gnode-sub">
        {node.sub}
        {node.agent && tag && <span className="gnode-tag">{tag}</span>}
        {node.agent && node.agent.wave > 1 && (
          <span className="gnode-wave">wave {node.agent.wave}</span>
        )}
      </span>
      {node.metric && <span className="gnode-metric">{node.metric}</span>}
      {node.agent && <span className="gnode-plane">{node.agent.plane}</span>}
    </button>
  );
}
