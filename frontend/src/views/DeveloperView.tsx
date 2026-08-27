/**
 * The developer tab: one request, end to end.
 *
 * Three readings of the same trace, most abstract first. The graph says what
 * the architecture is and which of its paths this request took. The waterfall
 * says when, and is the only place the parallelism is visible as overlap rather
 * than as a claim. The inspector says exactly what a single span recorded.
 *
 * Nothing here is synthesised for display - every number comes from a span the
 * backend emitted, because this view doubles as the audit read.
 */

import { useEffect, useMemo, useState } from "react";
import type { Span } from "../api";
import { buildGraph, depths, isGuardBlock } from "../graph";
import { explainGuardrail } from "../guardrails";
import type { GuardrailNote } from "../guardrails";
import type { Orchestrator } from "../useOrchestrator";
import { IconAlert, IconClock, IconCpu, IconFlow } from "../icons";
import FlowGraph from "./FlowGraph";

const KIND_LABEL: Record<string, string> = {
  guardrail: "guardrail",
  route: "router",
  agent: "agent",
  tool: "tool",
  reconcile: "reconcile",
  synthesize: "synthesis",
  memory: "memory",
};

export default function DeveloperView({ o }: { o: Orchestrator }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const graph = useMemo(
    () => buildGraph(o.spans, o.agents, !o.busy),
    [o.spans, o.agents, o.busy],
  );
  const depth = useMemo(() => depths(o.spans), [o.spans]);
  const selected = o.spans.find((s) => s.span_id === selectedId) ?? null;

  // A new run invalidates every span id, so holding the old selection would
  // silently show a span from the previous request.
  useEffect(() => {
    setSelectedId(null);
  }, [o.outcome, o.busy]);

  if (!o.hasRun) {
    return (
      <div className="dev">
        <div className="empty">
          <IconFlow size={22} />
          <h2>No request traced yet</h2>
          <p>
            Send a request from the Console tab. Every box and arrow below is drawn from the spans
            the orchestrator emits while it runs — nothing is reconstructed after the fact.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="dev">
      <div className="dev-main">
        <section className="stats">
          <Stat label="Request" value={o.outcome?.request_id ?? "—"} mono />
          <Stat label="Session" value={o.outcome?.session_id?.slice(0, 12) ?? "—"} mono />
          <Stat label="Wall clock" value={graph.totalMs + " ms"} />
          <Stat label="Spans" value={String(o.spans.length)} />
          <Stat label="Waves" value={String(graph.waves)} />
          <Stat label="Redactions" value={String(o.outcome?.phi_redactions ?? 0)} />
          <Stat
            label="Guardrail stops"
            value={String(graph.blocks.length)}
            tone={graph.blocks.length ? "danger" : undefined}
          />
        </section>

        <section className="panel">
          <header className="panel-head">
            <h2>
              <IconFlow size={15} />
              Orchestration graph
            </h2>
            <p className="fine">
              The graph <code>orchestrator/graph.py</code> compiles, lit from this run. A box glows
              while its spans are open; a dim box is a path this request did not take. Click any
              box for the spans behind it.
            </p>
          </header>

          <FlowGraph
            graph={graph}
            onSelectSpan={setSelectedId}
            selectedSpanId={selectedId}
          />
        </section>

        <section className="panel">
          <header className="panel-head">
            <h2>
              <IconClock size={15} />
              Timeline
            </h2>
            <p className="fine">
              {graph.totalMs} ms end to end. Overlapping bars are genuinely concurrent
              specialists, not a queue.
            </p>
          </header>

          <div className="ruler" aria-hidden>
            {[0, 0.25, 0.5, 0.75, 1].map((f) => (
              <span key={f} style={{ left: f * 100 + "%" }}>
                {Math.round(graph.totalMs * f)}
              </span>
            ))}
          </div>

          <ol className="waterfall">
            {o.spans.map((span) => {
              const width = Math.max(0.5, ((span.duration_ms ?? 0) / graph.totalMs) * 100);
              const left = (span.offset_ms / graph.totalMs) * 100;
              const indent = depth.get(span.span_id) ?? 0;
              return (
                <li key={span.span_id}>
                  <button
                    type="button"
                    className={
                      "row kind-" +
                      span.kind +
                      " st-" +
                      span.status +
                      (span.span_id === selectedId ? " is-selected" : "") +
                      (isGuardBlock(span) ? " is-block" : "")
                    }
                    aria-pressed={span.span_id === selectedId}
                    onClick={() =>
                      setSelectedId(span.span_id === selectedId ? null : span.span_id)
                    }
                  >
                    <span className="row-kind">{KIND_LABEL[span.kind] ?? span.kind}</span>
                    <span className="row-name" style={{ paddingLeft: indent * 14 }}>
                      {span.name}
                    </span>
                    <span className="row-track">
                      <span
                        className="row-bar"
                        style={{ left: left + "%", width: width + "%" }}
                        title={span.offset_ms + " ms → +" + (span.duration_ms ?? 0) + " ms"}
                      />
                    </span>
                    <span className="row-ms tabular">
                      {span.duration_ms === null ? "…" : span.duration_ms + " ms"}
                    </span>
                  </button>
                </li>
              );
            })}
          </ol>

          <div className="legend">
            {Object.entries(KIND_LABEL).map(([kind, label]) => (
              <span key={kind} className={"legend-item kind-" + kind}>
                <i aria-hidden />
                {label}
              </span>
            ))}
          </div>
        </section>

        <section className="panel">
          <header className="panel-head">
            <h2>
              <IconCpu size={15} />
              Payloads
            </h2>
            <p className="fine">What crossed each trust boundary.</p>
          </header>

          <div className="payloads">
            <Payload label="Model-facing request (redacted)" body={o.outcome?.redacted_request} />
            <Payload label="Answer returned to the user (re-hydrated)" body={o.answer} />
          </div>

          <details className="raw">
            <summary>Raw outcome JSON</summary>
            <pre>{o.outcome ? JSON.stringify(o.outcome, null, 2) : "—"}</pre>
          </details>
        </section>
      </div>

      <aside className="inspector" aria-label="Span inspector">
        <div className="inspector-inner">
          <h2>Span inspector</h2>
          {!selected && (
            <p className="fine">
              Select any bar in the timeline, or a span behind a box in the graph, to see exactly
              what it recorded.
            </p>
          )}
          {selected && <SpanDetail span={selected} />}
        </div>
      </aside>
    </div>
  );
}

function Stat({
  label,
  value,
  mono,
  tone,
}: {
  label: string;
  value: string;
  mono?: boolean;
  tone?: string;
}) {
  return (
    <div className={"stat" + (tone ? " tone-" + tone : "")}>
      <span className="stat-label">{label}</span>
      <span className={"stat-value" + (mono ? " mono" : " tabular")}>{value}</span>
    </div>
  );
}

function SpanDetail({ span }: { span: Span }) {
  const entries = Object.entries(span.detail).filter(([, v]) => v !== null && v !== undefined);
  const note = explainGuardrail(span);
  return (
    <div className="detail">
      <div className={"detail-head kind-" + span.kind}>
        <span className="detail-kind">{KIND_LABEL[span.kind] ?? span.kind}</span>
        <strong>{span.name}</strong>
      </div>

      {note ? (
        <GuardrailExplainer note={note} />
      ) : (
        isGuardBlock(span) && (
          // Fallback for a guardrail `guardrails.ts` has no note for. A generic
          // sentence beats a paragraph written for a different check.
          <p className="banner banner-danger tight">
            <IconAlert size={14} />
            The guardrail stopped this before it reached a data plane.
          </p>
        )
      )}

      <dl className="kv">
        <dt>span_id</dt>
        <dd className="mono">{span.span_id}</dd>
        <dt>status</dt>
        <dd>{span.status}</dd>
        <dt>started</dt>
        <dd className="tabular">+{span.offset_ms} ms</dd>
        <dt>duration</dt>
        <dd className="tabular">{span.duration_ms === null ? "running" : span.duration_ms + " ms"}</dd>
        {span.parent_id && (
          <>
            <dt>parent</dt>
            <dd className="mono">{span.parent_id}</dd>
          </>
        )}
      </dl>

      {entries.length > 0 && (
        <>
          <h3>Recorded detail</h3>
          <dl className="kv">
            {entries.map(([key, value]) => (
              <div key={key} className="kv-row">
                <dt>{key}</dt>
                <dd>{renderValue(value)}</dd>
              </div>
            ))}
          </dl>
        </>
      )}
    </div>
  );
}

const ACTION_LABEL: Record<GuardrailNote["action"], string> = {
  blocked: "blocked before any data plane",
  flagged: "reported, nothing blocked",
  applied: "applied to every request",
};

/**
 * Which guardrail this was, and why it fired here.
 *
 * Named, then ruled, then reasoned - in that order, because the reader's first
 * question is "which check is this?" and answering it with a paragraph makes
 * them find the name themselves. The action chip sits in the header rather than
 * at the end for the same reason: whether anything was actually stopped changes
 * how the rest of the panel should be read.
 */
function GuardrailExplainer({ note }: { note: GuardrailNote }) {
  return (
    <section className={"guard-note guard-" + note.action} aria-label="Guardrail explanation">
      <header className="guard-note-head">
        <div>
          <strong>{note.guard}</strong>
          <code className="guard-source">{note.source}</code>
        </div>
        <span className="guard-action">{ACTION_LABEL[note.action]}</span>
      </header>

      <dl className="guard-body">
        <dt>The rule</dt>
        <dd>{inlineCode(note.rule)}</dd>
        <dt>Why it fired</dt>
        <dd>{inlineCode(note.why)}</dd>
        <dt>What happened</dt>
        <dd>{inlineCode(note.effect)}</dd>
      </dl>
    </section>
  );
}

/**
 * Render `backticked` runs as code, leave everything else alone.
 *
 * The notes name real argument and tool names, and a sentence where
 * `patient_id` is set in prose reads as English rather than as the literal
 * string the span recorded. Kept to this one substitution - a general markdown
 * renderer here would be a dependency and a parser for two characters of
 * formatting.
 */
function inlineCode(text: string) {
  return text.split(/`([^`]+)`/g).map((part, i) =>
    i % 2 === 1 ? <code key={i}>{part}</code> : <span key={i}>{part}</span>,
  );
}

/** Span details are free-form JSON, so this stays deliberately shallow. */
function renderValue(value: unknown) {
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="fine">empty</span>;
    return (
      <span className="chips">
        {value.map((v, i) => (
          <code key={i}>{String(v)}</code>
        ))}
      </span>
    );
  }
  if (typeof value === "object" && value !== null) {
    const rows = Object.entries(value as Record<string, unknown>);
    if (rows.length === 0) return <span className="fine">empty</span>;
    return (
      <span className="chips">
        {rows.map(([k, v]) => (
          <code key={k}>
            {k}={String(v)}
          </code>
        ))}
      </span>
    );
  }
  if (typeof value === "boolean") return <span className="mono">{String(value)}</span>;
  return <span>{String(value)}</span>;
}

function Payload({ label, body }: { label: string; body?: string }) {
  return (
    <div className="payload">
      <span className="payload-label">{label}</span>
      <pre>{body || "—"}</pre>
    </div>
  );
}
