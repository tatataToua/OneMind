import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchAgents,
  fetchExamples,
  streamChat,
  type AgentInfo,
  type ExampleInfo,
  type Outcome,
  type Span,
} from "./api";

type AgentState = "idle" | "running" | "done" | "error";

const KIND_LABEL: Record<string, string> = {
  guardrail: "guardrail",
  route: "router",
  agent: "agent",
  tool: "tool",
  synthesize: "synthesis",
};

export default function App() {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [examples, setExamples] = useState<ExampleInfo[]>([]);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState("");
  const [spans, setSpans] = useState<Span[]>([]);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showRedacted, setShowRedacted] = useState(false);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    fetchAgents().then(setAgents).catch(() => setError("Cannot reach the API. Is the backend running on :8080?"));
    fetchExamples().then(setExamples).catch(() => undefined);
  }, []);

  /** Which agents are lit, derived from the live span stream. */
  const agentState = useMemo(() => {
    const state: Record<string, AgentState> = {};
    for (const span of spans) {
      if (span.kind !== "agent") continue;
      const key = String(span.detail.agent ?? "");
      if (!key) continue;
      state[key] =
        span.status === "running"
          ? "running"
          : span.status === "error"
            ? "error"
            : "done";
    }
    return state;
  }, [spans]);

  const activeTools = useMemo(() => {
    const byAgent: Record<string, string[]> = {};
    const agentSpans = new Map(
      spans.filter((s) => s.kind === "agent").map((s) => [s.span_id, String(s.detail.agent ?? "")]),
    );
    for (const span of spans) {
      if (span.kind !== "tool" || !span.parent_id) continue;
      const key = agentSpans.get(span.parent_id);
      if (!key) continue;
      (byAgent[key] ??= []).push(span.name);
    }
    return byAgent;
  }, [spans]);

  const totalMs = useMemo(() => {
    let max = 1;
    for (const span of spans) max = Math.max(max, span.offset_ms + (span.duration_ms ?? 0));
    return max;
  }, [spans]);

  const submit = useCallback(
    async (text: string) => {
      const message = text.trim();
      if (!message || busy) return;

      abort.current?.abort();
      abort.current = new AbortController();

      setBusy(true);
      setAnswer("");
      setSpans([]);
      setOutcome(null);
      setError(null);
      setShowRedacted(false);

      try {
        await streamChat(
          message,
          {
            onSpanStart: (span) => setSpans((prev) => [...prev, span]),
            // Spans are matched by id, so an end event updates the row the
            // start event created rather than appending a duplicate.
            onSpanEnd: (span) =>
              setSpans((prev) =>
                prev.map((s) =>
                  s.span_id === span.span_id ? { ...s, ...span, offset_ms: s.offset_ms } : s,
                ),
              ),
            onToken: (token) => setAnswer((prev) => prev + token),
            onDone: (result) => {
              setOutcome(result);
              // The streamed tokens are the model's redacted output; the done
              // payload has been through outbound re-hydration.
              setAnswer(result.answer);
              setSpans(result.trace.spans);
            },
            onError: setError,
          },
          abort.current.signal,
        );
      } catch (err) {
        if ((err as Error).name !== "AbortError") setError(String(err));
      } finally {
        setBusy(false);
      }
    },
    [busy],
  );

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="mark" aria-hidden />
          <div>
            <h1>OneMind</h1>
            <p>Healthcare multi-agent orchestrator</p>
          </div>
        </div>
        <div className="badges">
          <span className="badge">qwen3.5:4b</span>
          <span className="badge">local inference</span>
          <span className="badge accent">no PHI leaves this machine</span>
        </div>
      </header>

      <main className="layout">
        <aside className="rail">
          <h2>Specialists</h2>
          <p className="rail-note">
            Each owns one data source no other can reach. That is what makes routing decidable.
          </p>
          {agents.map((agent) => {
            const state = agentState[agent.key] ?? "idle";
            return (
              <article key={agent.key} className={`agent ${state}`}>
                <div className="agent-head">
                  <span className="dot" aria-hidden />
                  <h3>{agent.display_name}</h3>
                </div>
                <p className="plane">{agent.data_plane}</p>
                <div className="tools">
                  {agent.tools.map((tool) => (
                    <span
                      key={tool}
                      className={
                        activeTools[agent.key]?.includes(tool) ? "tool fired" : "tool"
                      }
                    >
                      {tool}
                    </span>
                  ))}
                </div>
              </article>
            );
          })}
        </aside>

        <section className="center">
          <form
            className="composer"
            onSubmit={(event) => {
              event.preventDefault();
              void submit(prompt);
            }}
          >
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void submit(prompt);
                }
              }}
              placeholder="Ask about a patient, a claim, a policy, or a device reading."
              rows={3}
              disabled={busy}
            />
            <button type="submit" disabled={busy || !prompt.trim()}>
              {busy ? "Working" : "Send"}
            </button>
          </form>

          <div className="examples">
            {examples.map((example) => (
              <button
                key={example.label}
                className="chip"
                title={example.expect}
                disabled={busy}
                onClick={() => {
                  setPrompt(example.prompt);
                  void submit(example.prompt);
                }}
              >
                {example.label}
              </button>
            ))}
          </div>

          {error && <div className="error">{error}</div>}

          {(answer || busy) && (
            <div className="answer-card">
              <div className="answer-head">
                <h2>{outcome && !outcome.is_actionable ? "Needs clarification" : "Answer"}</h2>
                {outcome && (
                  <div className="meta">
                    {outcome.agents.map((key) => (
                      <span key={key} className="pill">
                        {agents.find((a) => a.key === key)?.display_name ?? key}
                      </span>
                    ))}
                    {outcome.phi_redactions > 0 && (
                      <button className="pill toggle" onClick={() => setShowRedacted((v) => !v)}>
                        {outcome.phi_redactions} PHI redacted
                      </button>
                    )}
                  </div>
                )}
              </div>

              {showRedacted && outcome && (
                <pre className="redacted">
                  <strong>What the model actually received:</strong>
                  {"\n"}
                  {outcome.redacted_request}
                </pre>
              )}

              <div className="answer-body">
                {answer || <span className="cursor" />}
              </div>

              {outcome?.rationale && (
                <p className="rationale">
                  <strong>Routing rationale:</strong> {outcome.rationale}
                </p>
              )}

              {outcome && outcome.citations.length > 0 && (
                <div className="citations">
                  <strong>Sources</strong>
                  <ul>
                    {outcome.citations.map((citation) => (
                      <li key={citation}>{citation}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {spans.length > 0 && (
            <div className="trace">
              <div className="trace-head">
                <h2>Execution trace</h2>
                <span className="muted">
                  {totalMs} ms total &middot; overlapping bars are concurrent agents
                </span>
              </div>
              <ol>
                {spans.map((span) => {
                  const width = Math.max(0.6, ((span.duration_ms ?? 0) / totalMs) * 100);
                  const left = (span.offset_ms / totalMs) * 100;
                  return (
                    <li key={span.span_id} className={`span ${span.kind} ${span.status}`}>
                      <span className="kind">{KIND_LABEL[span.kind] ?? span.kind}</span>
                      <span className="name">{span.name}</span>
                      <span className="bar-track">
                        <span
                          className="bar"
                          style={{ left: `${left}%`, width: `${width}%` }}
                        />
                      </span>
                      <span className="ms">
                        {span.duration_ms === null ? "..." : `${span.duration_ms} ms`}
                      </span>
                    </li>
                  );
                })}
              </ol>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
