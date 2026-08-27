/**
 * One request's worth of orchestrator state, lifted out of the view layer.
 *
 * Both tabs read the same run. That is the whole reason this is a hook held by
 * the shell rather than state inside a view: switching from Console to
 * Developer must show the trace of the answer you are looking at, not start a
 * new one.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchAgents,
  fetchExamples,
  fetchHealth,
  streamChat,
  type AgentInfo,
  type ExampleInfo,
  type Fact,
  type Health,
  type Outcome,
  type Span,
} from "./api";

export interface Orchestrator {
  agents: AgentInfo[];
  examples: ExampleInfo[];
  /** Provider and model as the server reports them. Null until the first
   *  health call lands, and after it fails. */
  health: Health | null;
  online: boolean | null;
  busy: boolean;
  answer: string;
  spans: Span[];
  outcome: Outcome | null;
  facts: Fact[];
  error: string | null;
  /** Wall-clock ms since submit, ticking while busy. Frozen once done. */
  elapsedMs: number;
  /** True once a run has produced anything at all. */
  hasRun: boolean;
  submit(text: string): void;
  stop(): void;
}

export function useOrchestrator(): Orchestrator {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [examples, setExamples] = useState<ExampleInfo[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState("");
  const [spans, setSpans] = useState<Span[]>([]);
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [facts, setFacts] = useState<Fact[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);

  const abort = useRef<AbortController | null>(null);
  const startedAt = useRef(0);
  // Deliberately a ref and not localStorage. Memory should last exactly as
  // long as the tab: a refresh drops the id, the server TTLs the orphan, and
  // no redaction vocabulary outlives the session that created it.
  const sessionId = useRef<string | undefined>(undefined);

  // Health and the roster together: both are "is the server there and what is
  // it", so one failure is one offline state rather than a half-populated
  // header claiming a model it never got.
  useEffect(() => {
    Promise.all([fetchHealth(), fetchAgents()])
      .then(([info, list]) => {
        setHealth(info);
        setAgents(list);
        setOnline(true);
      })
      .catch(() => {
        setOnline(false);
        setError("Cannot reach the API. Is the backend running on :8080?");
      });
    fetchExamples().then(setExamples).catch(() => undefined);
  }, []);

  // A 100ms tick is enough to read as live without being a render storm, and
  // it stops the instant the stream closes so the final number is the real one.
  useEffect(() => {
    if (!busy) return;
    const id = window.setInterval(() => setElapsedMs(Date.now() - startedAt.current), 100);
    return () => window.clearInterval(id);
  }, [busy]);

  const stop = useCallback(() => {
    abort.current?.abort();
    setBusy(false);
  }, []);

  const submit = useCallback(
    (text: string) => {
      const message = text.trim();
      if (!message || busy) return;

      abort.current?.abort();
      const controller = new AbortController();
      abort.current = controller;

      startedAt.current = Date.now();
      setElapsedMs(0);
      setBusy(true);
      setAnswer("");
      setSpans([]);
      setOutcome(null);
      setError(null);

      void (async () => {
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
                sessionId.current = result.session_id || sessionId.current;
                setFacts(result.facts ?? []);
                // The streamed tokens are the model's redacted output; the done
                // payload has been through outbound re-hydration.
                setAnswer(result.answer);
                setSpans(result.trace.spans);
              },
              onError: setError,
            },
            controller.signal,
            sessionId.current,
          );
        } catch (err) {
          if ((err as Error).name !== "AbortError") setError(String(err));
        } finally {
          setElapsedMs(Date.now() - startedAt.current);
          setBusy(false);
        }
      })();
    },
    [busy],
  );

  return {
    agents,
    examples,
    health,
    online,
    busy,
    answer,
    spans,
    outcome,
    facts,
    error,
    elapsedMs,
    hasRun: busy || spans.length > 0 || answer.length > 0,
    submit,
    stop,
  };
}
