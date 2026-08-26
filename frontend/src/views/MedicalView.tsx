/**
 * The clinician-facing tab.
 *
 * Everything here answers "can I act on this?" - the answer, what the system
 * refuses to vouch for, and what the conversation is currently scoped to.
 * Timings, span ids, payloads, the computed cross-plane comparisons and the
 * state of each data plane belong to the Developer tab; putting them here
 * would make a clinical read harder, not more transparent.
 */

import { useEffect, useRef, useState } from "react";
import { Markdown } from "../markdown";
import type { Orchestrator } from "../useOrchestrator";
import {
  IconAlert,
  IconDatabase,
  IconSend,
  IconShield,
  IconStop,
  IconTag,
} from "../icons";

export default function MedicalView({ o }: { o: Orchestrator }) {
  const [prompt, setPrompt] = useState("");
  const [showRedacted, setShowRedacted] = useState(false);
  const box = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setShowRedacted(false);
  }, [o.outcome]);

  const send = (text: string) => {
    o.submit(text);
    box.current?.blur();
  };

  const clarifying = o.outcome && !o.outcome.is_actionable;

  return (
    <div className="medical">
      <div className="medical-main">
        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            send(prompt);
          }}
        >
          <label className="composer-label" htmlFor="ask">
            Clinical request
          </label>
          <textarea
            id="ask"
            ref={box}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(prompt);
              }
            }}
            placeholder="Ask about a patient, a claim, a policy, or a device reading."
            rows={3}
          />
          <div className="composer-foot">
            <p className="hint">
              <IconShield size={14} />
              Identifiers are masked before the model reads this.{" "}
              <span className="nowrap">Enter to send · Shift + Enter for a new line.</span>
            </p>
            {o.busy ? (
              <button type="button" className="btn btn-ghost" onClick={o.stop}>
                <IconStop size={15} />
                Stop
              </button>
            ) : (
              <button type="submit" className="btn btn-primary" disabled={!prompt.trim()}>
                <IconSend size={15} />
                Send
              </button>
            )}
          </div>
        </form>

        {/* These load the composer rather than submitting. An example is a
            starting point - the useful demo is editing one to name a different
            patient, and sending on click took that away and spent ninety
            seconds of inference on a question nobody had finished asking. */}
        <div className="starters" role="group" aria-label="Example requests. Loads the box for editing.">
          {o.examples.map((example) => (
            <button
              key={example.label}
              type="button"
              className="starter"
              title={`${example.expect} — loads the box for editing; press Send to run it.`}
              disabled={o.busy}
              onClick={() => {
                setPrompt(example.prompt);
                // Focus and drop the caret at the end so the example is
                // immediately editable, not merely visible.
                const el = box.current;
                if (el) {
                  el.focus();
                  requestAnimationFrame(() => {
                    el.selectionStart = el.selectionEnd = el.value.length;
                  });
                }
              }}
            >
              {example.label}
            </button>
          ))}
        </div>

        {o.error && (
          <div className="banner banner-danger" role="alert">
            <IconAlert size={16} />
            <span>{o.error}</span>
          </div>
        )}

        {!o.hasRun && !o.error && <EmptyState />}

        {o.hasRun && (
          <article className={"answer" + (clarifying ? " is-clarifying" : "")}>
            <header className="answer-head">
              <h2>{clarifying ? "Needs clarification" : "Answer"}</h2>
              <div className="answer-meta">
                {o.busy && <span className="working">Working…</span>}
                {!o.busy && o.elapsedMs > 0 && (
                  <span className="tabular">{(o.elapsedMs / 1000).toFixed(1)}s</span>
                )}
                {o.outcome?.agents.map((key) => (
                  <span key={key} className={"src-pill plane-" + key}>
                    {o.agents.find((a) => a.key === key)?.display_name ?? key}
                  </span>
                ))}
              </div>
            </header>

            <div className="answer-body">
              {o.answer || o.outcome?.clarifying_question ? (
                <Markdown text={o.answer || o.outcome?.clarifying_question || ""} />
              ) : (
                <span className="caret" />
              )}
            </div>

            {o.outcome && o.outcome.unverified?.length > 0 && (
              <section className="block block-unverified">
                <h3>
                  <IconAlert size={14} />
                  Not supported by the retrieved records
                </h3>
                <p>
                  The answer asserts these values and the tool results do not back them. Treat them
                  as unconfirmed.
                </p>
                <div className="chips">
                  {o.outcome.unverified.map((v) => (
                    <code key={v}>{v}</code>
                  ))}
                </div>
              </section>
            )}
          </article>
        )}
      </div>

      <aside className="rail" aria-label="Request context">
        <section className={"card safety" + (o.outcome?.phi_redactions ? " is-active" : "")}>
          <h2>
            <IconShield size={15} />
            PHI safeguard
          </h2>
          {/* A count of zero before anything has run reads as a result. Until
              there is an outcome, the card states the guarantee instead. */}
          {o.outcome ? (
            <p className="big">
              <strong className="tabular">{o.outcome.phi_redactions}</strong>
              <span>identifiers masked before inference</span>
            </p>
          ) : (
            <p className="fine">No request sent yet.</p>
          )}
          <p className="fine">
            Identifiers are replaced with placeholders before the model reads anything, and restored
            only in the answer you see. Nothing leaves this machine.
          </p>
          {o.outcome && o.outcome.phi_redactions > 0 && (
            <>
              <button
                type="button"
                className="link"
                aria-expanded={showRedacted}
                onClick={() => setShowRedacted((v) => !v)}
              >
                {showRedacted ? "Hide" : "Show"} what the model received
              </button>
              {showRedacted && <pre className="redacted">{o.outcome.redacted_request}</pre>}
            </>
          )}
        </section>

        {o.facts.length > 0 && (
          <section className="card">
            <h2>
              <IconTag size={15} />
              Session memory
            </h2>
            <p className="fine">
              Carried into the next turn, so a follow-up need not name the subject again. Values are
              placeholders, never real identifiers.
            </p>
            <ul className="facts">
              {o.facts.map((fact) => (
                <li key={fact.key}>
                  <code className="fact-key">{fact.key}</code>
                  <code className="fact-value">{fact.value}</code>
                  <span className="fine">
                    {fact.source} · turn {fact.turn}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )}
      </aside>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty">
      <IconDatabase size={22} />
      <h2>Ask across four data planes at once</h2>
      <p>
        A request is masked, routed to the specialists that own the relevant records, run in
        parallel, and returned as one cited answer. Anything the retrieved records do not back is
        flagged rather than left to read as fact.
      </p>
      <p className="fine">Pick a starter above to load it, then edit and send.</p>
    </div>
  );
}
