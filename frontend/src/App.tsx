/**
 * Shell: brand, environment status, and the two tabs.
 *
 * Run state lives here rather than in either view, so switching tabs mid-answer
 * shows the trace of the request you are reading instead of resetting it.
 */

import { useEffect, useState } from "react";
import MedicalView from "./views/MedicalView";
import DeveloperView from "./views/DeveloperView";
import { useOrchestrator } from "./useOrchestrator";
import { IconFlow, IconPulse, IconShield } from "./icons";

type Tab = "medical" | "developer";

const TABS: { id: Tab; label: string; Icon: typeof IconPulse }[] = [
  { id: "medical", label: "Medical", Icon: IconPulse },
  { id: "developer", label: "Developer", Icon: IconFlow },
];

/** The hash is the source of truth so a tab is linkable and survives reload. */
function tabFromHash(): Tab {
  return window.location.hash.replace(/^#\/?/, "") === "developer" ? "developer" : "medical";
}

export default function App() {
  const o = useOrchestrator();
  const [tab, setTab] = useState<Tab>(tabFromHash);

  useEffect(() => {
    const sync = () => setTab(tabFromHash());
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);

  const go = (next: Tab) => {
    window.location.hash = "/" + next;
    setTab(next);
  };

  return (
    <div className="app">
      <a className="skip" href="#main">
        Skip to content
      </a>

      <header className="topbar">
        <div className="brand">
          <span className="mark" aria-hidden />
          <div className="brand-text">
            <strong>OneMind</strong>
            <span>Clinical multi-agent assistant</span>
          </div>
        </div>

        <nav className="tabs" role="tablist" aria-label="Views">
          {TABS.map(({ id, label, Icon }, i) => (
            <button
              key={id}
              role="tab"
              id={"tab-" + id}
              aria-selected={tab === id}
              aria-controls="main"
              tabIndex={tab === id ? 0 : -1}
              className={"tab" + (tab === id ? " is-active" : "")}
              onClick={() => go(id)}
              onKeyDown={(e) => {
                if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
                e.preventDefault();
                const next = TABS[(i + (e.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length];
                go(next.id);
              }}
            >
              <Icon size={15} />
              {label}
              {id === "developer" && o.spans.length > 0 && (
                <span className="tab-count tabular">{o.spans.length}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="env">
          <span className={"status" + (o.online === false ? " is-down" : o.online ? " is-up" : "")}>
            <i aria-hidden />
            {o.online === false ? "API offline" : o.online ? "Local inference" : "Connecting"}
          </span>
          <span className="env-chip mono">qwen3.5:4b</span>
          <span className="env-chip is-safe">
            <IconShield size={13} />
            No PHI leaves this machine
          </span>
        </div>
      </header>

      <main id="main" role="tabpanel" aria-labelledby={"tab-" + tab} className="main">
        {tab === "medical" ? <MedicalView o={o} /> : <DeveloperView o={o} />}
      </main>
    </div>
  );
}
