<!-- bmad:context -->
<!-- Verified 2026-08-20 against 3728308. Managed by bmad-project-context; edits inside this block are replaced on refresh. Keep anything you want preserved outside the markers. -->

## OneMind

A healthcare multi-agent orchestrator: routes a request to the specialists that own the relevant data, runs them in parallel, and returns one cited, PHI-redacted answer. Python/FastAPI/LangGraph backend in `backend/` (managed with `uv`), React/TypeScript/Vite frontend in `frontend/`, both driving a local Ollama model. Design decisions live in `docs/decisions.md`; planning artefacts under `docs/planning-artifacts/`.

## Where things are

- Orchestration pipeline (redact → route → parallel dispatch → synthesize → rehydrate): `backend/src/onemind/orchestrator/graph.py`
- Design decisions with the evidence behind them: `docs/decisions.md` — read before changing routing, PHI, or synthesis behavior.
- Why the four specialists map to OneData's healthcare line: `docs/company-research.md`
- Routing evaluation: `evals/run_eval.py`, dataset `evals/datasets/routing.jsonl`

## Running and verifying

- No Makefile; `run.ps1` is the task runner (`./run.ps1 test|eval|demo|check`) — `make` isn't available on this dev machine.
- `evals/run_eval.py` needs Ollama serving `qwen3.5:4b` locally; `backend/tests/` runs fully offline against `StubProvider` and needs neither Ollama nor a model.

## Conventions that differ from defaults

- Structured model output (routing decisions, tool-call plans) goes through JSON-Schema-constrained decoding via `LLMProvider.structured` (`format` on Ollama, forced tool-use on Bedrock) — never free-text parsing or a LangChain chat wrapper. `backend/src/onemind/llm/ollama.py`, `llm/bedrock.py`.
- The router asks the model for a boolean `is_actionable`, never a numeric confidence score — a 4B model doesn't calibrate confidence reliably. `backend/src/onemind/orchestrator/router.py`.
- Arithmetic (means, trends, breach counts, denial rates) is computed in tool code, never left to the model. `backend/src/onemind/tools/`.
- PHI redaction crosses the model boundary in both directions — redact toward the model, rehydrate at the tool call and in tool results — never redact-and-stop. `backend/src/onemind/guardrails/phi.py`.

<!-- /bmad:context -->
