<!-- bmad:context -->
<!-- Verified 2026-08-20 against 3728308. Managed by bmad-project-context; edits inside this block are replaced on refresh. Keep anything you want preserved outside the markers. -->

## OneMind

A healthcare multi-agent orchestrator: routes a request to the specialists that own the relevant data, runs them in parallel, and returns one cited, PHI-redacted answer. Python/FastAPI/LangGraph backend in `backend/` (managed with `uv`), React/TypeScript/Vite frontend in `frontend/`, both driving a local Ollama model. Design decisions live in `docs/decisions.md`; planning artefacts under `docs/planning-artifacts/`.

## Where things are

- Orchestration pipeline (redact → route → parallel dispatch → synthesize → rehydrate): `backend/src/onemind/orchestrator/graph.py`
- Design decisions with the evidence behind them: `docs/decisions.md` — read before changing routing, PHI, or synthesis behavior.
- Why the four specialists map to OneData's healthcare line: `docs/company-research.md`
- Routing evaluation: `evals/run_eval.py`, dataset `evals/datasets/routing.jsonl`
- Whether the orchestration earns its keep: `evals/arms.py` holds both architectures; `--arm both` scores them with one function. Fairness invariants are pinned offline in `backend/tests/test_eval_arms.py`.
- CI (`.github/workflows/ci.yml`) runs the offline suite and the frontend typecheck only. Evals need a live model and are run by hand; their reports are committed.

## Running and verifying

- No Makefile; `run.ps1` is the task runner (`./run.ps1 test|eval|conv|compare|demo|check`) — `make` isn't available on this dev machine.
- `evals/run_eval.py` needs Ollama serving `qwen3.5:4b` locally; `backend/tests/` runs fully offline against `StubProvider` and needs neither Ollama nor a model.

## Conventions that differ from defaults

- Structured model output (routing decisions, tool-call plans) goes through JSON-Schema-constrained decoding via `LLMProvider.structured` (`format` on Ollama, forced tool-use on Bedrock) — never free-text parsing or a LangChain chat wrapper. `backend/src/onemind/llm/ollama.py`, `llm/bedrock.py`.
- The router asks the model for a boolean `is_actionable`, never a numeric confidence score — a 4B model doesn't calibrate confidence reliably. `backend/src/onemind/orchestrator/router.py`.
- Arithmetic (means, trends, breach counts, denial rates) is computed in tool code, never left to the model. `backend/src/onemind/tools/`.
- PHI redaction crosses the model boundary in both directions — redact toward the model, rehydrate at the tool call and in tool results — never redact-and-stop. `backend/src/onemind/guardrails/phi.py`.

<!-- /bmad:context -->

## Pitfalls

Kept outside the managed block above so a context refresh does not drop them.
Each one cost a session real time.

- **Run `ruff` from `backend/`, never from the repo root.** The `line-length = 100`
  setting lives in `backend/pyproject.toml`. Invoked from the root, ruff finds no
  config and silently uses its default of 88 — `check` merely reports odd results,
  but `format` rewrites files into a state that `./run.ps1 lint` and CI then reject.
  `./run.ps1 lint` always gets this right; use it.
- **The README's evaluation tables are generated.** Everything between
  `<!-- eval:begin -->` and `<!-- eval:end -->` comes from
  `evals/comparison_report.json` via `evals/update_readme.py`, and CI runs that
  script with `--check`. Hand-editing the numbers turns the build red. Prose outside
  the markers is hand-written and safe.
- **Never `git add -A` in this repo.** More than one agent session has worked in this
  single working tree at once, and a blind stage-all has already swept another
  session's in-progress files into a commit whose message described something else.
  Stage explicit paths.
- **`asyncio.run` once per process, not once per phase.** `OllamaProvider` builds one
  `httpx.AsyncClient` and reuses it, and a pooled connection belongs to the loop that
  opened it. A second `asyncio.run` dies on `RuntimeError: Event loop is closed` —
  and in an eval harness it dies *after* the first phase printed a clean table.
  `evals/run_eval.py:run_all` is the shape to copy.
