# Cross-plane reconciliation

**Status:** approved, not yet implemented
**Date:** 2026-08-21

## 1. The problem

Asked *"Claim CLM-8849 for patient 12345 was denied. Check their diagnosis
history and tell me whether the billed code matches"*, the system answered that
it could not determine the match.

Both halves of the answer had been retrieved:

| Specialist | Tool | Retrieved |
|---|---|---|
| Clinical | `fhir_search_patient` | patient 12345, conditions `[N18.3]` |
| Revenue Cycle | `claim_lookup` | CLM-8849, `icd10_code: N18.3` |

The codes match. Nothing in the system was capable of noticing.

Four defects, in descending order of importance.

### 1.1 The join is nobody's job

`build_graph` fans out with `Send("specialist", {...redacted})` — each
specialist receives only the original request. Fan-in merges `SpecialistResult`
objects through an `operator.add` reducer, and `Orchestrator.stream` passes them
straight to the synthesiser, which sees only `result.answer` prose.

Each specialist correctly reported the limits of its own data plane. The
synthesiser is correctly forbidden from adding facts. Two accurate "I do not
have that" statements compose into a wrong answer. Every component behaved as
designed; the design has no place for a cross-plane comparison.

### 1.2 `validate_code` ran on a fabricated code

`_is_identifier` in `agents/base.py` returns true for `*_id`, `mrn` and
`birth_date`. The `code` argument of `validate_code` is not covered, so the
grounding check let through **E11.9** — a value that appears nowhere in the
request or the fixtures. It comes from the tool's own description string:
`"e.g. E11.9, 99214, or CO-197"`.

This is the exact failure the guard's docstring describes ("a small model asked
a question that names nobody will fill the argument from the nearest example it
can see — including the ones in the tool descriptions it was just handed"),
arriving through an argument the predicate does not classify. The fabricated
code then produced a fabricated mismatch in the answer.

### 1.3 `ungrounded_values` flags request-supplied terms

Reproduced against Clinical's real evidence and answer:

```
CLINICAL flagged: ['CLM-8849', 'ICD-10']
```

`ungrounded_values(answer, evidence)` never sees the request, so a specialist
restating an identifier **the user typed** is recorded as a fabrication. The
sibling guard in `base.py` already has the correct rule —
`text.casefold() in request.casefold()` — this one does not use it. The result
is a trust-eroding "2 unverified" badge on values that were never in doubt.

### 1.4 Single planning round, no observe step

No specialist can act on what a tool returned. Subsumed by §1.1 and not
addressed separately: with reconciliation in place there is no question in the
demo set that requires a second planning round.

---

## 2. Goal

When answering requires comparing facts held in two different data planes,
**code performs the comparison and the model only explains the result.**

Three properties the system must have afterwards:

1. A cross-plane question produces a computed answer whenever both halves were
   retrieved — never "cannot determine" when the data is present.
2. The comparison is deterministic and reproducible. Same claim, same problem
   list, same sentence, every run.
3. The comparison carries provenance — which field of which record was compared
   against which — so a reader can verify it without trusting the model.

### Non-goals

- Specialists do not gain tools, message each other, or loop.
- The router is unchanged.
- The single-planning-round design (`decisions.md`) stands.
- No general cross-plane query language. Registered checks only.

---

## 3. Design principle

The reconciler sits **above** the data planes, after fan-in. It has no tool
access and cannot retrieve anything. It reads only evidence the specialists
already legitimately fetched, and computes a relation between two results.

Clinical still cannot call `claim_lookup`. Revenue Cycle still cannot read a
chart. The isolation boundary in `SpecialistSpec.data_plane` is unchanged.

The defensible statement: *nothing new is retrieved, and the join is not
performed by a language model.*

This mirrors how production claim scrubbers work. Medical-necessity and
bundling edits (NCCI PTP pairs, MUE limits, LCD covered-diagnosis lists) are
deterministic table lookups that cite a rule ID and effective date, because the
result has to survive a payer appeal. Where those products use models at all it
is for CDI query drafting and appeal letters — after a deterministic engine has
decided.

---

## 4. Architecture

```
redact -> route -+-> [specialist] --+-> reconcile -> END
                 |   [specialist]   |   (fan-in)
                 +-> END (clarify)
```

`specialist -> END` becomes `specialist -> reconcile -> END`. LangGraph runs
`reconcile` once, after every `Send` branch completes.

**Why inside the graph rather than in `Orchestrator.stream`.** The graph's
docstring states the split: the graph decides and gathers, synthesis is
transport. Reconciling is gathering. Placing it in the graph also gives it a
trace span without special-casing, and the audit reader should see the
comparison happen.

**No new evidence plumbing is required.** `SpecialistResult.tool_calls` already
carries `{tool, arguments, result}` with the redacted tool output, and
`operator.add` already merges it at fan-in. The evidence is sitting in
`state["results"]` and nothing reads it. This step adds a consumer, not a store.

### 4.1 Redaction space

Tool outputs in `tool_calls` are `session.redact_json(output)` — already
redacted. Checks therefore operate in redacted space, and this is *safe for
joining*: `PHISession` is per-request with a stable value-to-placeholder
mapping, so the same patient id redacts to the same placeholder in both the
claim result and the FHIR result. Comparing `PHI_PATIENT_2 == PHI_PATIENT_2` is
a valid identity test.

`Finding.statement` and `Finding.provenance` may contain placeholders and are
rehydrated on the way out, alongside the answer, in `Orchestrator.stream`.

---

## 5. The check contract

New module: `backend/src/onemind/orchestrator/reconcile.py`.

```python
class Finding(BaseModel):
    check: str
    verdict: Literal["match", "mismatch", "not_applicable", "insufficient_evidence"]
    statement: str            # formatted by code; never model output
    compared: dict[str, str]  # {"billed_icd10": "N18.3", "problem_list": "N18.3"}
    provenance: str           # "CLM-8849.icd10_code vs patient 12345 conditions[]"
```

`statement` being a formatted string is the point of the whole design. The
sentence the user reads about whether the codes match never passes through a
model.

A read-only view over merged evidence:

```python
class Evidence:
    def __init__(self, results: Sequence[SpecialistResult]) -> None: ...
    def outputs(self, tool: str) -> list[dict[str, Any]]: ...
    def first(self, tool: str) -> dict[str, Any] | None: ...
    def has(self, *tools: str) -> bool: ...
```

Registration follows the existing `@tool(tools, ...)` pattern in
`tools/base.py`, so the idiom is already in the codebase:

```python
@check(name="billed_diagnosis_matches_problem_list",
       requires=("claim_lookup", "fhir_search_patient"))
def billed_diagnosis_matches(ev: Evidence) -> Finding | None: ...
```

`reconcile(results) -> list[Finding]` runs each check whose `requires` tools are
all present in the evidence and collects the non-`None` findings.

### 5.1 The join-key guard

Every check that spans two patient-scoped records **must verify the join key
before comparing**. If the claim's `patient_id` differs from the resolved
clinical record's `patient_id` — an ambiguous name match, a different subject —
the check returns `insufficient_evidence` with a statement saying the records
are for different patients and no comparison was made.

Comparing one person's claim against another person's problem list is the worst
failure this component could produce. This is the same instinct as
`resolve_patients` refusing on ambiguity, applied one layer up.

---

## 6. The checks

Two. The mechanism is the deliverable; a third check is a function, not a new
code path.

### 6.1 `billed_diagnosis_matches_problem_list`

Requires `claim_lookup` + `fhir_search_patient`. Compares the claim's
`icd10_code` against the set of `conditions[].code` on the patient record.

- billed code is on the problem list → `match`
- billed code is absent from a non-empty problem list → `mismatch`
- patient ids differ → `insufficient_evidence` (§5.1)
- either result has `found: false`, or the problem list is empty → no finding

`claim_lookup(patient_id=...)` returns a list, so the check runs per claim and
may emit several findings.

For CLM-8849: `match`.

### 6.2 `denial_is_coding_related`

Requires `claim_lookup`. Classifies the claim's `denial_code` against a
reference table. `CO-197` is an authorization denial, so the finding is
`not_applicable`: no amount of recoding fixes a missing precertification.

This check is what turns a correct-but-useless answer into a useful one. It is
also single-plane, which is a deliberate exception to the cross-plane framing —
it belongs here because it is a deterministic table lookup that the model
currently guesses at, and it shares the whole mechanism.

**No new fixture file.** `codesets.json` already carries a `denial_codes`
system with all five codes and their official display text, and `validate_code`
already searches it. It lacks only the classification, so each entry gains two
fields in `generate.py`'s `DENIALS` table:

```json
{"code": "CO-197", "display": "Precertification/authorization absent",
 "category": "authorization", "coding_related": false}
```

| Code | category | coding_related |
|---|---|---|
| `CO-197` | authorization | false |
| `CO-16` | submission | false |
| `CO-29` | timely_filing | false |
| `PR-204` | benefit | false |
| `CO-11` | coding | **true** |

An unrecognised denial code yields no finding rather than a guess.

### 6.3 The fixture generator makes `mismatch` unreachable

Stronger than a gap — it is structural. `generate.py` builds every claim with:

```python
"icd10_code": patient["conditions"][0]["code"],
```

The billed diagnosis is *by construction* the patient's first condition, for all
14 claims. `billed_diagnosis_matches_problem_list` therefore returns `match`
every time, and the `mismatch` branch is unreachable outside unit tests.
Separately, no generated claim carries `CO-11`, so §6.2's `coding_related: true`
branch is unreachable too. A reviewer asking "show me a case where it does *not*
match" would get nothing.

Editing `claims.json` by hand does not fix this — `_read_json` treats fixtures as
generated output and `generate.py` would overwrite it. **The change belongs in
`generate.py`.**

**Append two claims after the existing loop, do not modify the loop.** The
generator is seeded specifically so that "every field, MRN, SSN, claim id and
device id stays byte-identical" (its own docstring). Inserting a branch inside
the `range(14)` loop advances the RNG stream and reshuffles every downstream
claim id, amount and payer — breaking any test or eval case that names a
specific claim. Appending after the loop, with literal values rather than `rng`
draws, leaves the first 14 untouched.

The two appended claims, both for patient `12345` (problem list `[N18.3]`):

| claim_id | icd10 | status | denial | Demonstrates |
|---|---|---|---|---|
| `CLM-8883` | `E11.9` | denied | `CO-11` | `mismatch` + coding-related denial |
| `CLM-8886` | `N18.3` | denied | `CO-29` | `match` + non-coding denial |

Re-run `generate.py` and confirm `git diff` on `claims.json` shows **only** the
two appended objects. If earlier rows moved, the loop was touched.

---

## 7. Reaching the answer

Findings enter the synthesiser prompt as a section that is deliberately **not**
a specialist heading — attributing a computed fact to "Clinical" would be false,
and that prompt already fights misattribution hard (see its own comment about
the 4B model copying example phrasings).

```
VERIFIED FINDINGS — computed directly from the records the specialists retrieved.
State these as established fact. Do not contradict them or hedge them.

- Billed diagnosis N18.3 on claim CLM-8849 matches the patient's active problem list.
  [CLM-8849.icd10_code vs patient 12345 conditions]
- Denial CO-197 is an authorization denial, not a coding denial.
  [CLM-8849.denial_code]
```

Plus one rule: lead with the verified findings when they answer the question.

### 7.1 The `needs_synthesis` trap

`Synthesizer.needs_synthesis` currently returns true only when more than one
specialist produced non-empty prose. A specialist can retrieve evidence
successfully and still return an empty answer (timeout after tool execution,
model returning whitespace). In that case the single-answer fast path passes the
other specialist's prose through unchanged and **silently drops every finding**.

`needs_synthesis` becomes: more than one answer **or** any findings present.

### 7.2 Grounding interaction

`Finding.statement` is code-generated from tool output, so every value in it is
by construction present in the evidence. Findings therefore cannot trip
`ungrounded_values`, and no exemption is needed. Worth asserting in a test so
the property does not silently break.

---

## 8. Guardrail fixes

### 8.1 Coded-value grounding

`_is_identifier` keeps its current definition and docstring — it describes
arguments that select *whose record* is returned, and a billing code does not.
A sibling predicate is added:

```python
def _is_coded_value(name: str) -> bool:
    """Arguments naming a clinical or billing code.

    A different harm from `_is_identifier`: a fabricated code does not open the
    wrong person's chart, it produces a confident statement about the wrong
    procedure. Both must be grounded; the reasons are not the same.
    """
    return name in {"code", "icd10_code", "cpt_code", "denial_code"}
```

The call site in `_run` unions the two. Two named harms, each with its own
rationale, rather than one predicate quietly grown a second meaning.

### 8.2 Request-aware answer grounding

```python
def ungrounded_values(answer: str, evidence: str, request: str = "") -> list[str]:
```

The request is folded into the haystack, same rule as `_is_grounded`. The
parameter is optional so existing tests and `evals/` continue to work unchanged;
`agents/base.py` passes it.

**Tradeoff, stated explicitly.** This relaxes the guard. If a user asks about a
claim id that does not exist and the model then asserts things about it, the id
is no longer flagged. That is judged correct — the guard exists to catch values
the *model* invented, and a value the user typed is not invented — but it is a
real narrowing, and the docstring must say so.

---

## 9. Files touched

| File | Change |
|---|---|
| `orchestrator/reconcile.py` | **new** — `Evidence`, `Finding`, `@check`, the two checks |
| `orchestrator/graph.py` | `reconcile` node; `findings` in state and `OrchestratorOutcome`; rehydrate findings |
| `orchestrator/synthesizer.py` | findings section in prompt; `needs_synthesis` |
| `observability/trace.py` | `SpanKind.RECONCILE` |
| `agents/base.py` | `_is_coded_value`; pass `request` to `ungrounded_values` |
| `guardrails/grounding.py` | optional `request` parameter |
| `fixtures/generate.py` | `DENIALS` gains `category`/`coding_related`; two appended claims (§6.3) |
| `fixtures/claims.json`, `codesets.json` | regenerated output, committed |
| `frontend/src/api.ts` | `"reconcile"` in the `SpanKind` union; `findings` on the response type |
| `frontend/src/App.tsx` | `KIND_LABEL.reconcile`; findings block |
| `frontend/src/styles.css` | `.span.reconcile` |
| `README.md` | Known limits (§11) |
| `docs/decisions.md` | new decision entry |

The frontend `SpanKind` is a closed TypeScript union and `KIND_LABEL` has a
`?? span.kind` fallback, so an unknown kind renders but fails typecheck. All
three frontend files must change together.

### 9.1 UI surfacing (last item, safe to cut)

`OrchestratorOutcome` already carries `citations` and `unverified`, which the UI
renders as chips plus a detail block. `findings` follows the identical path: a
"2 verified" chip and a block mirroring the existing "Not supported by the
retrieved records" section. Cheap because the pattern exists; droppable without
affecting correctness.

---

## 10. Testing

TDD. Most of this is pure functions over dicts — fast, no model required.

- **`backend/tests/test_reconcile.py`** (new)
  - each check across `match` / `mismatch` / `not_applicable` / `insufficient_evidence`
  - the §5.1 mismatched-patient refusal
  - `found: false` on either side yields no finding
  - unknown denial code yields no finding
  - a check whose `requires` tools are absent does not run
  - §7.2: findings never trip `ungrounded_values`
- **`test_answer_grounding.py`** — the §1.3 reproduction: `['CLM-8849', 'ICD-10']`
  today, empty once `request` is passed. Plus a case proving a genuinely
  fabricated code is *still* caught with the request present.
- **`test_orchestrator.py`** — graph wiring reaches `reconcile`; findings survive
  to `OrchestratorOutcome`; the §7.1 empty-answer case still synthesises.
- **`test_tools.py`** — the classified `denial_codes` codeset loads; `validate_code`
  still returns the same shape for a denial code; the §6.3 append left the first
  14 claims byte-identical.
- **`evals/datasets`** — CLM-8849 added as a cross-plane case, and the §6.3
  mismatch claim as its negative counterpart, so regressions surface in
  `run_eval.py` rather than in a demo.

---

## 11. Known limits to add to the README

- **Only registered joins work.** An unanticipated cross-plane question still
  degrades to "cannot determine". The improvement is that it degrades for a
  reason that can be pointed at, rather than because the architecture had no
  path from evidence to comparison.
- **The model still writes the prose around the findings.** The "do not
  contradict" instruction plus `ungrounded_values` constrain it, but at 4B it
  can still hedge awkwardly next to a finding stated flatly.
- **The denial-code classification is a five-row fixture,** not a maintained
  CARC/RARC set with effective dates. Production would version it and cite the
  edition, the way a scrubber cites the NCCI quarter it edited against.
- **Diagnosis matching is set membership, not medical necessity.** The check
  answers "is the billed diagnosis on this patient's problem list", which is a
  weaker question than "does this payer cover this CPT for this diagnosis" —
  the latter needs LCD/NCD coverage tables the fixtures do not have.

---

## 12. Rejected alternatives

**Give the synthesiser raw `tool_calls` instead of prose.** Far less code —
extend `_format`, relax "add no facts" to "you may compare values across
sections". Rejected: it asks a 4B model to perform the join, which reintroduces
the fabrication risk the rest of the system exists to prevent, and the answer
stops being reproducible. The comparison is the one part of this answer that
must not be generated.

**Full plan-act-observe loops with a re-dispatching supervisor.** Most capable,
least explainable, and `decisions.md` already rejected open loops at this model
size for non-termination. No demo-set question needs it once §1.1 is fixed.

**Sequential dispatch — run Revenue Cycle first, inject the claim's ICD-10 into
Clinical's request.** Keeps agents isolated and lets facts flow, but it makes
the router responsible for dependency ordering, doubles latency on every
cross-agent request, and still leaves the comparison itself to a model.
