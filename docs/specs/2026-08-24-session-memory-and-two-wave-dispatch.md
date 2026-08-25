# Session memory and two-wave dispatch

**Status:** implemented — §14 records where the build departed from this design, §15 the follow-on work on identity
**Date:** 2026-08-24

## 1. The problem

Two questions the system cannot answer, for the same underlying reason.

### 1.1 A two-hop question has no path

Asked *"Pull up Sarah Chen and check whether her blood pressure has been
trending high"*, both specialists are routed correctly and both fail:

| Specialist | Tool | Outcome |
|---|---|---|
| Clinical | `fhir_search_patient` | resolves the name, returns `patient_id` |
| Remote Monitoring | `telemetry_series` | no `patient_id` in the request — call blocked |

`build_graph` fans out with `Send("specialist", {agent, redacted})`. Every
branch receives the original request and nothing else, and all branches run
concurrently. Clinical resolves the identifier that Remote Monitoring needs
*while Remote Monitoring is already deciding it has nothing to work with*.

The grounding guard then does its job correctly and makes the failure
permanent: `_is_grounded` accepts a tool argument only when it is a placeholder
this session issued or a literal in the request. A `patient_id` that Clinical
resolved from a name is neither, so even a hypothetical retry would be blocked.

### 1.2 There is no conversation

`ChatRequest` carries a single `message` field. No session identifier, no
history, no store. `Orchestrator.stream` mints a fresh `PHISession` per request
at `graph.py:203`.

A follow-up — *"and her telemetry?"* — reaches the router as four words naming
no subject. The router correctly classifies it as not actionable and returns a
clarifying question. There is nothing wrong with that behaviour given the
information available; the information is simply not being kept.

### 1.3 One cause

Both are the same defect. Something established in one place cannot reach
another place that needs it — across specialists in §1.1, across turns in
§1.2. One mechanism fixes both.

---

## 2. Goal

A specialist blocked for want of an identifier runs again once that identifier
becomes available, whether it was established moments ago by a sibling
specialist or several turns ago in the same conversation.

### Non-goals

- **Agent-to-agent messaging.** No specialist gains the ability to address,
  query, or wait on another. Data-plane exclusivity is what makes routing
  decidable and is enforced by `test_specialists_do_not_share_tools`.
- **An open plan–act–observe loop.** The single planning round in
  `agents/base.py` exists because a 4B model given an open loop re-calls the
  same tool with cosmetic argument variations and never terminates.
- **Durable storage.** Nothing survives process restart, by design. Durable PHI
  requires a retention policy, access control, and an erasure path — all of
  which the policy corpus in `fixtures/policies/` describes and none of which
  this system implements.
- **Widening the trust boundary.** The model must still never see a real
  identifier. See §4.1, which is the constraint the whole design turns on.

---

## 3. Design principle

This is the **blackboard pattern**: independent specialists that know nothing
about each other, reading from and writing to a shared structure controlled by
an orchestrator. The specialists are unchanged in kind — they still receive a
request, plan, call their own tools, and answer. They simply start better
informed.

The orchestrator owns the blackboard. Nothing is added to it that a specialist
did not legitimately retrieve, and nothing on it is a claim — only an
identifier and where it came from.

Two properties follow, and both are worth stating because they are what make
the feature safe to demo:

**The dispatch decision is not a model call.** Whether a second wave runs is a
set intersection computed in code: does a blocked specialist's declared `needs`
overlap the facts now available? This follows the rule the codebase already
applies to arithmetic (`tools/`) and to cross-plane comparison
(`reconcile.py`: *"the join is not performed by a language model"*).

**The wave count is a counter, not a judgement.** Two waves maximum, enforced
structurally. There is no prompt that could talk the system into a third.

---

## 4. Architecture

```
turn N
  request ──redact──► route ──► [ wave 1 ] ──► reconcile
                        ▲                          │
                     history                    extract facts
                        │                          │
                        │                    blocked ∩ needs?
                        │                     ┌────┴────┐
                        │                   yes         no
                        │                     │          │
                        │                [ wave 2 ] ──► reconcile ──► synth
                        │                                              │
  Conversation ◄────────┴──────────────────────────────────────────────┘
    PHISession · Facts · turns · retained evidence

turn N+1  "and her telemetry?"  ──►  router sees history
                                     specialists see Facts
```

Wave 2 receives the same request text as wave 1. The only difference is an
`ESTABLISHED FACTS` block in its planning prompt. It still touches only its own
tools, still gets one planning round, still runs under the same guardrails.

> **Superseded by §14.3.** Offering facts to the planner turned out to let
> memory override a subject the request named explicitly. The prompt block is
> now shown only when the request names nobody, and the second wave supplies
> the resolved identifier by substituting the argument instead.

### 4.1 Redaction space

**Facts hold placeholders, never real values.** This is the single most
important decision in this document and it is easy to get wrong.

`redact_json` does not tokenise a `patient_id` field. The `_PATIENT_ID` pattern
requires a patient-ish word adjacent to the digits:

```python
r"\b(patient|member|subscriber|pt\.?)\s+(?:id\s*)?#?\s*(\d{4,6})\b"
```

Serialised FHIR output reads `"patient_id": "12345"` — the `_id": "` between
the word and the digits does not match `\s+(?:id\s*)?#?\s*`. The identifier
survives redaction intact, which is correct and deliberate for tool output the
specialist quotes back.

A naive Facts store would therefore hold `12345` and inject it into wave 2's
planning prompt, putting a real identifier in front of the model and breaking
the rule the entire PHI design rests on.

So extraction registers the value with the session first:

```python
token = session.tokenize("PATIENT", "12345")   # → "PHI_PATIENT_2"
facts.set("patient_id", token, source="clinical.fhir_search_patient")
```

Three consequences, all good:

1. The model sees `PHI_PATIENT_2` in its prompt, as it does for every other
   identifier.
2. `rehydrate_args` converts it back at the tool call, on the existing path.
3. **`_is_grounded` requires no change at all.** It already accepts any value
   that rehydration alters. The guard's guarantee is untouched rather than
   relaxed — an invented identifier is still rejected exactly as today.

An earlier draft of this design widened `_is_grounded` to accept
fact-derived values. Working in redaction space deletes that change and the
argument that would have been needed to defend it.

`PHISession` needs one new public method:

```python
def tokenize(self, kind: str, value: str) -> str:
    """Register a value seen in tool output and return its stable token."""
    return self._token_for(kind, value.strip())
```

`_token_for` already returns the existing token for a value it has seen, so a
patient named in the request and later resolved from FHIR gets one token, not
two.

---

## 5. The Facts contract

New module: `orchestrator/facts.py`.

```python
@dataclass(frozen=True)
class Fact:
    key: str        # "patient_id", "mrn", "claim_id"
    value: str      # ALWAYS a redaction placeholder — see §4.1
    source: str     # "clinical.fhir_search_patient"
    turn: int
```

`Facts` is a keyed store scoped to one subject (§5.2), with `get`, `set`,
`keys`, and a `as_prompt_block()` renderer.

### 5.1 Extractors are declared on the tool, not the specialist

An earlier draft put both `needs` and `provides` on `SpecialistSpec`. That was
wrong: *"`fhir_search_patient` yields a `patient_id`"* is a fact about the
tool. Declaring it twice invites drift.

```python
@provides("fhir_search_patient", keys=("patient_id", "mrn"))
def _from_patient_search(output: dict) -> dict[str, str]:
    if not output.get("found"):
        return {}
    return {k: str(output[k]) for k in ("patient_id", "mrn") if output.get(k)}
```

This mirrors the `@check(...)` decorator in `reconcile.py` and the
`@tool(tools, ...)` decorator in `tools/base.py` — the idiom this codebase
already uses for *declare a capability and its contract in one place*.

`SpecialistSpec.provides` becomes a derived property: the union of the keys
declared by extractors for the tools in `spec.tool_names`. Only `needs` is
written by hand.

The `found: False` guard is load-bearing. `fhir_search_patient` returns
`found: False, ambiguous: True` when several patients match the same name —
precisely the case where extracting an identifier would attach the wrong
person's chart to the conversation for the rest of the session.

### 5.2 Subject scoping

A session that discusses patient A at turn 2 and patient B at turn 9 must not
answer turn 10 with A's identifiers.

`Facts` carries a `subject` — the placeholder token of the patient it describes.
When extraction produces a `patient_id` that differs from the current subject,
the subject switches and the previous facts are dropped rather than merged.
Facts with no subject (a `claim_id` looked up on its own) are kept
independently.

The reconciler already treats answering about two patients at once as *"the
worst output this module could produce"* and guards its join key accordingly.
Leaving that hole open one layer up would be inconsistent.

---

## 6. Two-wave dispatch

### 6.1 A structural blocked signal

`SpecialistResult` gains one field:

```python
blocked: bool = False
```

Set in `_run` where `ungrounded` is already tracked, and where no usable call
was selected. The trigger reads a boolean rather than matching on the text of
an error message.

`SpecialistSpec` gains:

```python
needs: tuple[str, ...] = ()
```

Meaning: *facts that scope this specialist's lookups; a specialist that failed
for want of an identifier is retried when one of these becomes available.* Not
a precondition — Revenue Cycle runs perfectly well on a bare `claim_id`.

| Specialist | `needs` | `provides` (derived) |
|---|---|---|
| Clinical | — | `patient_id`, `mrn` |
| Revenue Cycle | `patient_id` | `patient_id`, `claim_id` |
| Compliance | — | — |
| Remote Monitoring | `patient_id` | — |

### 6.2 The trigger

After `reconcile`, in code:

```python
retry = [
    r.agent for r in results
    if r.blocked and set(specs[r.agent].needs) & facts.keys()
]
```

Non-empty and `wave < 2` → `Send` to `specialist` for each, carrying the facts.
Otherwise `END`.

Recorded as a `SpanKind.MEMORY` span naming the fact and its source
(`unblocked_by: clinical.patient_id`), so an audit reader sees the literal
value that unblocked the retry rather than a rationale.

### 6.3 The cap

`OrchestratorState` gains `wave: int`. `reconcile_node` increments it and
returns it; conditional edges observe state after the node's write, so the
guard reads the updated value with no reducer involved.

Wave 1 → `wave = 1` → retry permitted. Wave 2 → `wave = 2` → `END`
unconditionally. A wave-2 specialist that is still blocked simply reports so.

### 6.4 Duplicate results

`results` uses an `operator.add` reducer, so a retried specialist appears
twice. Deduplicate by agent keeping the later entry, at the existing ordering
step in `Orchestrator.stream:222`. The wave-1 entry is a blocked result with
empty `tool_calls`, so `Evidence` ignores it either way — but the synthesiser
would otherwise render two sections under one specialist heading.

---

## 7. The session

New module: `orchestrator/conversation.py`.

```python
@dataclass
class Turn:
    request: str          # redacted
    answer: str           # redacted
    agents: list[str]

class Conversation:
    session_id: str
    phi: PHISession
    facts: Facts
    turns: list[Turn]
    evidence: list[tuple[int, SpecialistResult]]   # §10, optional
    lock: asyncio.Lock
    last_seen: float
```

The entry point takes the conversation rather than looking it up, keeping the
composition root the only place that knows the store exists:

```python
async def stream(
    self,
    request: str,
    trace: Trace | None = None,
    conversation: Conversation | None = None,
) -> AsyncIterator[dict[str, Any]]:
```

`None` reproduces today's behaviour exactly — a fresh `PHISession`, empty
Facts, no history — so `run()`, the eval harness, and every existing test call
it unchanged.

### 7.1 Identity and lifetime

**The server mints the session id.** A client-chosen identifier means guessing
someone else's id hands you their PHI vocabulary. The first request omits
`session_id`; the `done` event returns the one the server created; subsequent
requests echo it.

The frontend holds it in a `useRef`, not `localStorage`, so a refresh drops it
and the orphaned session expires on its own. This is the requested behaviour:
memory lasts as long as the tab, and no longer.

`ConversationStore` evicts on an idle TTL (`session_ttl_s`, default 1800) and
caps concurrent sessions (`max_sessions`, default 200). Nothing is written to
disk.

### 7.2 Remembering everything, prompting with almost nothing

The store keeps the whole session. What enters a *prompt* is a separate
question, and the answer is deliberately narrow — `ollama_num_ctx` is 16384 and
the specialist answer prompt already carries up to 12000 characters of tool
output.

| Consumer | Receives | Why |
|---|---|---|
| Router | last 3 turns, compact | Resolving *"her"* is a routing problem |
| Specialist | Facts only | Needs identifiers, not narrative |
| Synthesiser | current turn only | Unchanged |

Specialists never see conversation history. Their prompts stay exactly the size
they are today regardless of how long the conversation runs, and Facts is a
handful of key–value pairs that does not grow with turn count.

The router's history block is labelled as context and never as the request, so
a prior turn cannot be mistaken for the thing being asked.

### 7.3 Concurrency

Two requests on one session id would both mutate the same `PHISession`
mid-redaction. Each turn takes the conversation's `asyncio.Lock`; a second
concurrent request on the same session waits. Three lines, and it removes a
class of bug that is miserable to reproduce.

---

## 8. Surfacing it

Memory nobody can see is memory that cannot be demonstrated. `OrchestratorOutcome`
gains:

```python
session_id: str
facts: list[dict[str, str]]      # key, value (placeholder), source, turn
```

The UI renders a small panel beside the trace — *"tracking: PHI_PATIENT_1
(Clinical, turn 2)"* — that updates as the conversation proceeds. The
follow-up question becomes a visible consequence of state on screen rather than
an unexplained success.

`SpanKind.MEMORY` is added to `observability/trace.py` and to the `SpanKind`
union in `frontend/src/api.ts`.

---

## 9. Files touched

| File | Change |
|---|---|
| `orchestrator/facts.py` | **new** — `Fact`, `Facts`, `@provides` registry, extractors |
| `orchestrator/conversation.py` | **new** — `Turn`, `Conversation`, `ConversationStore` |
| `guardrails/phi.py` | `PHISession.tokenize()` |
| `orchestrator/registry.py` | `SpecialistSpec.needs`; `provides` derived |
| `agents/catalog.py` | `needs` on Revenue Cycle and Remote Monitoring |
| `agents/base.py` | `SpecialistResult.blocked`; `ESTABLISHED FACTS` prompt block; accept `facts` |
| `orchestrator/graph.py` | `wave` counter, retry edge, fact extraction, dedupe, conversation plumbing |
| `orchestrator/router.py` | optional history block |
| `observability/trace.py` | `SpanKind.MEMORY` |
| `api/main.py` | `session_id` in and out; store lifecycle |
| `bootstrap.py` | `ConversationStore` alongside `default_orchestrator` |
| `config.py` | `session_ttl_s`, `max_sessions`, `history_turns` |
| `frontend/src/api.ts` | `session_id`, `facts`, `memory` span kind |
| `frontend/src/App.tsx` | session ref, facts panel |
| `examples.py` | a two-hop example and a follow-up pair |

---

## 10. Cross-turn reconciliation (last item, safe to cut)

With the session retaining redacted `SpecialistResult` objects, `reconcile` can
be handed prior turns' evidence alongside the current turn's. Turn 2 pulls the
claim; turn 6 pulls the chart; the existing
`billed_diagnosis_matches_problem_list` check fires across them with no new
comparison logic.

This is the strongest thing the system could do in a demo. It is listed last
because it is the only item here that changes `reconcile.py`: `Evidence` must
tag each output with the turn it came from so `Finding.provenance` can say so,
and the two helpers `_claims` and `_patient` must carry that tag through.

It also carries the one honest risk in this document. Cached evidence goes
stale, and *"is that claim still denied?"* has no good answer from a copy taken
four minutes ago. Against fixed fixtures the risk is zero, which is exactly why
it needs saying out loud rather than discovering later.

There is a second interaction to handle. `_in_scope` narrows findings to the
records the request names, falling back to *all* findings when it names none —
a sensible default when the evidence came from one turn. Across a session that
discussed four claims, a follow-up naming none would report findings for all
four, which is the exact failure `_in_scope` was written to prevent
(*"handing eight findings to a 4B model for a question about one claim is how a
correct finding set produces a bad answer"*). The scope must therefore fall
back to the conversation's current subject before falling back to everything.

Cut this section and everything above still works.

---

## 11. Testing

Offline against `StubProvider`, as the existing suite runs.

**Facts and redaction space**
- An extracted `patient_id` is stored as a placeholder, never as digits.
- A value named in the request and later resolved from FHIR yields one token.
- `found: False, ambiguous: True` contributes no fact.
- Rendering a Facts prompt block emits no digit sequence matching a fixture id.

**Grounding, unchanged**
- A fact-derived placeholder passes `_is_grounded`.
- An identifier appearing in neither the request, the session, nor Facts is
  still rejected. *This test must not weaken.*

**Two-wave dispatch**
- Blocked specialist + matching fact → exactly one retry.
- Blocked specialist + no matching fact → no retry.
- Unblocked specialists are never retried.
- Wave count never exceeds 2, including when wave 2 is also blocked.
- A retried agent appears once in `outcome["agents"]`.

**Session**
- Token stability: `PHI_PATIENT_1` denotes the same person at turn 5 as turn 1.
- Two session ids never share a vocabulary or a Facts store.
- Subject switch drops the previous patient's facts.
- Idle TTL evicts; eviction releases the `PHISession`.
- Concurrent requests on one session serialise rather than interleave.
- An unknown or omitted `session_id` starts a new conversation rather than
  erroring.

**Regression**
- `test_orchestrator.py`, `test_reconcile.py`, `test_phi.py`,
  `test_answer_grounding.py` pass unchanged.
- `test_specialists_do_not_share_tools` still passes — nothing here grants a
  specialist a tool it did not own.

**Evals** — two datasets under `evals/datasets/`: two-hop questions within one
turn, and follow-up sequences across turns. The README already commits the
fixture generator so the synthetic-data claim is *checkable rather than
asserted*; the same standard applies to *"follow-ups work"*.

---

## 12. Known limits to add to the README

- **Two waves inside one question.** A three-hop question fails. The cap is a
  counter, not a judgement, so it cannot stretch when a question genuinely
  needs more.
- **Only declared facts travel.** An unanticipated hop fails exactly as it does
  today. Adding one is an extractor plus a `needs` entry, not a new code path.
- **One subject at a time.** Facts follow the most recently resolved patient.
  Switching subjects mid-session drops the previous one's identifiers rather
  than keeping both.
- **Nothing survives a restart, or a refresh.** By design, and also why this is
  a demo rather than a product.
- **The PHI vocabulary now lives for a session rather than a request.** In
  memory, never on disk, evicted when idle — but a longer window than before,
  and it should be described as one.
- **Still no negotiation.** A blocked specialist reports being blocked. It
  cannot ask for what it needs; the system either happens to hold the missing
  piece or it does not.

---

## 13. Rejected alternatives

**Direct agent-to-agent messaging.** Would let Clinical ask Revenue Cycle for a
claim, and would dissolve the data-plane exclusivity that makes routing
decidable. It also reintroduces the unbounded dialogue that the single planning
round exists to prevent. The blackboard gets the same information across
without either cost.

**An LLM replanner deciding whether to run wave 2.** Broader coverage of
unanticipated gaps, at the price of a 4B model making an open judgement on
every multi-agent request, an extra round trip, and a trace that records a
rationale instead of a fact. Rejected for the same reason `reconcile.py`
computes its joins in code.

**A supervisor loop running until satisfied.** The plan–act–observe cycle
`agents/base.py` already documents as failing on this model class.

**A LangGraph checkpointer keyed by `thread_id`.** The textbook answer, and it
fights the state: `results` is `Annotated[list, operator.add]`, so persisting
across turns would append turn 2's results to turn 1's indefinitely. The reset
logic needed to undo that is larger than the store it replaces.

**Widening `_is_grounded` to accept fact-derived values.** Superseded by §4.1.
Keeping Facts in redaction space achieves the same result with no change to a
guard whose whole value is that it has not been relaxed.

**Durable cross-session memory.** Out of scope per §2.

---

## 14. As built: where implementation departed from this design

Status changed to **implemented**. Five departures, four of them found by
running the thing against the live model rather than against a stub.

### 14.1 `blocked` needed a second definition

The design set `blocked` where the grounding guard rejects an invented
identifier. Against `qwen3.5:4b` that path almost never fires for a two-hop
question. What actually happens: asked about a patient by name, the planner
passes `PHI_NAME_1` as `patient_id`. That is a real placeholder from the
request, so `_is_grounded` accepts it — correctly — and the telemetry plane,
which keys by id, matches nothing and returns `found: false`.

Wave one therefore "succeeded" and answered *no data*. No retry, no second
wave, feature invisible.

`_retrieved_nothing` adds the missing half: every lookup a specialist ran
reported no match. Only an explicit `found: false` counts, so `policy_search`,
which returns prose, is never read as empty.

### 14.2 An omitted scoping argument is not harmless

`telemetry_series(patient_id="")` does not fail — it reads every device in the
store. A question about one person silently became a scan across all of them,
and the specialist then reported "no data" with complete confidence.

Now a specialist that declares `needs` a key, on a call carrying no identifier
at all, with a request that plainly names somebody, declines instead of
widening — and reports itself blocked, which is what earns it the second wave.

The qualifier matters: `claim_lookup(claim_id=...)` is fully scoped and wants
no patient id. An earlier version without it blocked that call and cost a
pointless second wave, which `test_the_reconcile_span_is_in_the_trace` caught.

### 14.3 Memory outranked an explicit instruction

The worst bug of the build, found by `conv-04`. Asked *"what is A taking?"* and
then *"now look up B and show their labs"*, the planner used the `patient_id`
on the board — A's — and reported that no data existed for B. Memory beat the
request, and the system answered confidently about the wrong patient.

The rule is now that the request always wins: established facts are offered to
a planner only when the request names nobody of its own. The second wave does
not depend on this, because it substitutes the resolved identifier into the
argument directly rather than suggesting it to the model — deterministic, and
nothing a prompt can talk it out of.

### 14.4 A regex that was never true

`_names_a_subject` was written as `\bPHI[_ ]?(?:NAME|PATIENT|...)\b`. The
trailing `\b` never fires: `_` is a word character, so there is no boundary
between `NAME` and `_1`. The predicate returned False for every input, and both
§14.2 and §14.3 were silently inert until a direct probe showed it.

Worth recording because the tests passed the whole time it was broken — they
were asserting the old behaviour, which a dead predicate faithfully preserved.

### 14.5 The chart arrives from either of two tools

`billed_diagnosis_matches_problem_list` required `fhir_search_patient`. Asked
*"is that diagnosis on their chart?"*, the planner reaches for
`fhir_get_resource(resource="conditions")` at least as often — same `{code,
display}` rows, different key. The check never fired.

Pre-existing, not caused by this work, but it is precisely what §10's
cross-turn reconciliation needed in order to ever demonstrate. `_patient` now
accepts both shapes, and `requires` names only the claim side because it has no
way to express "either of these".

### 14.6 Verification

- `backend/tests/`: **185 passed**, offline, no model.
- `evals/run_eval.py`: **97.1%** overall exact match — unchanged from before
  this work, as expected: routing sees no history on a standalone request.
- `evals/conversations.py`: **6/6** conversations, live model.
- `evals/phi_leak.py`: **12/12**, zero inbound, audit, or answer leaks.
- Live cross-turn check across a subject switch: facts stay placeholders, no
  real identifier reaches any trace span, and the subject follows the most
  recently named patient.

### 14.7 Not built

`Finding.from_turn` was specified in §10 and dropped. A finding computed from
evidence retrieved three turns ago reads identically to one computed this turn.
Provenance names the tool and field, so the comparison is checkable; what is
missing is *when* the underlying record was read. Against fixed fixtures that
distinction is invisible, which is exactly why it should be written down rather
than left to be discovered.

---

## 15. Follow-on: identity, not names

Added after the memory work, because the same live testing kept surfacing the
same root cause.

### 15.1 One field meant three things

`fhir_search_patient` took a single `patient_id` argument documented as "patient
identifier, MRN, or full name". That overload caused two unrelated failures.

A name reached a plane that only knows ids. The planner passed `PHI_NAME_1` as
`patient_id` to `telemetry_series`, which filtered on it, matched nothing, and
reported "no monitored device for this patient" - a statement about the
argument dressed as a statement about the patient. §14.1's `_retrieved_nothing`
recovered from this; it should not have had to.

And a lookup could not say whether it had been handed a bad identifier or a name
it should have searched on, so the caller had no way to correct itself.

Now `patient_id`, `mrn`, and `name` are three fields. A name lookup is
explicitly a search that may match several people; an id or MRN is an exact
fetch. `name` joins `_must_be_grounded`, because a fabricated name selects the
wrong person as effectively as a fabricated id and looks more like something the
user said.

### 15.2 Capability decides who refuses and who repairs

Separate fields tell a model what each one means. They cannot make it comply,
and the plan is fixed in one round with no chance to observe a complaint and
retry - the live model kept putting names in `patient_id` regardless of the
description.

So the rule is capability, not policy:

- `store.match_patients` **repairs**. It routes by shape - digits to
  `patient_id`, `MRN-` to `mrn`, anything else to `name` - because this plane
  can resolve a name, so the honest reading of `patient_id="Samuel Ferreira"` is
  a name search. `Router._normalise` sets the precedent for repairing output
  that satisfies the schema but not the intent.
- `telemetry` and `claims` **refuse**, with `invalid_key`, because they hold no
  way to resolve a person from a name. A refusal is also what earns a second
  wave; a silent empty result did not.

### 15.3 Ambiguity became a conversation

A name matching several patients used to be a full stop: the asker retyped the
whole question with an MRN attached. `orchestrator/disambiguate.py` holds the
blocked question on the conversation and resumes it when the next turn supplies
one identifier.

    "What medications is Samuel Ferreira taking?"
    -> 2 patients share that name. Supply an MRN, a patient id, or a date of birth.
    "MRN-217621"
    -> Fluticasone/salmeterol (250/50 mcg) twice daily.

Deterministic throughout - no model call decides any of it. An exact identifier
is substituted for the name token; a date of birth is appended naming the
`birth_date` argument, because it narrows rather than identifies. A turn
carrying two identifiers, or one that asks a question of its own, is a new
request and the held question is dropped rather than answered over the top of it.

An earlier phrasing appended the date parenthesised - `"... taking? (date of
birth PHI_DOB_1)"` - and the planner searched for a patient *called*
`1957-03-18`. Naming the argument is what makes it land.

### 15.4 What may be disclosed

**How many** patients matched, never **which**. A count is what the asker needs
in order to understand the refusal and know what would fix it. A candidate list
- names, MRNs, dates of birth - is a disclosure about people the asker has
established no business with, and enumerating them is the same disclosure as
browsing the store. A real EHR shows that picker behind authentication; this
system has none, so it does not. `conv-07` asserts the refusal contains neither
twin's MRN nor either birth year.

### 15.5 Verification

- `backend/tests/`: **207 passed**, offline.
- `evals/conversations.py`: **9/9**, live - including disambiguation by MRN and
  by date of birth, and an unrelated turn that must drop the held question.
- Routing and PHI-leak evals unchanged.
