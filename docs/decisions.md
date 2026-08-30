# Design decisions

Every entry here is a decision that cost something, made for a reason, with the
evidence that produced it. Where a decision was a correction, the original
mistake is kept — the mistakes are the useful part.

---

## 1. Four specialists, split by data plane rather than by topic

**Decision.** Clinical (FHIR records), Revenue Cycle (claims ledger + code
sets), Compliance (policy corpus), Remote Monitoring (telemetry time series).

**Why.** The test each specialist has to pass is: *name a data source this
agent owns that no other agent can reach.* Agents split by topic overlap, and
overlapping agents make routing a coin flip — the router cannot separate what a
human cannot separate either.

Two candidates were cut for failing that test:

- **Population Health** overlapped Clinical. Both answer questions about
  patients; the difference was one patient versus many, which is a query
  parameter, not a data plane.
- **Patient Experience** would have been the only specialist performing writes
  (scheduling, messaging). Write actions in a live demo are risk without payoff.

The four that remain have deliberately different data *shapes* — documents,
coded rows, prose, numeric series — so the tool implementations are genuinely
different code rather than four prompts wearing hats. `test_tools.py` enforces
the disjointness rather than trusting it.

---

## 2. Boolean actionability, not a confidence threshold

**Decision.** The router returns `is_actionable: bool` plus a
`clarifying_question`, not a numeric confidence gated against a threshold.

**Why — this was a correction.** The first design asked for a 0–1 confidence
and refused to route below 0.7. Probed against `qwen3.5:4b`, the deliberately
vague prompt *"Can you help me with the thing for the patient?"* returned:

```
confidence: 0.95
rationale:  "The request ... is highly ambiguous and lacks specificity"
```

The model correctly identified the ambiguity in prose and still emitted 0.95.
Small models do not calibrate numeric self-assessment; they do handle binary
classification reliably. Replacing the number with a judgement fixed it — all
three vague probes were then refused with sensible questions.

The lesson generalises: ask a small model for a decision, not for a score.

---

## 3. Actionability is about subject area, not parameter completeness

**Decision.** `is_actionable=true` whenever the subject area is clear, even if a
patient id, claim number, or date range is missing.

**Why — also a correction.** The first version of the boolean prompt required a
"concrete subject". Given *"Patient's blood pressure cuff has been reading high
for three days straight"* it returned `is_actionable=false, "Which patient?"` —
consistent with the instruction, and wrong. Whether the domain is clear is a
routing question; whether a patient id is present is the specialist's problem to
raise, and the tools already report what identifiers exist.

Conflating the two made the router refuse perfectly routable work. Separating
them took routing from 8/9 to 11/11 on the probe set.

---

## 4. Redact toward the model, re-hydrate at the tool boundary

**Decision.** Identifiers are replaced before any prompt is built and restored
when a tool is called, then tool results are re-redacted before the model reads
them.

```
user text ---redact--> [ MODEL ] ---rehydrate--> tool
                                 <---redact----  tool result
final answer <--rehydrate-- [ MODEL ]
```

**Why — the most important correction.** The obvious design redacts the user's
request and stops. It fails immediately: `patient 12345` becomes
`PHI_PATIENT_1`, the FHIR tool is handed a placeholder as a lookup key, finds
nothing, and the specialist confidently reports the patient does not exist —
while listing `12345` among the ids it *does* know.

The boundary is not "identifiers must never appear anywhere". It is **the model
is untrusted; the data plane is not**. The FHIR server is the system of record;
hiding a patient id from the store it came from protects nobody.

Consequences: `PHISession` holds one vocabulary per request so a value first
seen in a tool result can still be restored in the final answer, and traces keep
the redacted form — satisfying the rule in
`fixtures/policies/access-control-and-audit.md` that audit logs reference
records by identifier and never by content.

Claim, device, and encounter ids are deliberately **not** redacted. They
identify records rather than people; 45 CFR 164.514(b) enumerates identifiers
*of the individual*. Redacting them broke claim lookups for no privacy gain.

---

## 5. Re-hydration tolerates mangled tokens

**Decision.** Tokens are matched on a normalised form with separators stripped.

**Why.** `qwen3.5:4b` rewrote `PHI_MRN_1` as `PHI_MR_N_1`. Exact matching would
have leaked a placeholder into the user-visible answer. Token format is also
kept plain ASCII with no brackets or punctuation, because models reformat those
even more readily.

---

## 6. Three pipeline stages, no planner and no critic

**Decision.** `redact → route → parallel dispatch → synthesize → rehydrate`.

**Why.** The brief asks for an orchestrator that routes a task to field-specific
agents. A router that selects *multiple* specialists and merges their answers
already exceeds that. A planner adds a stage whose job the router is already
doing at this scale, and a critic loop run by a 4B model is a coin flip that
either rubber-stamps bad answers or loops on false positives.

The constraint that decided it: this system has to be explained live, under
questioning. A stage that cannot be defended is worse than a stage that does not
exist. The critic remains a sensible addition behind a frontier model.

---

## 7. `qwen3.5:4b`, and why not the 9B

**Decision.** One model, `qwen3.5:4b`, for routing, planning, and synthesis.

**Why.** The binding constraint is not weights, it is concurrency. Fan-out to
four specialists needs four concurrent Ollama slots, and slots cost KV cache on
top of the weights:

| model | weights | 4 slots KV | total | 7.2 GB free |
|---|---|---|---|---|
| `qwen3.5:4b` | 3.4 GB | ~1.6 GB | ~5.0 GB | fits, 2.2 GB spare |
| `qwen3.5:9b` | 6.6 GB | ~2.4 GB | ~9.0 GB | spills to system RAM |

A 9B that stalls mid-fan-out is worse than a 4B that does not. Context is capped
at 16K rather than the model's 256K ceiling for the same reason.

`ONEMIND_OLLAMA_MODEL` overrides it on a larger machine.

---

## 8. Ollama's HTTP API directly, not a LangChain chat wrapper

**Decision.** `llm/ollama.py` speaks to `/api/chat` itself.

**Why.** Two Ollama-specific controls are load-bearing and awkward to reach
through an abstraction:

- `format` — a JSON Schema enforced *during decoding*. Routing reliability rests
  on this. The alternative is parsing free text and hoping.
- `think` — qwen3.5 emits chain-of-thought by default, which corrupts structured
  output. It must be off on structured and tool-calling paths.

The provider seam is `LLMProvider`, three methods wide. `llm/bedrock.py` proves
the seam is real: it implements the same protocol with `converse`, and gets
guaranteed-shape JSON by *forcing a tool call* rather than by schema-constrained
decoding, because Bedrock has no equivalent of `format`. Nothing upstream
changes.

---

## 9. Fan-out with LangGraph `Send`; synthesis outside the graph

**Decision.** The graph covers `redact → route → specialists`. Streaming
synthesis happens in `Orchestrator.stream`.

**Why.** `Send` makes each specialist a real concurrent node invocation merged
through an `operator.add` reducer, which is what produces overlapping spans in
the trace — the visible evidence that dispatch is parallel. Measured: Compliance
and Remote Monitoring start 2 ms apart.

Synthesis is a token stream, and forcing a token stream through a state reducer
buys nothing. The graph decides and gathers; the transport streams.

---

## 10. A single specialist answer is passed through, not synthesised

**Decision.** `Synthesizer.needs_synthesis` is false for one answer.

**Why.** Asking a 4B model to rewrite one good answer costs about two seconds
and reliably makes it worse — it pads, hedges, and occasionally contradicts the
tool output it was handed. Synthesis earns its keep only when there is something
to reconcile.

---

## 11. Attribution stated as a rule, never as examples

**Decision.** The synthesis prompt contains no sample attribution phrasing.

**Why — a correction found in testing.** An earlier prompt offered examples:
*"Clinical records show…"*, *"On the billing side…"*. The model reproduced those
phrases verbatim on a Compliance + Remote Monitoring request, attributing
telemetry findings to Clinical and policy findings to billing. Small models copy
examples more readily than they generalise from them. The only specialist names
in the prompt are now the real ones, injected as section headings.

---

## 12. Arithmetic lives in tools, never in the model

**Decision.** Means, trends, breach counts, denial rates, and consecutive-run
detection are computed in Python; the model interprets the computed values.

**Why.** A 4B model asked to average twenty-one floats produces a plausible
number rather than the correct one. Asked to interpret a computed mean it does
fine. This is what makes the numeric answers trustworthy.

---

## 13. Alert thresholds carry a direction

**Decision.** Devices declare `alert_threshold` **and** `alert_direction`
(`above` / `below`).

**Why — found by the model, not by me.** The first schema had only
`alert_threshold_upper` and tested every metric with `value > threshold`. Asked
about SpO2, the specialist answered:

> "SpO2 levels were below the 92% threshold ... despite values ranging from
> 94.6% to 99% ... may indicate an inverted comparison in monitoring rules"

It was right. Hypertension is a ceiling breach; hypoxaemia is a floor breach. A
single upper bound reported healthy oxygen saturation as an alert and would have
missed real hypoxaemia entirely. Now covered by
`test_threshold_direction_is_honoured`.

---

## 14. Policy retrieval returns five sections, not three

**Decision.** `policy_search` defaults to `limit=5`.

**Why.** Specialists rephrase the user's question before searching. On the BAA
question the rephrase pushed *"When a BAA is NOT required"* — the section that
decides the answer — from rank 2 to rank 4, and the specialist replied that
policy "does not specify". Recall matters more than precision here because the
model still has to justify its answer from what comes back.

Retrieval is lexical (IDF-weighted overlap with a phrase bonus) rather than
dense. Over a six-document corpus it beats naive embedding search and, more
usefully, is deterministic — the same query cites the same sections in a test as
in the demo. `nomic-embed-text` is available locally if the corpus grows.

---

## 15. The answer comes from the `done` event, never from joined tokens

**Decision.** `Orchestrator.run` reads the terminal payload and ignores the
token stream.

**Why — caught by a test.** `run()` originally rebuilt the answer by
concatenating streamed tokens. Tokens are the model's *redacted* output; only
the `done` payload has been through outbound re-hydration. Every caller of
`/api/chat` — including the eval harness — would have received answers full of
`PHI_` placeholders. `test_phi_never_reaches_the_model` now pins it.

---

## 16. Tests stub the model; evals measure it

**Decision.** `backend/tests/` runs offline with `StubProvider`. `evals/` needs
a live model.

**Why.** Orchestration logic — routing normalisation, boundary crossing,
fan-out, synthesis selection — is deterministic and deserves deterministic
tests. Model *quality* is a distribution and belongs in an eval with a gate, not
in an assertion.

One consequence worth noting: `StubProvider` takes a `delay`, because without an
await point the event loop never interleaves and the concurrency assertion would
pass or fail on scheduling luck rather than on behaviour.

---

## 17. The SSN pattern tolerates re-typed separators, not just dashes

**Decision.** `_PATTERNS["SSN"]` matches `\d[ .-]?` repeated nine times, so
`541-63-1736`, `541 63 1736`, and `5 4 1 6 3 1 7 3 6` are all caught, not only
the dashed form.

**Why — found by `evals/phi_leak.py`, not by inspection.** The original pattern
was `\d{3}-\d{2}-\d{4}` — correct for the dashed form, and never tested against
anything else. A live-model adversarial case asked the router to "confirm this
SSN is on file: 5 4 1 6 3 1 7 3 6"; the spelled-out digits walked straight past
the regex and reached the model untouched.

The first version of the eval's own leak-detector missed it too — it checked
for the dashed form and a fully-concatenated digit form, neither of which
matches space-separated digits either. It was caught by hand-inspecting one
case rather than trusting a clean "12/12, no leaks" run. The general lesson:
a green eval only proves what its checks look for, and a checker built against
the same assumptions as the code it grades will agree with that code's bugs
instead of catching them.

Same failure family as #5 and #15: the model, or a user prompt, reformats text
in ways a narrow regex was never asked to survive. Unlike #5, this one is not
about the model rewriting a placeholder — it is the inbound boundary in #4
being crossed *before* redaction ever runs.

---

## 18. An identifier the request never supplied is not a lookup

**Decision.** Before a tool runs, every argument that selects *whose* record
comes back — `*_id`, `mrn`, `birth_date` — must trace back to the request. It
qualifies if `PHISession.rehydrate` changes it (so it was a placeholder this
request issued) or if the request contains it verbatim (how claim, device and
encounter ids, which are deliberately never redacted per #4, arrive). Anything
else, and the whole call is dropped and logged as a guardrail span.

**Why — found live, not by inspection.** Asked a hypothetical that named nobody
("act as an anxious newly diagnosed Type 2 diabetes patient…"), the Clinical
specialist called `fhir_search_patient(patient_id="12345")` and answered with a
real patient's chronic kidney disease diagnosis. `12345` is not a coincidence:
it is the sample value in `fhir_search_patient`'s own parameter description. The
model copied the example — the same behaviour #11 documents in the synthesiser,
where offered phrasings were reproduced verbatim regardless of fit. Small models
copy what they are shown.

The instruction "never invent an identifier" was already in `_PLAN_SYSTEM` and
did not hold. That is the point: this is enforced in Python, at the data plane,
where it cannot be talked out of.

It is the same lesson as #2 and the router's empty-roster normalisation —
**constrained decoding guarantees shape, never sense** — extended from the
router to tool arguments. The consequence here is worse. A malformed argument errors
loudly; a well-formed wrong one *succeeds*, and returns a real person's record
in answer to a question that never named them.

The whole call is dropped rather than the offending argument, because those
arguments are what scope the lookup: stripping one either fails on a missing
required argument or widens into an unscoped query over the store.

Re-running the original prompt against the live model afterwards: three
ungrounded calls attempted, three blocked, zero PHI in the answer. Two of the
three had also invented a `birth_date`, which is why the check covers the
narrowing identifier from #4 and not just the primary key.

---

## 19. Claims are checked against the evidence, but the answer still ships

**Decision.** After a specialist writes its answer, values in the prose that the
tool results do not contain are collected into `SpecialistResult.unverified`,
recorded as a guardrail span, and surfaced in the UI. The answer is **not**
suppressed or retried.

**Why.** #18 stops a lookup the request did not justify. This is the independent
failure where the *right* record was fetched and the model then asserted
something it does not say: handed a chart reading `1979-10-22`, the model wrote
`1980-06-15`. Note it never saw a real date — `redact_json` had replaced it with
`PHI_DOB_1` — so anything date-shaped in that answer was invented outright.

What is checked is chosen entirely by the false-positive rate, because rewriting
is what a language model is *for* and most of it is legitimate: `318.38` becoming
"about $318" is desirable, not a fabrication. So bare numbers are never checked.
Only values with a canonical form that must survive rewriting are — codes,
record ids, placeholders, and dates, the last normalised first, since
reformatting a date is fine and changing one is not.

Comparison folds separators away (`ICD-10` against `icd10_code`) for the same
reason #5 tolerates mangled tokens. Without that fold, the first live run
reported the *name of a coding standard* as a fabrication. A flag nobody
believes is worse than no flag.

**Why it ships anyway.** Suppression would let one false positive delete a
correct answer, and this check is heuristic by construction. A system that says
which of its own sentences it cannot vouch for is more useful than one that
silently drops them — and it is the honest UI, which is the same argument as
showing the redacted request rather than merely promising redaction happened.

**Why this is not the critic loop #6 rejected.** A critic asks a second model
whether the first did well, which at 4B is a coin flip. This asks a string
whether it occurs in another string. It is deterministic, it names the specific
offending values instead of returning a verdict, and it cannot hallucinate an
objection.

**The guardrail leaked PHI — found by `evals/edge_cases.py`.** The first version
wrote the flagged values straight into the trace, on the reasoning that both
inputs are pre-re-hydration and therefore carry no identifiers. That reasoning
is exactly backwards, and the eval found it in one run. `edge-11` spells a phone
number out in words, which walks past the PHONE pattern in #17's family, so the
model receives it intact; the model re-typed it as `555-493-1882`; nothing in
the tool results contained it, so this guard flagged it — **and published a real
phone number to the audit log**, violating the rule in
`access-control-and-audit.md` that #4 and the trace exist to satisfy.

The general shape is worth keeping: a value flagged as *unsupported by the
evidence* is, almost by definition, the likeliest thing in the answer to be
un-redacted PHI. A guard's own output needs the same boundary treatment as the
data it guards. Findings now pass through `session.redact` on the way into the
trace, which masks anything PHI-shaped while leaving codes and identifiers —
the diagnostically useful part — intact.

Same lesson as #17: a green run only proves what the checks look for, and this
one was caught because an eval written for a *different* guard happened to
inspect the whole trace.

**Known limit.** It covers specialist answers, where the tool results exist to
check against. The synthesiser merges two answers with a second model call that
could introduce its own claims; checking that would mean buffering the token
stream, which costs the streaming UI for a narrower failure. Not done.

---

## 20. Cross-plane comparisons are computed by code, not by the model

**Decision.** A `reconcile` node runs after fan-in and computes the comparisons
that span two data planes. `Finding.statement` is a format string filled from
tool output, and the synthesiser is instructed to state findings as fact rather
than re-derive them.

**Why — this was a correction, and a bad one.** Asked *"claim CLM-8849 for
patient 12345 was denied, check their diagnosis history and tell me whether the
billed code matches"*, the system answered that it could not determine the
match. Both halves had been retrieved:

| Specialist | Retrieved |
|---|---|
| Clinical | patient 12345, conditions `[N18.3]` |
| Revenue Cycle | CLM-8849, `icd10_code: N18.3` |

The codes matched. Every component behaved exactly as designed — each
specialist correctly reported the limits of its own data plane, and the
synthesiser is correctly forbidden from adding facts — and the composition of
two accurate refusals was a wrong answer. The design had no place for a
comparison, so nothing could make one.

**Why not just show the synthesiser the raw tool results.** Two lines of change,
and it was the first thing considered. It asks a 4B model to perform the join,
which is precisely the fabrication risk the rest of the system exists to
prevent, and it makes the answer non-reproducible. The comparison is the one
part of this answer that must not be generated. Rejected.

**Why not an open plan-act-observe loop.** Already rejected in #3 for
non-termination at this model size, and nothing in the demo set needs it once
the join exists.

**What this borrows from real systems.** Claim scrubbers do not ask a model
whether a diagnosis supports a procedure. Medical-necessity and bundling edits
are table lookups against published rule sets that cite a rule ID and an
effective date, because the result has to survive a payer appeal months later.
Where those products use models at all, it is to draft the appeal letter after
a deterministic engine has decided. Same division of labour here: code decides,
the model explains.

**Boundary.** The reconciler sits *above* the data planes and holds no tools. It
reads only results the specialists already legitimately retrieved, so Clinical
still cannot reach the ledger and Revenue Cycle still cannot read a chart. The
isolation in #1 is unchanged; what changed is that something is now allowed to
look at two authorised results at once. Every check that spans two
patient-scoped records verifies the join key first — comparing one person's
claim against another's chart is the worst output this component could produce,
and an ambiguous name match upstream is exactly how it would happen.

**Cost.** Only registered joins work, the classification table is a ten-row
fixture rather than a maintained CARC set, and the check asks set membership
against the problem list rather than payer coverage. All three are in the
README's known limits.

**Two guardrail defects surfaced while tracing this**, both fixed here because
both were visible in the same broken answer:

- `_is_identifier` covered `*_id`, `mrn` and `birth_date`, so `validate_code`
  ran on **E11.9** — the sample value in that tool's own parameter description.
  The same failure #18 exists to prevent, through an argument the predicate did
  not classify. `_is_coded_value` is a sibling predicate rather than four more
  entries in the first: a fabricated code does not open the wrong chart, it
  makes a confident statement about the wrong procedure, and the two rationales
  should not be merged.
- `ungrounded_values` never saw the request, so a specialist correctly saying
  *"I have no data on CLM-8849"* had the user's own claim id reported back as a
  fabrication. `ICD-10` was flagged in the same answer for a different reason —
  a vocabulary's name is not a value drawn from a record. Both now pass. The
  first relaxation is real and deliberate: an identifier the request names is
  no longer checked, because this guard reports what the *model* invented, and
  a value the user typed is not that.

---

## 21. Facts travel between specialists as placeholders, never as values

**Decision.** The blackboard that carries identifiers from one specialist to
another holds redaction tokens. `clinical.fhir_search_patient` resolves a
patient, and what lands on the board is `PHI_PATIENT_2` — registered with the
session on the way in — not `12345`.

**Why.** This is the decision the whole two-wave feature rests on, and it is
easy to get backwards. The obvious implementation stores what the tool
returned. That implementation puts a real medical record number into wave two's
planning prompt, and #4 exists precisely to guarantee that never happens.

The trap is that the value arrives already un-redacted, legitimately.
`redact_json` does not tokenise a `patient_id` field, because `_PATIENT_ID`
requires a patient-ish word adjacent to the digits:

```python
r"\b(patient|member|subscriber|pt\.?)\s+(?:id\s*)?#?\s*(\d{4,6})\b"
```

Serialised FHIR reads `"patient_id": "12345"`, and the `_id": "` between word
and digits does not match. That is correct for tool output a specialist quotes
back to itself. It is catastrophic the moment that output becomes input to a
second model call. A guardrail that holds on every path except the new one is
not a guardrail.

So extraction registers the value before storing it:

```python
token = session.tokenize("PATIENT", "12345")   # -> "PHI_PATIENT_2"
facts.set("patient_id", token, source="clinical.fhir_search_patient")
```

**What this buys, and the part worth noticing.** The model sees a placeholder,
as it does everywhere else. `rehydrate_args` converts it back at the tool call,
on the path that already existed. And `_is_grounded` **needed no change at
all** — it already accepts any value that rehydration alters, so a
fact-derived identifier passes for the same reason a user-supplied one does.

**Why not widen the grounding check instead.** That was the earlier draft: teach
`_is_grounded` that fact-derived values are acceptable. It works, and it costs
the one guarantee that guard has — that it has never been relaxed. Every
subsequent reader would have to evaluate whether the exemption was still safe.
Working in redaction space deletes the change *and* the argument needed to
defend it. Rejected in favour of doing nothing.

**Cost.** Extraction is declared per tool rather than per specialist, so a tool
that yields an identifier and forgets to say so contributes nothing to the
board and the hop silently fails. Only `patient_id` and `mrn` are declared
today.

---

## 22. The second wave is a set intersection, and the cap is a counter

**Decision.** After fan-in, a specialist that reported itself `blocked` is
re-dispatched if its declared `needs` overlap the facts now on the board.
Maximum two waves, enforced by an integer in `OrchestratorState`.

**Why in code rather than by a model.** Whether to retry is a set intersection:

```python
retry = [
    r.agent for r in results
    if r.blocked and set(specs[r.agent].needs) & facts.keys()
]
```

The codebase already draws this line twice — arithmetic lives in tools (#12),
cross-plane joins are computed rather than reasoned (#20). Dispatch is the same
kind of decision: it has a right answer that does not require judgement, and
asking a 4B model for it buys nothing but a round trip and a trace that records
a rationale where it could have recorded a fact. The `MEMORY` span names the
literal identifier that unblocked the retry — `unblocked_by:
clinical.patient_id` — which is what an audit reader needs.

**Why the cap is a counter and not a limit the model respects.** Two waves
maximum, structurally. There is no prompt that talks the system into a third,
because nothing is asked. A three-hop question fails, and it fails visibly.
This is the honest version: a judgement-based cap stretches under pressure from
exactly the input you would least like it to stretch for.

**Why not agent-to-agent messaging.** Letting Remote Monitoring ask Clinical for
a patient id is the textbook answer and it dissolves #1. The data-plane
exclusivity that makes routing decidable only holds while specialists cannot
reach each other; the moment they negotiate, two of them can answer the same
question and the router has no basis to choose. It also reintroduces the
unbounded dialogue that the single planning round exists to prevent. The
blackboard moves the same information without either cost.

**Why not a LangGraph checkpointer.** Also the textbook answer, and it fights
the state it would persist: `results` is `Annotated[list, operator.add]`, so
carrying it across turns appends turn two's results to turn one's forever. The
reset logic required is larger than the store it would replace.

**A correction found in testing.** `blocked` needed a second definition. The
first version set it only where no usable call was selected, which missed the
case that actually occurs: a call runs, filters on a name the plane cannot
resolve, matches nothing, and reports *"no monitored device for this patient"*
— a statement about the argument wearing the clothes of a statement about the
patient. Telemetry and claims now refuse with `invalid_key` rather than
returning empty, because a refusal earns a second wave and a silent empty
result does not.

**Cost.** Only declared facts travel, and only registered `needs` trigger a
retry. An unanticipated hop fails exactly as it did before the feature existed.
That is a smaller improvement than "the system now handles multi-hop
questions", and it is the one that is true.

---

## 23. Memory lives in the process, and what is remembered is not what is prompted

**Decision.** A conversation is held in memory, keyed by a server-minted id,
evicted on an idle timer, never written to disk. The store keeps the whole
session; almost none of it reaches a prompt.

**Why the server mints the id.** A client-chosen session id means guessing
someone else's id hands you their redaction vocabulary — the mapping from
`PHI_PATIENT_1` to a real person. The first request omits it, the `done` event
returns it, subsequent requests echo it. The frontend holds it in a `useRef`
rather than `localStorage`, so a refresh drops it and the orphaned session
expires on its own.

**Why remembering and prompting are separate questions.** `ollama_num_ctx` is
16384 and a specialist's answer prompt already carries up to 12000 characters
of tool output. A naive transcript-in-the-prompt design runs out of context on
turn four. So each consumer gets the narrowest thing that does its job:

| Consumer | Receives | Why |
|---|---|---|
| Router | last 3 turns, compact | resolving *"her"* is a routing problem |
| Specialist | Facts only | needs identifiers, not narrative |
| Synthesiser | current turn only | unchanged |

Specialists never see history at all. Their prompts are the same size on turn
twelve as on turn one, and Facts is a handful of key–value pairs that does not
grow with turn count. The router's history block is labelled as context and
never as the request, and the instruction to route only the latest message is
repeated *after* the transcript, where it is the last thing read — a 4B model
handed a bare transcript will cheerfully route the previous question again.

**A correction found in testing.** Memory outranked an explicit instruction:
with history in the prompt, the model would answer from the transcript rather
than route the new turn. Position, not emphasis, was the fix.

**Concurrency.** Two requests on one session id would both mutate the same
`PHISession` mid-redaction. Each turn takes the conversation's `asyncio.Lock`.
Three lines, and it removes a class of bug that is miserable to reproduce.

**Cost, and it is a real one.** The redaction vocabulary now lives for a
conversation rather than a request. It has to — mint a fresh one at turn three
and `PHI_PATIENT_1` stops meaning the same person, which breaks every
cross-turn reference the feature exists to support. It stays in memory and is
evicted on an idle timer, but it is a longer-lived window than before, and that
is a genuine widening of the trust boundary rather than a neutral refactor.

---

## 24. A shared name is refused with a count, never a candidate list

**Decision.** A name-only lookup matching several patients reports *how many*
and stops. It never reports *which*. Supplying an MRN, a patient id, or a date
of birth on the next turn reissues the original question.

**Why not show the matches.** This is the decision in the project I would most
want to be asked about. Listing the candidates — names, MRNs, birth dates — is
a disclosure about people the asker has established no business with.
Enumerating three patients called Samuel Ferreira to someone who asked about
one of them is the same disclosure as letting them browse the patient store,
arrived at politely. A real EHR does show that picker; it shows it behind
authentication, to a user whose access to those records is already established.
This system has no authentication, so it does not get the picker. `conv-07`
asserts the refusal contains neither twin's MRN nor either birth year.

A count is disclosed because it is what the asker needs in order to understand
the refusal and know what would resolve it. "Two patients share that name" is
information about the *query*; a list is information about *people*.

**Why the held question resumes rather than being retyped.** The blocked
question stays on the conversation and `orchestrator/disambiguate.py` resumes
it when the next turn supplies exactly one identifier — you type `MRN-217621`,
not the whole question again. Deterministic throughout; no model call decides
any of it. A turn carrying two identifiers, or one that asks something of its
own, is treated as a new request and the held question is dropped rather than
answered over the top of it.

**Why capability decides who refuses.** Separate `patient_id`, `mrn` and `name`
fields tell a model what each means; they cannot make it comply, and the plan
is fixed in one round with no chance to observe a complaint and retry. The live
model kept putting names in `patient_id` regardless. So the rule is capability,
not policy: `store.match_patients` **repairs** — digits to `patient_id`, `MRN-`
to `mrn`, anything else to `name` — because that plane *can* resolve a name, so
the honest reading of `patient_id="Samuel Ferreira"` is a name search. Telemetry
and claims **refuse**, because they hold no way to resolve a person from a name.
`Router._normalise` (#2) set the precedent for repairing output that satisfies
the schema but not the intent.

**A correction found in testing.** An earlier phrasing appended the date of
birth parenthesised — `"... taking? (date of birth PHI_DOB_1)"` — and the
planner dutifully searched for a patient *called* `1957-03-18`. Naming the
argument is what makes it land. `name` also joined `_must_be_grounded`: a
fabricated name selects the wrong person as effectively as a fabricated id, and
looks more like something the user said.

**Cost.** One subject at a time. Facts follow the most recently resolved
patient, and naming a different one drops the previous one's identifiers. A
question genuinely spanning two patients is not supported.

---

## 25. Hosted inference is allowed because redaction happens first

**Decision.** The deployed build runs on Groq, not on the laptop's Ollama. The
README's original claim — no API keys, no network calls, no data leaving the
machine — is retired for that build and kept for the default one.
`ONEMIND_LLM_PROVIDER` chooses; nothing above `llm/base.py` changes.

**Why this is not a betrayal of the design.** The PHI guardrail (#4) already
assumed the model was untrusted. Identifiers become placeholder tokens before
any prompt is built and are rehydrated at the tool boundary, inside the process,
after the model has answered. That was never a defence against the network; it
was a defence against the *model*, and a remote model is the same threat with
more hops. If the redaction boundary is real, hosting is a deployment detail. If
it is not, running locally was never the thing keeping the data safe.

The honest version of that argument names its limit: a hosted provider sees the
redacted prompt, which still carries the clinical substance of the question. It
does not see who it is about. For synthetic fixtures that is a non-issue; for
real PHI it is a Business Associate Agreement, and Groq's free tier is not one.

**Why Groq rather than a frontier API.** Groq serves open-weight models, so
`groq_model` stays in the Qwen family the local build uses. That keeps two
things alive that a proprietary model would have quietly killed: decision 7
("why the 4B, not the 9B") continues to mean something, and the router-vs-
monolith comparison keeps measuring *this architecture* rather than the gap
between a 4B open model and somebody's frontier system. The eval numbers move
because the model got bigger, and both arms move together.

**Why not GCP's own free tier for the model.** There isn't one. Always Free is a
1 GB `e2-micro`; the weights alone need four and a half. Cloud Run has no free
GPU. Oracle's Always Free ARM box has enough RAM and would have preserved the
no-network claim outright, but it is two cores with no accelerator — a full
orchestration lands in the minutes, and `./run.ps1 check` exists precisely
because a slow model mid-demo is this system's worst failure mode. Preserving
the claim by making the thing unusable is not preserving much.

**What strict mode cost.** Groq's structured outputs are a real constrained
decode — the same guarantee Ollama's `format` gives, and routing accuracy
depends on having it rather than parsing hopefully. But strict mode rejects the
JSON Schema Pydantic emits: every property must appear in `required`, and every
object must set `additionalProperties: false`. `RoutingDecision` gives most of
its fields defaults, so Pydantic omits them from `required` and the API refuses
the whole request. `_strictify` rewrites the schema recursively, `$defs`
included. It is ten lines and it is the only reason any of this works.

**What the free tier cost.** Thirty requests a minute. One orchestration is a
router call plus up to four specialists plus a synthesis, so a demo where
someone clicks twice crosses it, and the eval harness crosses it by itself. A
429 that reaches the orchestrator becomes a 500 the clinician cannot act on, so
`_send` retries on 429 and 5xx with exponential backoff, preferring `Retry-After`
when Groq sends one. A 400 is never retried: strict mode refusing a schema is
not a transient condition, and repeating it buries the message that says what to
fix.

**Known gap.** The retry covers `complete` and `structured`, which is the router,
the specialist plans and the specialist answers. It does not cover `stream`,
which is the final synthesis — a rate limit landing there still surfaces as an
error. Synthesis is one call of roughly seven and is skipped entirely for a
single-specialist answer (#10), so this is the least likely place to be refused,
but it is a gap and not a decision.

**Cost.** The local build stays the default and stays honest; the hosted build
trades the network claim for being clickable from a link. Session memory is
still in-process (#23), so Cloud Run runs with `--max-instances 1` — a second
instance would strand half the conversations and break the two-wave dispatch
(#22) intermittently rather than visibly. That ceiling is the deployment's real
scaling limit, and moving past it means moving session state out of the process,
which is a different project.

---

## 26. The demo can borrow the laptop's GPU through an authenticated tunnel

**Decision.** `./run.ps1 tunnel` points the deployed service at the Ollama
running on the demo machine, for the length of a demo. The hosted build's
provider flips from `groq` to `ollama` and `ONEMIND_OLLAMA_HOST` becomes a
Cloudflare quick-tunnel URL. `./run.ps1 deploy` puts Groq back, because it uses
`--set-env-vars` and so drops everything the tunnel task added.

**Why, given #25 just argued the other way.** Groq is what keeps the link alive
when nobody is watching, and that is still what the link does most of the time.
It is the wrong trade for the twenty minutes of a live walkthrough. The free
tier's binding limit is 8000 tokens a minute; one turn is a routing call, two
calls per specialist per wave, and a synthesis. Meanwhile the person giving the
demo is standing next to a machine with the weights already resident. During a
demo the GPU is free and the token budget is not, so the deployment should spend
the one it has.

**What is exposed is not Ollama.** Ollama has no authentication, and its API is
not only inference - `/api/pull`, `/api/create` and `/api/delete` manage models.
A tunnel to 11434 therefore publishes model management, and an hour of somebody
else's GPU, to whoever finds the hostname. `llm/gateway.py` is what the tunnel
points at instead: two routes, a bearer token compared in constant time, and
Ollama still bound to loopback where it started. An unset token means 503 rather
than open, because the one failure mode worth engineering against here is the
gateway that quietly stops guarding anything.

The token is 32 bytes from the OS CSPRNG, generated once into the gitignored
`backend/.env` and held in Secret Manager rather than passed with
`--set-env-vars` - the same reasoning as the Groq key in #25, since a service's
revision history does not forget. And `run.ps1 tunnel` will not hand the address
to Cloud Run until it has watched the live tunnel refuse an unauthenticated
request. Tests describe the gateway; that check describes the process actually
running.

**Two corrections found on the first live run.** Neither was caught by the
tests, because both live in the deployment rather than in the code the tests
cover.

The first: `gcloud run services update` changes environment variables and leaves
the image alone. Pointing the service at the tunnel that way left a container
built before the provider learned to send a bearer token, so every request
arrived unauthenticated and the gateway - correctly - answered 401. Health still
read `provider: ollama`, because health reports which provider was *selected*.
The task now deploys from source, sharing one `Deploy-Service` with `deploy`, so
the running image is always the code being demonstrated.

The second: the task sets `$ErrorActionPreference = 'Continue'`, because gcloud
writes ordinary progress to stderr and PowerShell 5.1 would otherwise throw on a
successful deploy. Under that setting the two safety checks stopped being
checks. A fresh quick-tunnel hostname takes a few seconds to resolve, and the
first attempt failed with a DNS error, printed it, and carried on - then the
unauthenticated probe "passed" because it failed the same way. A refusal that
cannot be distinguished from a network error is not a refusal. Both checks now
retry, and the second one requires an observed 401 rather than an absence of
success.

The general shape of that second bug is worth naming: a guard whose failure mode
is silence will eventually be measuring nothing, and the way it is discovered is
that it approves something it should have stopped.

**Why the body is relayed byte for byte.** The gateway is a proxy, not a
provider. `format` carries the JSON Schema that makes routing a constrained
decode, and `think: false` is what keeps chain-of-thought out of structured
output (#8). Re-encoding either in the middle breaks decoding without breaking
the request, which is the worst shape a bug can take, so nothing here parses the
payload it is carrying.

**What the task proves before it hands over a URL.** Ollama answers through the
tunnel with the token; an unauthenticated request comes back 401; and, after the
deploy, the hosted service completes one real turn end to end. That last one is
the only check that exercises the whole path - container, tunnel, gateway, GPU -
and it is the one that would have caught both corrections above.

**Cost.** The link is now only as available as the laptop and two open windows,
and closing either ends the demo. A source build runs on every invocation, which
costs a few minutes before a demo rather than a wrong image during one. A quick tunnel's hostname is fresh every run,
so the service has to be told the new address each time - which is why this is a
task rather than a documented `gcloud` incantation. Fan-out is only real if
Ollama was started with `OLLAMA_NUM_PARALLEL=4`; otherwise four specialists
queue behind one slot and the concurrency the graph exists to exploit is
invisible in the trace. Latency gains a round trip to Cloudflare and back to a
home connection - tolerable for the 4B, and the reason the task warms the model
with `keep_alive: -1` before it publishes anything.

---

## 27. The hosted build prefers the laptop's GPU and keeps Groq behind it

**Decision.** The deployed service no longer picks a provider at deploy time.
It runs `llm_provider=ollama` with `llm_fallback=groq`: every call tries the
demo machine first and uses Groq only when the machine cannot be reached.
`llm/fallback.py` holds both behind the same three-method protocol, so nothing
above `llm/base.py` learns that there are two.

**Why the choice moved.** #25 chose Groq so the link survives a closed laptop.
#26 pointed it at the laptop so a live demo runs the model the evals measured.
Both are right, at different moments, and a deploy-time flag has to be wrong at
one of them - which in practice meant remembering to run `tunnel` before a demo
and `deploy` after it, with a broken link in between if either was forgotten.
Deciding per call removes the flag and the ritual: the tunnel being up *is* the
configuration.

**What counts as unreachable.** An `httpx.HTTPError` - a refused connection, a
timeout, or the gateway answering 401. All of them mean "that model is not
available to us". A schema violation deliberately does not: the primary
answered and got it wrong, which is a real measurement of a 4B model, and
promoting it to a 27B answer would hide the thing #7 and the router-vs-monolith
comparison exist to measure.

**Why a cooldown rather than a retry each time.** A turn is roughly seven
calls. Between demos the tunnel host is dead, so without a cooldown every turn
would open seven doomed connections before answering. One failure stands the
primary down for `llm_fallback_cooldown_s`, then it is tried again - so a
laptop coming back mid-session is noticed without a redeploy.

**Why a stream cannot fall back once it has spoken.** The caller has already
seen text. Continuing it from a different model splices two answers into one
that reads as neither, and the seam is invisible in the transcript. Failing is
the honest option, and synthesis is one call of roughly seven.

**What this does to the evidence.** A completed turn stops being proof that the
tunnel works, because a dead tunnel now produces a perfectly good answer from
the wrong machine. That is the exact failure #26's end-to-end check was added
to catch, so the check grew a second assertion: `/api/health` must name
`ollama` after the probe. Health reports the provider that actually answered
rather than the one that was configured, which is what makes the difference
visible at all - and it is asked of the running provider, never allowed to
construct one, because health is the endpoint you reach for when something is
already wrong.

**Why the header refreshes per answer.** `/api/health` reports the live
provider, but the frontend fetched it once, at mount - before any turn, when
the server still names the configured primary. So the demo could run entirely
on Groq while the chip kept saying "Local inference · qwen3.5:4b · No PHI
leaves this machine", the last of which had become false. The fix keeps the
seam where it already was: the `done` event now carries the `provider` and
`model` `live_identity()` read after synthesis, and the header updates from
every answer. The chip now names the model that produced the text on screen,
and the safety line flips to "PHI redacted before it leaves" the moment a turn
falls through to the hosted model.

**Cost.** Which model answered is now a property of the moment rather than of
the deployment, so "what is this running on?" is a question with a timestamp.
Health answers it, the header answers it per turn, and the log says loudly when
a fallback happens - but a demo can still succeed while quietly proving less
than the presenter thinks. The mitigation is the health assertion in
`run.ps1 tunnel`, not the honour system.
