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
