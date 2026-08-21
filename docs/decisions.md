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
