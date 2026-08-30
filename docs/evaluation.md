# Evaluation: reading the results

The numbers below come from `evals/comparison_report.json`, regenerated with
`./run.ps1 compare`. This document explains what they show and what they do not.

## Routing accuracy

102 labelled prompts against `qwen3.5:4b` via Ollama, 3 attempts each (306 runs):

| metric | result |
|---|---|
| single-agent accuracy | **98.8%** |
| multi-agent exact match | 91.7% |
| abstain accuracy (vague prompts refused) | 90.0% |
| overall exact match | 97.1% |
| label precision / recall | 97.2% / 99.0% |
| router latency | p50 1235.4 ms, p95 1553.3 ms |
| stability (identical set on all 3 attempts) | **100.0%** |

Precision and recall are reported separately because their costs differ. Missing
a specialist yields an incomplete answer; waking a spare one costs a few seconds
and some noise.

## The router versus a single agent

The same prompts and the same scorer, against one agent that holds every tool and
has no router. Both arms get a constrained decode over the real tool descriptions
and the same way to abstain — see [`evals/arms.py`](../evals/arms.py) for why the
baseline is not a strawman, and `backend/tests/test_eval_arms.py` for the
fairness invariants pinned offline.

| metric | router | monolith | delta |
|---|---|---|---|
| single-agent accuracy | **98.8%** | 62.5% | +36.3 |
| multi-agent exact match | **91.7%** | 83.3% | +8.4 |
| abstain accuracy (vague prompts refused) | 90.0% | 90.0% | +0.0 |
| overall exact match | **97.1%** | 67.6% | +29.5 |
| label precision | **97.2%** | 83.3% | +13.9 |
| label recall | **99.0%** | 81.7% | +17.3 |
| selection latency p50 | 1235.4 ms | 1655.7 ms | -420.3 ms |

Three points stand out.

The comparison arm runs the same prompts and the same scorer against one agent
that holds every tool and has no router. Three points stand out.

**Abstention is a tie.** Both architectures refuse the same vague prompts at the
same rate. The expectation going in was that a model holding eight tools would
always want to use one; it does not. Routing buys nothing on the decision of
whether to answer at all.

**The gap is precision and recall on prompts that are actionable.** The
single-agent baseline reaches for a spare data plane on prompts that need only
one, and misses a plane on prompts that need two. Recall is the more expensive
side: a missing specialist produces a confident, incomplete answer, and nothing
downstream can detect that something is absent.

**The router is also faster.** Both arms are a single constrained decode. The
baseline's prompt carries eight tool descriptions where the router's carries four
one-line roster entries, and the difference shows up in time to first token.

**Neither result is run-to-run luck.** Both arms scored 100% stability: every
prompt produced an identical answer on all three attempts, misses included. The
routing failures below are deterministic.

## The three routing prompts the system gets wrong

Three of 102, each failing in a different way, all three failing on every
attempt. (The single-agent baseline fails 33.)

| id | prompt | expected | got | kind |
|---|---|---|---|---|
| `x-05` | *Are we allowed to keep this telemetry, and what is the device actually reporting?* | compliance + remote monitoring | remote monitoring | recall |
| `clin-05` | *Is the patient's A1c trending in the right direction?* | clinical | clinical + remote monitoring | precision |
| `amb-08` | *review the file* | abstain | clinical + revenue cycle | abstention |

`x-05` is the one to fix first. It is the only recall failure: the compliance
half of a two-part question is dropped, so the answer comes back fluent and
half-missing, and nothing downstream can detect the absence.

`clin-05` wakes a spare specialist because *trending* is a telemetry word
attached to a lab value. It costs a few seconds and some noise.

`amb-08` guesses at a prompt that names no subject, which the clarifying question
exists to catch.

The ordering — recall failures above precision failures above abstention
failures — is the same asymmetry the metrics are split along, applied to
deciding what to work on.

## Regenerating

`./run.ps1 compare` runs 102 cases × 3 attempts × 2 arms and writes
`evals/comparison_report.json`; update the tables above from it.
