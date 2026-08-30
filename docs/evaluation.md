# Evaluation: reading the results

The headline tables are in the [README](../README.md#tests-and-evaluation),
generated from `evals/comparison_report.json`. This document explains what those
numbers show and what they do not.

## The router versus a single agent

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

```
./run.ps1 compare
python evals/update_readme.py
```

`./run.ps1 compare` runs 102 cases × 3 attempts × 2 arms and writes
`evals/comparison_report.json`. `update_readme.py` projects that report into the
README between the `eval:begin` / `eval:end` markers; CI runs it with `--check`.
