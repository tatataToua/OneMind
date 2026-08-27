/**
 * What each guardrail span means, in the reader's language.
 *
 * A trace that says "Ungrounded identifier blocked" and nothing else answers
 * the wrong question. The reader is not asking what happened - the span name
 * already said that. They are asking *which check* this was, *what rule* it
 * enforces, and *why that rule fired on this call*. An audit read that cannot
 * answer the third question is a log, not an explanation.
 *
 * So each note has four parts, and the split is deliberate:
 *
 *   rule    invariant - the same sentence every time this guard fires, which
 *           is what makes it a rule rather than a verdict
 *   why     built from what *this* span recorded, so two firings of the same
 *           guard read differently
 *   effect  what the system did about it, because "blocked" and "flagged" are
 *           different promises and the reader must not have to guess which
 *   source  where the rule lives, so the claim is checkable rather than taken
 *           on trust
 *
 * Every sentence here is a restatement of a docstring in the module named by
 * `source`. Nothing is invented for display, for the same reason the timeline
 * synthesises no numbers: this view doubles as the audit read, and a UI that
 * paraphrases the enforcement into something the enforcement does not say is
 * worse than one that shows the raw span.
 */

import type { Span } from "./api";

export interface GuardrailNote {
  /** The check's own name - what to call it when someone asks "which one?" */
  guard: string;
  /** Module and function, so the rule can be read at the source. */
  source: string;
  /** The rule, invariant across firings. */
  rule: string;
  /** Why this particular span fired, from what it recorded. */
  why: string;
  /** What the system did as a result. */
  effect: string;
  /**
   * Blocking guards refuse a call before it reaches a data plane. Reporting
   * guards annotate a trace and change nothing. Applied guards are the two
   * ends of the PHI boundary, which run on every request and refuse nothing.
   *
   * The distinction is the one readers get wrong: an amber "Unverified values"
   * span next to a red "blocked" span looks like two refusals, and only one of
   * them stopped anything.
   */
  action: "blocked" | "flagged" | "applied";
}

const list = (value: unknown): string[] =>
  Array.isArray(value) ? value.map(String).filter(Boolean) : [];

const num = (value: unknown): number => (typeof value === "number" ? value : 0);

const str = (value: unknown): string => (typeof value === "string" ? value : "");

/** "a, b and c" - a list read aloud, since these appear inside sentences. */
function prose(items: string[]): string {
  if (items.length === 0) return "";
  if (items.length === 1) return items[0];
  return items.slice(0, -1).join(", ") + " and " + items[items.length - 1];
}

const plural = (n: number, one: string, many: string) => (n === 1 ? one : many);

/**
 * Argument names carrying a code rather than an identity.
 *
 * Mirrors `_is_coded_value` in `agents/base.py`. The two harms are different
 * enough to be worth different sentences: a fabricated identifier opens the
 * wrong person's chart, a fabricated code makes a confident statement about
 * the wrong procedure. Telling the reader "an identifier was invented" when a
 * CPT code was invented is the kind of small imprecision that makes an auditor
 * stop believing the rest of the panel.
 */
const CODED = new Set(["code", "icd10_code", "cpt_code", "denial_code"]);

function ungroundedWhy(tool: string, args: string[]): string {
  const named = args.length > 0 ? prose(args.map((a) => `\`${a}\``)) : "an identifier argument";
  const coded = args.length > 0 && args.every((a) => CODED.has(a));
  const call = tool ? `\`${tool}\`` : "a tool";

  const harm = coded
    ? "A made-up code still resolves. The answer then reports on the wrong procedure, in the " +
      "same clean format as every checked one."
    : "A made-up id still resolves. The specialist then answers about a real person the " +
      "question never named.";

  return (
    `The planner called ${call} with ${named}. That value came from neither place. It usually ` +
    `comes from the example in the tool's own parameter description, the closest thing to an ` +
    `identifier the model can see. ${harm}`
  );
}

function unscopedWhy(tool: string, args: string[]): string {
  const named = args.length > 0 ? prose(args.map((a) => `\`${a}\``)) : "its scoping argument";
  const call = tool ? `\`${tool}\`` : "a tool";
  return (
    `The request is about one person. ${call} was planned with no value for ${named}, nothing ` +
    `else on the call narrowed it, and no other specialist had found one. Running it would read ` +
    `every record in the store and then report "no data".`
  );
}

function unverifiedWhy(values: string[], count: number): string {
  const n = count || values.length;
  const shown = values.length > 0 ? ` ${prose(values.map((v) => `\`${v}\``))}.` : ".";
  return (
    `${n} ${plural(n, "value", "values")} in the answer ${plural(n, "appears", "appear")} in ` +
    `neither the tool results nor the question:${shown} The lookup was scoped correctly and ` +
    `returned the right record. The model then asserted something that record does not say.`
  );
}

function injectionWhy(fragments: string[], count: number): string {
  const n = count || fragments.length;
  return (
    `${n} instruction-shaped ${plural(n, "fragment", "fragments")} arrived inside a retrieved ` +
    `record. ${plural(n, "It", "They")} came out of a record system, so nobody saw ` +
    `${plural(n, "it", "them")} before the model did.`
  );
}

/**
 * The note for `span`, or null when the span is not one this map covers.
 *
 * Returning null rather than a generic note is deliberate: a new guardrail
 * should show its raw recorded detail and no explanation, not a confident
 * paragraph written for a different check.
 */
export function explainGuardrail(span: Span): GuardrailNote | null {
  if (span.kind !== "guardrail") return null;

  const tool = str(span.detail.tool);
  const args = list(span.detail.arguments);

  switch (span.name) {
    case "Ungrounded identifier blocked":
      return {
        guard: "Argument grounding",
        source: "agents/base.py · _is_grounded",
        rule:
          "An argument that picks whose record comes back, or which code gets looked up, has to " +
          "come from the request. Two forms count: a placeholder this session issued, or text " +
          "the request contains word for word.",
        why: ungroundedWhy(tool, args),
        effect:
          "The whole call was dropped. Removing the one argument would leave a lookup that " +
          "either errors on what is missing or widens to the entire store. The rejected value " +
          "is not shown here. It is unverified, and it might be a real identifier the model " +
          "happened to guess.",
        action: "blocked",
      };

    case "Unscoped lookup declined":
      return {
        guard: "Scope check",
        source: "agents/base.py · under_scoped",
        rule:
          "When the request names a person, the lookup has to carry an identifier. Leave it out " +
          "and the tool returns every record rather than an error.",
        why: unscopedWhy(tool, args),
        effect:
          "The call was declined and this specialist was marked for a second wave. If another " +
          "specialist finds the identifier, it runs again with it.",
        action: "blocked",
      };

    case "Unverified values":
      return {
        guard: "Answer grounding",
        source: "guardrails/grounding.py · ungrounded_values",
        rule:
          "Every code, identifier and date in the answer has to appear in the evidence or the " +
          "question. Plain numbers are exempt. Rounding 318.38 to “about $318” is the job, and " +
          "checking it would reject far more right answers than wrong ones.",
        why: unverifiedWhy(list(span.detail.values), num(span.detail.count)),
        effect:
          "Nothing was blocked. The answer ships with these values marked, so you can see which " +
          "claims it cannot vouch for. They pass through redaction on the way into this trace, " +
          "because PHI the inbound patterns missed looks exactly like a value absent from the " +
          "evidence.",
        action: "flagged",
      };

    case "Instruction-like text in retrieved data":
      return {
        guard: "Prompt-injection detector",
        source: "guardrails/injection.py · suspicious_spans",
        rule:
          "Retrieved records are data, never instructions. Each pattern needs an override verb " +
          "attached to an object meaning these instructions. That is what keeps clinical prose " +
          "like “follow the instructions on the label” from setting it off.",
        why: injectionWhy(list(span.detail.fragments), num(span.detail.count)),
        effect:
          "Nothing was blocked. The fence around the evidence is the defence, and it runs on " +
          "every request. This detector is the audit signal. Text no pattern here matches can " +
          "still talk a model round, so leaning on detection would rest the system on the " +
          "weaker of the two.",
        action: "flagged",
      };

    case "PHI redaction (inbound)": {
      const enabled = span.detail.enabled !== false;
      const n = num(span.detail.redacted_count);
      const kinds = list(span.detail.kinds);
      return {
        guard: "PHI boundary, inbound",
        source: "guardrails/phi.py · PHISession.redact",
        rule:
          "The model is untrusted. The data plane is not. Every identifier the patterns " +
          "recognise becomes a placeholder before the model sees anything, and that placeholder " +
          "means the same person for the whole conversation.",
        why: enabled
          ? n > 0
            ? `${n} ${plural(n, "identifier", "identifiers")} in this request ${plural(n, "was", "were")} ` +
              `replaced with a placeholder${kinds.length ? ` (${prose(kinds)})` : ""}. The router, ` +
              `the specialists and this trace see only tokens.`
            : "No identifier in this request matched a redaction pattern, so nothing was tokenised."
          : "Redaction is off for this run, so the request reached the model as typed. The " +
            "grounding checks lose their signal too. With no tokens, an argument counts as " +
            "grounded only when it appears word for word in the request.",
        effect:
          "Identifiers cross back at the tool boundary, where they become real lookup keys " +
          "again. The tools are the system of record, and hiding an id from the store it came " +
          "from protects nobody.",
        action: "applied",
      };
    }

    case "PHI re-hydration (outbound)": {
      const n = num(span.detail.tokens);
      return {
        guard: "PHI boundary, outbound",
        source: "guardrails/phi.py · PHISession.rehydrate",
        rule:
          "The model writes the answer over placeholders, and they are restored last. The trace " +
          "and the logs keep the redacted form. You get real names.",
        why:
          n > 0
            ? `${n} ${plural(n, "token was", "tokens were")} in scope for this request. Matching ` +
              `strips separators first, because models reformat tokens: \`PHI_MRN_1\` came back ` +
              `as \`PHI_MR_N_1\` in testing. A placeholder left unresolved in a clinical answer ` +
              `is worse than no answer.`
            : "No tokens were minted for this request, so re-hydration had nothing to restore.",
        effect:
          "Only the answer and the findings are restored. Every span above this one keeps its " +
          "placeholders, which is what makes the trace safe to keep.",
        action: "applied",
      };
    }

    default:
      return null;
  }
}
