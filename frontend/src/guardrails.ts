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
    ? "A code the request never mentioned does not fail - it resolves, and the answer reports a " +
      "finding about the wrong procedure in the canonical form that makes it look checked."
    : "An identifier the request never mentioned does not fail - it resolves, and the specialist " +
      "answers confidently with a real person's record in reply to a question that never named them.";

  return (
    `The planner called ${call} with ${named}, whose value was neither a placeholder this ` +
    `session issued nor anything the request contains. The usual origin is the example value in ` +
    `the tool's own parameter description, which is the nearest identifier-shaped string the ` +
    `model can see. ${harm}`
  );
}

function unscopedWhy(tool: string, args: string[]): string {
  const named = args.length > 0 ? prose(args.map((a) => `\`${a}\``)) : "its scoping argument";
  const call = tool ? `\`${tool}\`` : "a tool";
  return (
    `The request is plainly about one person, but ${call} was planned with no value for ${named}, ` +
    `nothing else on the call scoped it, and no sibling specialist had established one. Running it ` +
    `anyway would not error - it would read every record in the store and then report "no data" ` +
    `perfectly confidently.`
  );
}

function unverifiedWhy(values: string[], count: number): string {
  const n = count || values.length;
  const shown = values.length > 0 ? ` ${prose(values.map((v) => `\`${v}\``))}.` : ".";
  return (
    `${n} ${plural(n, "value", "values")} in the written answer ${plural(n, "appears", "appear")} ` +
    `in neither the tool results nor the question:${shown} This is the independent failure - the ` +
    `lookup was correctly scoped and returned the right record, and the model then asserted ` +
    `something that record does not say.`
  );
}

function injectionWhy(fragments: string[], count: number): string {
  const n = count || fragments.length;
  return (
    `${n} instruction-shaped ${plural(n, "fragment", "fragments")} arrived inside a retrieved ` +
    `record. It came from a record system, not from the user, which means nobody read it before ` +
    `the model did - the same way a hostile string reaches a database concatenated into something ` +
    `parsed as a mix of instruction and data.`
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
          "An argument that selects whose record is returned, or which code is looked up, must " +
          "trace back to the request: either a redaction placeholder this session issued, or a " +
          "literal the request contains verbatim. There is no third legitimate origin.",
        why: ungroundedWhy(tool, args),
        effect:
          "The whole call was dropped, not just the offending argument - a lookup stripped of " +
          "what scopes it either fails on a missing argument or widens into an unscoped query. " +
          "The rejected value is deliberately not recorded here: it is unverified, and may be a " +
          "real identifier the model guessed correctly.",
        action: "blocked",
      };

    case "Unscoped lookup declined":
      return {
        guard: "Scope check",
        source: "agents/base.py · under_scoped",
        rule:
          "When the request names a particular person, a lookup must carry an identifier. A " +
          "missing scoping argument is not the harmless case it looks like: the tool does not " +
          "reject it, it returns everything.",
        why: unscopedWhy(tool, args),
        effect:
          "The call was declined rather than widened, and this specialist was marked for a " +
          "second wave - if another specialist resolves the identifier, it runs again with it.",
        action: "blocked",
      };

    case "Unverified values":
      return {
        guard: "Answer grounding",
        source: "guardrails/grounding.py · ungrounded_values",
        rule:
          "Every code, identifier and date the answer asserts must appear in the evidence or in " +
          "the question. Bare numbers are exempt by design - rounding 318.38 to “about $318” " +
          "is what a language model is for, and checking it would refuse correct answers far more " +
          "often than it would catch a wrong one.",
        why: unverifiedWhy(list(span.detail.values), num(span.detail.count)),
        effect:
          "Nothing was blocked. The answer ships with these values marked, because a system that " +
          "says which of its claims it cannot vouch for is more useful than one that silently " +
          "drops them. The values pass through redaction on the way into this trace: “absent " +
          "from the evidence” is precisely what PHI the inbound patterns missed looks like.",
        action: "flagged",
      };

    case "Instruction-like text in retrieved data":
      return {
        guard: "Prompt-injection detector",
        source: "guardrails/injection.py · suspicious_spans",
        rule:
          "Retrieved records are data and never instructions. Each pattern requires an override " +
          "verb bound to an object meaning these instructions, which is what keeps clinical prose " +
          "– “follow the instructions on the label” – from firing it.",
        why: injectionWhy(list(span.detail.fragments), num(span.detail.count)),
        effect:
          "Nothing was blocked, by design. The fence around the evidence is the defence and runs " +
          "unconditionally; this is the audit signal. A model can be talked into things by text " +
          "no pattern here matches, so treating detection as the barrier would mean trusting the " +
          "weaker of the two mechanisms.",
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
          "The model is untrusted; the data plane is not. Every identifier the patterns recognise " +
          "becomes a stable placeholder before the model sees anything, and the placeholder means " +
          "the same person for the whole conversation.",
        why: enabled
          ? n > 0
            ? `${n} ${plural(n, "identifier", "identifiers")} in this request ${plural(n, "was", "were")} ` +
              `replaced with a placeholder${kinds.length ? ` (${prose(kinds)})` : ""}. Everything ` +
              `downstream – the router, the specialists, this trace – sees only tokens.`
            : "No identifier in this request matched a redaction pattern, so nothing was tokenised."
          : "Redaction is switched off for this run, so the request reached the model as typed. " +
            "The grounding checks lose their signal with it: with no tokens, an argument can only " +
            "be grounded by appearing verbatim in the request.",
        effect:
          "Identifiers cross back only at the tool boundary, where they are rehydrated into real " +
          "lookup keys - the tools are the system of record, and hiding an id from the store it " +
          "came from protects nothing.",
        action: "applied",
      };
    }

    case "PHI re-hydration (outbound)": {
      const n = num(span.detail.tokens);
      return {
        guard: "PHI boundary, outbound",
        source: "guardrails/phi.py · PHISession.rehydrate",
        rule:
          "The answer is written entirely over placeholders and restored last, so the trace and " +
          "the logs keep the redacted form while the reader gets real names.",
        why:
          n > 0
            ? `${n} ${plural(n, "token was", "tokens were")} in scope for this request. Matching is ` +
              `done on a separator-stripped form, because models reformat tokens - \`PHI_MRN_1\` ` +
              `came back as \`PHI_MR_N_1\` in testing, and an unresolved placeholder in a clinical ` +
              `answer is worse than no answer.`
            : "No tokens were minted for this request, so re-hydration had nothing to restore.",
        effect:
          "Only the user-facing answer and the findings are restored. Every span above this one " +
          "keeps its placeholders, which is what makes the trace safe to retain.",
        action: "applied",
      };
    }

    default:
      return null;
  }
}
