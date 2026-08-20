export type SpanKind =
  | "guardrail"
  | "route"
  | "agent"
  | "tool"
  | "synthesize";

export interface Span {
  span_id: string;
  parent_id: string | null;
  kind: SpanKind;
  name: string;
  status: "running" | "ok" | "error";
  duration_ms: number | null;
  offset_ms: number;
  detail: Record<string, unknown>;
}

export interface Outcome {
  request_id: string;
  answer: string;
  agents: string[];
  citations: string[];
  clarifying_question: string;
  is_actionable: boolean;
  rationale: string;
  redacted_request: string;
  phi_redactions: number;
  trace: { request_id: string; spans: Span[] };
}

export interface AgentInfo {
  key: string;
  display_name: string;
  data_plane: string;
  description: string;
  tools: string[];
}

export interface ExampleInfo {
  label: string;
  prompt: string;
  expect: string;
}

export interface StreamHandlers {
  onSpanStart(span: Span): void;
  onSpanEnd(span: Span): void;
  onToken(text: string): void;
  onDone(outcome: Outcome): void;
  onError(message: string): void;
}

export async function fetchAgents(): Promise<AgentInfo[]> {
  const res = await fetch("/api/agents");
  return (await res.json()).agents;
}

export async function fetchExamples(): Promise<ExampleInfo[]> {
  const res = await fetch("/api/examples");
  return (await res.json()).examples;
}

/**
 * Reads the SSE stream by hand rather than with EventSource, because
 * EventSource cannot issue a POST and the request body carries the prompt.
 *
 * Events arrive as `event: <name>\ndata: <json>\n\n`. A chunk boundary can land
 * mid-event, so the tail of the buffer is held back until its terminator shows
 * up rather than being parsed optimistically.
 */
export async function streamChat(
  message: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
    signal,
  });

  if (!res.ok || !res.body) {
    handlers.onError(`Request failed: ${res.status} ${res.statusText}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      const match = block.match(/^event: (\w+)\ndata: ([\s\S]*)$/);
      if (!match) continue;

      const [, event, raw] = match;
      let data: unknown;
      try {
        data = JSON.parse(raw);
      } catch {
        continue;
      }

      switch (event) {
        case "span_start":
          handlers.onSpanStart(data as Span);
          break;
        case "span_end":
          handlers.onSpanEnd(data as Span);
          break;
        case "token":
          handlers.onToken((data as { text: string }).text);
          break;
        case "done":
          handlers.onDone(data as Outcome);
          break;
        case "error":
          handlers.onError((data as { message: string }).message);
          break;
      }
    }
  }
}
