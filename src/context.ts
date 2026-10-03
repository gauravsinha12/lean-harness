import type { Message, Model, Part } from "./types.ts";

/** Rough token estimate (~4 chars per token). Swap in a real tokenizer per model if needed. */
export function estimateTokens(messages: Message[], system = ""): number {
  let chars = system.length;
  for (const m of messages) for (const p of m.content) chars += partChars(p);
  return Math.ceil(chars / 4);
}

function partChars(p: Part): number {
  switch (p.type) {
    case "text": return p.text.length;
    case "tool_call": return p.name.length + JSON.stringify(p.input ?? null).length;
    case "tool_result": return p.content.length;
    case "opaque": return JSON.stringify(p.block ?? null).length;
  }
}

export type Compactor = (messages: Message[], ctx: { task: string; signal?: AbortSignal }) => Promise<Message[]>;

/**
 * Compaction that never edits history in place. Some providers bind reasoning blocks to the exact
 * conversation that produced them, so rewriting old turns can invalidate them. Instead this starts
 * a fresh conversation: the original task plus a progress summary, then the latest turns verbatim.
 *
 * `summarize` can be a model call; the default is a deterministic digest of tool calls and results.
 */
export function restartCompactor(opts: { keepLastTurns?: number; summarize?: (history: Message[]) => Promise<string> } = {}): Compactor {
  const keep = opts.keepLastTurns ?? 2;
  return async (messages, { task }) => {
    const cut = safeCutIndex(messages, keep);
    const older = messages.slice(0, cut);
    const recent = messages.slice(cut);
    const summary = opts.summarize ? await opts.summarize(older) : digest(older);
    const head: Message = {
      role: "user",
      content: [{ type: "text", text: `${task}\n\n[Progress so far, summarized to save context]\n${summary}` }],
    };
    // The conversation must alternate and start with the user: merge if `recent` also starts with user.
    if (recent[0]?.role === "user") return [{ role: "user", content: [...head.content, ...recent[0].content] }, ...recent.slice(1)];
    return [head, ...recent];
  };
}

/** Index where the kept tail begins: always at an assistant turn, never splitting a tool call from its result. */
function safeCutIndex(messages: Message[], keepTurns: number): number {
  let seen = 0;
  for (let i = messages.length - 1; i > 0; i--) {
    if (messages[i].role === "assistant") {
      seen++;
      if (seen === keepTurns) return i;
    }
  }
  return 1;
}

function digest(messages: Message[]): string {
  const lines: string[] = [];
  const names = new Map<string, string>();
  for (const m of messages) {
    for (const p of m.content) {
      if (p.type === "tool_call") names.set(p.id, `${p.name}(${JSON.stringify(p.input).slice(0, 120)})`);
      if (p.type === "tool_result") {
        const call = names.get(p.id) ?? "tool";
        lines.push(`- ${call} → ${p.isError ? "ERROR " : ""}${p.content.replace(/\s+/g, " ").slice(0, 200)}`);
      }
      if (p.type === "text" && m.role === "assistant" && p.text.trim()) lines.push(`- note: ${p.text.replace(/\s+/g, " ").slice(0, 200)}`);
    }
  }
  return lines.join("\n") || "- (no tool activity yet)";
}

/** A summarizer that asks a (cheap) model for the progress summary. */
export function modelSummarizer(model: Model, maxTokens = 800): (history: Message[]) => Promise<string> {
  return async (history) => {
    const res = await model.complete({
      messages: [{ role: "user", content: [{ type: "text", text: "Summarize the progress in this agent transcript as terse bullet points: facts found, tool results that matter, open questions. No preamble.\n\n" + digest(history) }] }],
      tools: [],
      maxTokens,
    });
    return res.content.filter((p) => p.type === "text").map((p) => (p as { text: string }).text).join("\n");
  };
}
