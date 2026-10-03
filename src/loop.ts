import { estimateTokens, restartCompactor, type Compactor } from "./context.ts";
import { validate } from "./schema.ts";
import type {
  FinishReason, HarnessEvent, JsonSchema, Message, Model, ModelResponse, Part, RunResult, Tool, ToolCallPart, ToolResultPart,
} from "./types.ts";

export type RunOptions = {
  model: Model;
  /** The task. A string becomes the first user message. */
  input: string | Message[];
  system?: string;
  tools?: Tool[];
  /** If set, the final answer must be JSON matching this schema; the model gets bounded repair attempts. */
  output?: JsonSchema;
  maxTurns?: number;            // model calls, default 12
  maxTokens?: number;           // per model call, default 4096
  contextBudget?: number;       // estimated tokens before compaction, default 60_000
  compactor?: Compactor;
  toolTimeoutMs?: number;       // default 30_000
  modelRetries?: number;        // default 3, exponential backoff on retryable errors
  maxRepeatCalls?: number;      // identical tool call repeated more than this = loop, default 3
  maxRepairs?: number;          // structured-output repair attempts, default 2
  signal?: AbortSignal;
  onEvent?: (e: HarnessEvent) => void;
};

export class HarnessAbortError extends Error {
  constructor() { super("run aborted"); this.name = "HarnessAbortError"; }
}

export async function runAgent<T = unknown>(opts: RunOptions): Promise<RunResult<T>> {
  const tools = new Map((opts.tools ?? []).map((t) => [t.name, t]));
  const specs = (opts.tools ?? []).map(({ name, description, inputSchema }) => ({ name, description, inputSchema }));
  const maxTurns = opts.maxTurns ?? 12;
  const budget = opts.contextBudget ?? 60_000;
  const compactor = opts.compactor ?? restartCompactor();
  const task = typeof opts.input === "string" ? opts.input : firstText(opts.input);
  let messages: Message[] = typeof opts.input === "string" ? [{ role: "user", content: [{ type: "text", text: opts.input }] }] : [...opts.input];

  const trace: HarnessEvent[] = [];
  const emit = (e: HarnessEvent) => { trace.push(e); opts.onEvent?.(e); };
  const usage = { input: 0, output: 0, modelCalls: 0, toolCalls: 0 };
  const callCounts = new Map<string, number>();
  let repairs = 0;

  const finish = (reason: FinishReason, text: string, output?: T): RunResult<T> => {
    emit({ type: "final", reason });
    return { reason, text, output, messages, trace, usage };
  };

  for (let turn = 1; turn <= maxTurns; turn++) {
    throwIfAborted(opts.signal);

    const before = estimateTokens(messages, opts.system);
    if (before > budget && messages.length > 2) {
      messages = await compactor(messages, { task, signal: opts.signal });
      emit({ type: "compaction", beforeTokens: before, afterTokens: estimateTokens(messages, opts.system) });
    }

    emit({ type: "model_call", turn, messages: messages.length });
    const res = await callWithRetry(opts, () => opts.model.complete({
      system: opts.system, messages, tools: specs, maxTokens: opts.maxTokens ?? 4096, signal: opts.signal,
    }), emit);
    usage.modelCalls++;
    usage.input += res.usage?.input ?? 0;
    usage.output += res.usage?.output ?? 0;
    emit({ type: "model_response", turn, stop: res.stop, usage: res.usage });
    messages = [...messages, { role: "assistant", content: res.content }];

    if (res.stop === "refusal") return finish("refusal", textOf(res.content));
    const calls = res.content.filter((p): p is ToolCallPart => p.type === "tool_call");

    if (calls.length === 0) {
      if (res.stop === "max_tokens") return finish("max_tokens", textOf(res.content));
      const text = textOf(res.content);
      if (!opts.output) return finish("done", text);
      const parsed = parseJson(text);
      const errors = parsed.ok ? validate(parsed.value, opts.output, "output") : [parsed.error];
      if (parsed.ok && errors.length === 0) return finish("done", text, parsed.value as T);
      if (repairs >= (opts.maxRepairs ?? 2)) return finish("invalid_output", text);
      repairs++;
      emit({ type: "repair", attempt: repairs, error: errors.join("; ") });
      messages = [...messages, { role: "user", content: [{ type: "text", text:
        `Your answer must be only JSON matching the required schema. Problems: ${errors.join("; ")}. Reply again with corrected JSON only.` }] }];
      continue;
    }

    // Loop detection: the same call with the same arguments, again and again.
    for (const call of calls) {
      const key = `${call.name}:${stableStringify(call.input)}`;
      callCounts.set(key, (callCounts.get(key) ?? 0) + 1);
      if ((callCounts.get(key) ?? 0) > (opts.maxRepeatCalls ?? 3)) return finish("loop_detected", textOf(res.content));
    }

    // Parallel tool calls run concurrently; all results go back in ONE user message.
    const results = await Promise.all(calls.map((call) => runTool(call, tools, opts, emit)));
    usage.toolCalls += calls.length;
    messages = [...messages, { role: "user", content: results }];
  }
  return finish("max_turns", lastAssistantText(messages));
}

async function runTool(call: ToolCallPart, tools: Map<string, Tool>, opts: RunOptions, emit: (e: HarnessEvent) => void): Promise<ToolResultPart> {
  emit({ type: "tool_call", name: call.name, input: call.input });
  const started = Date.now();
  const done = (content: string, isError = false): ToolResultPart => {
    emit({ type: "tool_result", name: call.name, ok: !isError, ms: Date.now() - started, content: content.slice(0, 500) });
    return { type: "tool_result", id: call.id, content, isError: isError || undefined };
  };
  const tool = tools.get(call.name);
  if (!tool) return done(`Unknown tool "${call.name}". Available: ${[...tools.keys()].join(", ") || "none"}.`, true);
  const errors = validate(call.input, tool.inputSchema);
  if (errors.length) return done(`Invalid input, nothing was run. ${errors.join("; ")}. Fix the arguments and call again.`, true);

  const timeoutMs = tool.timeoutMs ?? opts.toolTimeoutMs ?? 30_000;
  const ctl = new AbortController();
  const onAbort = () => ctl.abort();
  opts.signal?.addEventListener("abort", onAbort, { once: true });
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const value = await Promise.race([
      Promise.resolve().then(() => tool.run(call.input, { signal: ctl.signal })),
      new Promise((_, reject) => { timer = setTimeout(() => { ctl.abort(); reject(new Error(`timed out after ${timeoutMs} ms`)); }, timeoutMs); }),
    ]);
    return done(typeof value === "string" ? value : JSON.stringify(value ?? null));
  } catch (err) {
    throwIfAborted(opts.signal);
    return done(`Tool failed: ${err instanceof Error ? err.message : String(err)}`, true);
  } finally {
    clearTimeout(timer);
    opts.signal?.removeEventListener("abort", onAbort);
  }
}

async function callWithRetry(opts: RunOptions, fn: () => Promise<ModelResponse>, emit: (e: HarnessEvent) => void): Promise<ModelResponse> {
  const retries = opts.modelRetries ?? 3;
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn();
    } catch (err) {
      throwIfAborted(opts.signal);
      if (attempt >= retries || !isRetryable(err)) throw err;
      const delayMs = Math.min(8000, 500 * 2 ** attempt) + Math.floor(Math.random() * 250);
      emit({ type: "retry", attempt: attempt + 1, delayMs, error: err instanceof Error ? err.message : String(err) });
      await sleep(delayMs, opts.signal);
    }
  }
}

export function isRetryable(err: unknown): boolean {
  const e = err as { status?: number; retryable?: boolean; code?: string };
  if (e?.retryable !== undefined) return e.retryable;
  if (typeof e?.status === "number") return e.status === 408 || e.status === 409 || e.status === 429 || e.status >= 500;
  return ["ECONNRESET", "ETIMEDOUT", "ECONNREFUSED", "UND_ERR_SOCKET"].includes(e?.code ?? "");
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => { clearTimeout(t); reject(new HarnessAbortError()); }, { once: true });
  });
}

function throwIfAborted(signal?: AbortSignal) {
  if (signal?.aborted) throw new HarnessAbortError();
}

function textOf(parts: Part[]): string {
  return parts.filter((p) => p.type === "text").map((p) => (p as { text: string }).text).join("").trim();
}

function firstText(messages: Message[]): string {
  return textOf(messages.find((m) => m.role === "user")?.content ?? []);
}

function lastAssistantText(messages: Message[]): string {
  for (let i = messages.length - 1; i >= 0; i--) if (messages[i].role === "assistant") return textOf(messages[i].content);
  return "";
}

function parseJson(text: string): { ok: true; value: unknown } | { ok: false; error: string } {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  const body = (fenced ? fenced[1] : text).trim();
  try { return { ok: true, value: JSON.parse(body) }; }
  catch (e) { return { ok: false, error: `output: not valid JSON (${(e as Error).message})` }; }
}

export function stableStringify(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(stableStringify).join(",")}]`;
  if (v && typeof v === "object") return `{${Object.keys(v).sort().map((k) => `${JSON.stringify(k)}:${stableStringify((v as any)[k])}`).join(",")}}`;
  return JSON.stringify(v ?? null);
}
