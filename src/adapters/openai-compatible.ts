import type { Message, Model, ModelRequest, ModelResponse, Part, StopReason } from "../types.ts";

/**
 * Any server that speaks the OpenAI chat-completions format: llama.cpp `llama-server`, Ollama,
 * vLLM, LM Studio. This is the path for local models.
 *
 *   const model = openAICompatibleModel({ baseUrl: "http://127.0.0.1:8080/v1", model: "qwen3-8b" });
 */
export function openAICompatibleModel(opts: { baseUrl: string; model: string; apiKey?: string; fetch?: typeof fetch }): Model {
  const doFetch = opts.fetch ?? fetch;
  return {
    name: opts.model,
    async complete(req: ModelRequest): Promise<ModelResponse> {
      const body = {
        model: opts.model,
        max_tokens: req.maxTokens,
        messages: [...(req.system ? [{ role: "system", content: req.system }] : []), ...req.messages.flatMap(toOpenAI)],
        ...(req.tools.length ? {
          tools: req.tools.map((t) => ({ type: "function", function: { name: t.name, description: t.description, parameters: t.inputSchema } })),
        } : {}),
      };
      const res = await doFetch(`${opts.baseUrl.replace(/\/$/, "")}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json", ...(opts.apiKey ? { authorization: `Bearer ${opts.apiKey}` } : {}) },
        body: JSON.stringify(body),
        signal: req.signal,
      });
      if (!res.ok) {
        const err = new Error(`model server returned ${res.status}: ${(await res.text()).slice(0, 300)}`) as Error & { status: number };
        err.status = res.status;
        throw err;
      }
      const data = await res.json();
      const choice = data.choices?.[0] ?? {};
      const msg = choice.message ?? {};
      const content: Part[] = [];
      if (msg.content) content.push({ type: "text", text: String(msg.content) });
      for (const tc of msg.tool_calls ?? []) {
        content.push({ type: "tool_call", id: tc.id ?? `call_${content.length}`, name: tc.function?.name, input: safeJson(tc.function?.arguments) });
      }
      return { content, stop: mapStop(choice.finish_reason, content), usage: { input: data.usage?.prompt_tokens ?? 0, output: data.usage?.completion_tokens ?? 0 } };
    },
  };
}

function toOpenAI(m: Message): any[] {
  if (m.role === "assistant") {
    const text = m.content.filter((p) => p.type === "text").map((p: any) => p.text).join("");
    const calls = m.content.filter((p) => p.type === "tool_call") as Array<Extract<Part, { type: "tool_call" }>>;
    return [{ role: "assistant", content: text || null, ...(calls.length ? {
      tool_calls: calls.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: JSON.stringify(c.input) } })),
    } : {}) }];
  }
  const out: any[] = [];
  const text = m.content.filter((p) => p.type === "text").map((p: any) => p.text).join("");
  for (const p of m.content) if (p.type === "tool_result") out.push({ role: "tool", tool_call_id: p.id, content: p.isError ? `ERROR: ${p.content}` : p.content });
  if (text) out.push({ role: "user", content: text });
  return out;
}

function safeJson(s: unknown): unknown {
  if (typeof s !== "string") return s ?? {};
  try { return JSON.parse(s); } catch { return { _unparsed: s }; }  // validation then reports it back to the model
}

function mapStop(reason: string, content: Part[]): StopReason {
  if (content.some((p) => p.type === "tool_call")) return "tool_use";
  if (reason === "length") return "max_tokens";
  if (reason === "content_filter") return "refusal";
  return "end";
}
