import type { Message, Model, ModelRequest, ModelResponse, Part, StopReason } from "../types.ts";

/**
 * Claude via the official SDK (`npm i @anthropic-ai/sdk`), passed in so this package has no
 * runtime dependencies:
 *
 *   import Anthropic from "@anthropic-ai/sdk";
 *   const model = anthropicModel({ client: new Anthropic() });
 *
 * Thinking blocks come back as `opaque` parts and are sent back unchanged, because current
 * models bind them to the conversation that produced them. Server-side refusal fallback is on by
 * default; pass `fallbacks: false` to turn it off.
 */
export function anthropicModel(opts: {
  client: any;
  model?: string;
  effort?: "low" | "medium" | "high" | "xhigh" | "max";
  fallbacks?: boolean;
}): Model {
  const model = opts.model ?? "claude-opus-5-5";
  const useFallbacks = opts.fallbacks !== false;
  return {
    name: model,
    async complete(req: ModelRequest): Promise<ModelResponse> {
      const params: Record<string, unknown> = {
        model,
        max_tokens: req.maxTokens,
        messages: req.messages.map(toAnthropic),
        ...(req.system ? { system: req.system } : {}),
        ...(req.tools.length ? { tools: req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.inputSchema })) } : {}),
        ...(opts.effort ? { output_config: { effort: opts.effort } } : {}),
      };
      const res = useFallbacks
        ? await opts.client.beta.messages.create({ ...params, betas: ["server-side-fallback-2026-07-01"], fallbacks: "default" }, { signal: req.signal })
        : await opts.client.messages.create(params, { signal: req.signal });
      return {
        content: (res.content ?? []).map(fromAnthropic),
        stop: mapStop(res.stop_reason),
        usage: { input: res.usage?.input_tokens ?? 0, output: res.usage?.output_tokens ?? 0 },
      };
    },
  };
}

function toAnthropic(m: Message) {
  return {
    role: m.role,
    content: m.content.map((p) => {
      switch (p.type) {
        case "text": return { type: "text", text: p.text };
        case "tool_call": return { type: "tool_use", id: p.id, name: p.name, input: p.input };
        case "tool_result": return { type: "tool_result", tool_use_id: p.id, content: p.content, ...(p.isError ? { is_error: true } : {}) };
        case "opaque": return p.block;
      }
    }),
  };
}

function fromAnthropic(block: any): Part {
  if (block.type === "text") return { type: "text", text: block.text };
  if (block.type === "tool_use") return { type: "tool_call", id: block.id, name: block.name, input: block.input };
  return { type: "opaque", provider: "anthropic", block };
}

function mapStop(reason: string): StopReason {
  if (reason === "tool_use") return "tool_use";
  if (reason === "max_tokens") return "max_tokens";
  if (reason === "refusal") return "refusal";
  return "end";
}
