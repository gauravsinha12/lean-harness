// Provider-neutral message format. Adapters translate to and from each model API.

export type TextPart = { type: "text"; text: string };
export type ToolCallPart = { type: "tool_call"; id: string; name: string; input: unknown };
export type ToolResultPart = { type: "tool_result"; id: string; content: string; isError?: boolean };
/** A provider block the harness must hand back untouched (e.g. Anthropic thinking blocks). */
export type OpaquePart = { type: "opaque"; provider: string; block: unknown };
export type Part = TextPart | ToolCallPart | ToolResultPart | OpaquePart;

export type Message = { role: "user" | "assistant"; content: Part[] };

export type JsonSchema = {
  type?: "object" | "array" | "string" | "number" | "integer" | "boolean" | "null" | Array<string>;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  items?: JsonSchema;
  enum?: unknown[];
  additionalProperties?: boolean;
  description?: string;
  minimum?: number;
  maximum?: number;
};

export type ToolSpec = { name: string; description: string; inputSchema: JsonSchema };

export type Tool = ToolSpec & {
  /** Runs the tool. Throw to report an error back to the model; the loop keeps going. */
  run: (input: any, ctx: { signal: AbortSignal }) => unknown | Promise<unknown>;
  /** Per-tool timeout in ms (defaults to the run's toolTimeoutMs). */
  timeoutMs?: number;
};

export type StopReason = "end" | "tool_use" | "max_tokens" | "refusal";

export type ModelRequest = {
  system?: string;
  messages: Message[];
  tools: ToolSpec[];
  maxTokens: number;
  signal?: AbortSignal;
};

export type ModelResponse = {
  content: Part[];
  stop: StopReason;
  usage?: { input: number; output: number };
};

export interface Model {
  name: string;
  complete(req: ModelRequest): Promise<ModelResponse>;
}

export type HarnessEvent =
  | { type: "model_call"; turn: number; messages: number }
  | { type: "model_response"; turn: number; stop: StopReason; usage?: ModelResponse["usage"] }
  | { type: "retry"; attempt: number; delayMs: number; error: string }
  | { type: "tool_call"; name: string; input: unknown }
  | { type: "tool_result"; name: string; ok: boolean; ms: number; content: string }
  | { type: "compaction"; beforeTokens: number; afterTokens: number }
  | { type: "repair"; attempt: number; error: string }
  | { type: "final"; reason: FinishReason };

export type FinishReason = "done" | "max_turns" | "loop_detected" | "refusal" | "max_tokens" | "invalid_output";

export type RunResult<T = unknown> = {
  reason: FinishReason;
  text: string;
  /** Parsed and validated final answer when `output` schema was given. */
  output?: T;
  messages: Message[];
  trace: HarnessEvent[];
  usage: { input: number; output: number; modelCalls: number; toolCalls: number };
};
