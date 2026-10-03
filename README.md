# lean-harness

A small, dependency-free TypeScript agent harness: tool calling, the agent loop, context
compaction, structured outputs, failure handling, and record/replay evals. Use it with Claude or
any OpenAI-compatible local model server (llama.cpp, Ollama, vLLM).

```ts
import { runAgent, openAICompatibleModel } from "lean-harness";

const result = await runAgent({
  model: openAICompatibleModel({ baseUrl: "http://127.0.0.1:8080/v1", model: "qwen3-8b" }),
  input: "What's the weather in Mumbai?",
  tools: [{
    name: "get_weather",
    description: "Current temperature for a city.",
    inputSchema: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
    run: async ({ city }) => fetchWeather(city),
  }],
});
console.log(result.reason, result.text);
```

## What it handles

| Concern | Behaviour |
|---|---|
| **Tool input validation** | Inputs are checked against the tool's JSON Schema *before* running. Errors go back to the model in words it can act on (`input.city: required`), and nothing runs. Small local models get arguments wrong often, so this matters. |
| **Parallel tool calls** | Run concurrently; all results go back in a single message. |
| **Timeouts** | Per-tool timeout aborts the tool's `signal` and returns an error result. The run continues. |
| **Model errors** | 408/409/429/5xx and connection resets are retried with exponential backoff and jitter. 4xx errors fail fast. |
| **Loops** | The same tool called with identical arguments more than N times ends the run with `loop_detected`. |
| **Structured output** | Pass `output: schema`. The final answer is parsed (fenced or bare JSON) and validated, with bounded repair turns. |
| **Context budget** | Over budget, the conversation *restarts* as task + progress summary + the latest turns verbatim. It never edits old turns in place, because some providers bind reasoning blocks to the exact conversation that produced them. |
| **Cancellation** | One `AbortSignal` stops model calls, backoff sleeps and running tools. |
| **Provider blocks** | Unknown blocks (e.g. Claude thinking blocks) are carried as `opaque` parts and sent back unchanged. |
| **Observability** | `onEvent` streams model calls, retries, tool calls with latency, compactions and repairs; the full `trace` comes back with the result. |

## Testing non-deterministic agents

```ts
import { cassetteModel, runEvals, runAgent } from "lean-harness";

// Record once against a real model, commit the cassette, replay in CI with no network:
const model = cassetteModel({ inner: liveModel, path: "test/cassettes/weather.json", mode: "auto" });

const report = await runEvals([
  { name: "uses the tool", input: "Weather in Mumbai?", mustCall: ["get_weather"], expectReason: "done" },
  { name: "no tool for small talk", input: "hi", mustNotCall: ["get_weather"], maxToolCalls: 0 },
], (input) => runAgent({ model, input, tools }));
console.log(`${report.passed}/${report.total}`);
```

Evals grade the **trace**, not just the text: which tools were called, how many, how the run
ended, plus any custom check. Against a live model, run each case N times to get pass rates.
Against a cassette, the same suite is a deterministic regression test, and a changed prompt or
tool schema shows up as a loud "cassette miss".

## Adapters

- `openAICompatibleModel({ baseUrl, model, apiKey? })`: llama.cpp `llama-server`, Ollama, vLLM, LM Studio.
- `anthropicModel({ client: new Anthropic(), model?, effort? })`: Claude through the official SDK (optional peer dependency). It defaults to `claude-opus-5-5` with server-side refusal fallback on.
- `scriptedModel([...])` and `cassetteModel(...)` for tests.

An adapter is one function, `complete(request) → { content, stop, usage }`, so adding a provider
or an in-process runtime is about 50 lines.

## Run

Node 23.6+ runs the TypeScript sources directly (type stripping). No build step.

```bash
npm test                                           # 12 tests
node examples/local-files.ts ./docs "What does the deploy script do?"
```

MIT licensed.
