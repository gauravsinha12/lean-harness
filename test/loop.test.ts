import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { cassetteModel, restartCompactor, runAgent, runEvals, scriptedModel, validate, type Tool } from "../src/index.ts";

const text = (t: string) => ({ content: [{ type: "text" as const, text: t }], stop: "end" as const });
const call = (name: string, input: unknown, id = `c_${name}_${Math.random().toString(36).slice(2, 7)}`) =>
  ({ content: [{ type: "tool_call" as const, id, name, input }], stop: "tool_use" as const });

const weather: Tool = {
  name: "get_weather",
  description: "Current temperature for a city.",
  inputSchema: { type: "object", properties: { city: { type: "string" } }, required: ["city"], additionalProperties: false },
  run: ({ city }) => ({ city, tempC: 31 }),
};

test("runs a tool and returns the final answer", async () => {
  const model = scriptedModel([call("get_weather", { city: "Mumbai" }), text("It's 31°C in Mumbai.")]);
  const r = await runAgent({ model, input: "Weather in Mumbai?", tools: [weather] });
  assert.equal(r.reason, "done");
  assert.equal(r.text, "It's 31°C in Mumbai.");
  assert.equal(r.usage.toolCalls, 1);
  const toolResult = model.calls[1].messages.at(-1)!.content[0] as any;
  assert.equal(toolResult.type, "tool_result");
  assert.match(toolResult.content, /"tempC":31/);
});

test("invalid tool input is reported back, not executed, and the model can correct it", async () => {
  let ran = 0;
  const tool = { ...weather, run: (i: any) => { ran++; return weather.run(i, { signal: new AbortController().signal }); } };
  const model = scriptedModel([call("get_weather", { town: "Pune" }), call("get_weather", { city: "Pune" }), text("31°C")]);
  const r = await runAgent({ model, input: "Pune?", tools: [tool] });
  assert.equal(r.reason, "done");
  assert.equal(ran, 1);
  const firstResult = model.calls[1].messages.at(-1)!.content[0] as any;
  assert.equal(firstResult.isError, true);
  assert.match(firstResult.content, /input\.city: required/);
  assert.match(firstResult.content, /input\.town: unknown property/);
});

test("parallel tool calls run concurrently and return in one message", async () => {
  const slow: Tool = { ...weather, run: async ({ city }) => { await new Promise((r) => setTimeout(r, 100)); return city; } };
  const model = scriptedModel([
    { content: [{ type: "tool_call", id: "a", name: "get_weather", input: { city: "A" } }, { type: "tool_call", id: "b", name: "get_weather", input: { city: "B" } }], stop: "tool_use" },
    text("done"),
  ]);
  const t0 = Date.now();
  await runAgent({ model, input: "both", tools: [slow] });
  assert.ok(Date.now() - t0 < 190, "tools should overlap");
  const results = model.calls[1].messages.at(-1)!.content;
  assert.deepEqual(results.map((p: any) => p.id), ["a", "b"]);
});

test("tool timeout becomes an error result and the run continues", async () => {
  const hang: Tool = { ...weather, timeoutMs: 50, run: () => new Promise(() => {}) };
  const model = scriptedModel([call("get_weather", { city: "X" }), text("Service is down, sorry.")]);
  const r = await runAgent({ model, input: "X?", tools: [hang] });
  assert.equal(r.reason, "done");
  const result = model.calls[1].messages.at(-1)!.content[0] as any;
  assert.match(result.content, /timed out after 50 ms/);
});

test("retries retryable model errors, gives up on non-retryable ones", async () => {
  const flaky = scriptedModel([Object.assign(new Error("overloaded"), { status: 529 }), text("ok")]);
  const r = await runAgent({ model: flaky, input: "hi" });
  assert.equal(r.text, "ok");
  assert.equal(r.trace.filter((e) => e.type === "retry").length, 1);

  const broken = scriptedModel([Object.assign(new Error("bad request"), { status: 400 })]);
  await assert.rejects(runAgent({ model: broken, input: "hi" }), /bad request/);
});

test("stops on a loop of identical calls", async () => {
  const model = scriptedModel(Array.from({ length: 10 }, () => call("get_weather", { city: "Same" })));
  const r = await runAgent({ model, input: "loop", tools: [weather], maxRepeatCalls: 3 });
  assert.equal(r.reason, "loop_detected");
  assert.equal(r.usage.toolCalls, 3);
});

test("structured output is validated and repaired", async () => {
  const model = scriptedModel([text("Sure! The answer is 31."), text('```json\n{"city": "Mumbai", "tempC": "31"}\n```'), text('{"city": "Mumbai", "tempC": 31}')]);
  const r = await runAgent<{ city: string; tempC: number }>({
    model, input: "json please",
    output: { type: "object", properties: { city: { type: "string" }, tempC: { type: "number" } }, required: ["city", "tempC"] },
  });
  assert.equal(r.reason, "done");
  assert.deepEqual(r.output, { city: "Mumbai", tempC: 31 });
  assert.equal(r.trace.filter((e) => e.type === "repair").length, 2);
});

test("compaction restarts the conversation instead of editing history", async () => {
  const big = "x".repeat(4000);
  const tool: Tool = { ...weather, run: () => big };
  const model = scriptedModel([call("get_weather", { city: "1" }), call("get_weather", { city: "2" }), call("get_weather", { city: "3" }), text("summary done")]);
  const r = await runAgent({ model, input: "Collect three readings.", tools: [tool], contextBudget: 1500, compactor: restartCompactor({ keepLastTurns: 1 }) });
  assert.equal(r.reason, "done");
  assert.ok(r.trace.some((e) => e.type === "compaction"));
  const compacted = model.calls.at(-1)!.messages;
  assert.equal(compacted[0].role, "user");
  assert.match((compacted[0].content[0] as any).text, /Collect three readings\.[\s\S]*Progress so far/);
  // The kept tail is verbatim: the last assistant turn and its tool result are unchanged.
  assert.equal(compacted.at(-2)!.role, "assistant");
});

test("abort signal stops the run", async () => {
  const ctl = new AbortController();
  const tool: Tool = { ...weather, run: () => { ctl.abort(); return "x"; } };
  const model = scriptedModel([call("get_weather", { city: "A" }), text("never")]);
  await assert.rejects(runAgent({ model, input: "a", tools: [tool], signal: ctl.signal }), /aborted/);
});

test("cassette replays recorded responses deterministically and fails loudly on a miss", async () => {
  const path = join(mkdtempSync(join(tmpdir(), "harness-")), "cassette.json");
  const live = scriptedModel([call("get_weather", { city: "Goa" }, "c1"), text("29°C in Goa")]);
  const recorded = await runAgent({ model: cassetteModel({ inner: live, path, mode: "record" }), input: "Goa?", tools: [weather] });
  const replayed = await runAgent({ model: cassetteModel({ path, mode: "replay" }), input: "Goa?", tools: [weather] });
  assert.equal(replayed.text, recorded.text);
  await assert.rejects(runAgent({ model: cassetteModel({ path, mode: "replay" }), input: "Delhi?", tools: [weather] }), /cassette miss/);
});

test("evals grade the trace", async () => {
  const report = await runEvals(
    [
      { name: "uses the tool", input: "Mumbai", mustCall: ["get_weather"], expectReason: "done", check: (r) => r.text.includes("31") || "no temperature in answer" },
      { name: "answers without tools", input: "hi", mustNotCall: ["get_weather"] },
    ],
    (input) => runAgent({
      model: scriptedModel(input === "hi" ? [text("hello")] : [call("get_weather", { city: input }), text("31°C")]),
      input, tools: [weather],
    }),
  );
  assert.equal(report.passed, 2);
});

test("validator messages are model-readable", () => {
  assert.deepEqual(validate({ n: "5" }, { type: "object", properties: { n: { type: "integer", minimum: 1 } } }), ["input.n: expected integer, got string"]);
  assert.deepEqual(validate({ mode: "x" }, { type: "object", properties: { mode: { enum: ["a", "b"] } } }), ['input.mode: must be one of "a", "b"']);
});
