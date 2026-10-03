import type { RunResult } from "./types.ts";

export type EvalCase<T = unknown> = {
  name: string;
  input: string;
  /** Tools that must be called (in any order). */
  mustCall?: string[];
  /** Tools that must not be called. */
  mustNotCall?: string[];
  maxToolCalls?: number;
  expectReason?: RunResult["reason"];
  /** Return true to pass, or a string describing the failure. */
  check?: (result: RunResult<T>) => boolean | string;
};

export type EvalOutcome = { name: string; pass: boolean; failures: string[]; toolCalls: string[]; reason: string };

/**
 * Runs each case through `run` and grades the trace, not just the final text: which tools were
 * called, how many, how the run ended, and any custom check. Pair with `cassetteModel` for
 * deterministic CI runs, or point at a live model to measure pass rates over N samples.
 */
export async function runEvals<T>(cases: EvalCase<T>[], run: (input: string) => Promise<RunResult<T>>) {
  const outcomes: EvalOutcome[] = [];
  for (const c of cases) {
    const failures: string[] = [];
    let toolCalls: string[] = [];
    let reason = "error";
    try {
      const result = await run(c.input);
      reason = result.reason;
      toolCalls = result.trace.filter((e) => e.type === "tool_call").map((e) => (e as { name: string }).name);
      for (const t of c.mustCall ?? []) if (!toolCalls.includes(t)) failures.push(`expected a call to ${t}`);
      for (const t of c.mustNotCall ?? []) if (toolCalls.includes(t)) failures.push(`must not call ${t}`);
      if (c.maxToolCalls !== undefined && toolCalls.length > c.maxToolCalls) failures.push(`${toolCalls.length} tool calls > ${c.maxToolCalls}`);
      if (c.expectReason && result.reason !== c.expectReason) failures.push(`ended with ${result.reason}, expected ${c.expectReason}`);
      if (c.check) {
        const verdict = c.check(result);
        if (verdict !== true) failures.push(typeof verdict === "string" ? verdict : "custom check failed");
      }
    } catch (err) {
      failures.push(`threw: ${err instanceof Error ? err.message : String(err)}`);
    }
    outcomes.push({ name: c.name, pass: failures.length === 0, failures, toolCalls, reason });
  }
  const passed = outcomes.filter((o) => o.pass).length;
  return { passed, total: outcomes.length, passRate: outcomes.length ? passed / outcomes.length : 0, outcomes };
}
