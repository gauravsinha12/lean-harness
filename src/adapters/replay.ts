import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { stableStringify } from "../loop.ts";
import type { Model, ModelRequest, ModelResponse } from "../types.ts";

/**
 * Record/replay for testing non-deterministic agents.
 *
 * Record once against a real model, commit the cassette, and every later run replays the exact
 * same model responses, so agent logic (tool routing, validation, retries, compaction) can be
 * regression-tested in CI without network calls or flaky outputs. A request the cassette has
 * never seen fails loudly with mode "replay", which is how prompt or tool changes get noticed.
 */
export function cassetteModel(opts: { inner?: Model; path: string; mode: "record" | "replay" | "auto" }): Model {
  const store: Record<string, ModelResponse> = existsSync(opts.path) ? JSON.parse(readFileSync(opts.path, "utf8")) : {};
  return {
    name: `cassette(${opts.inner?.name ?? "replay"})`,
    async complete(req: ModelRequest): Promise<ModelResponse> {
      const key = requestKey(req);
      if (store[key] && opts.mode !== "record") return structuredClone(store[key]);
      if (opts.mode === "replay" || !opts.inner) {
        throw Object.assign(new Error(`cassette miss for request ${key.slice(0, 12)}: re-record ${opts.path}`), { retryable: false });
      }
      const res = await opts.inner.complete(req);
      store[key] = res;
      writeFileSync(opts.path, JSON.stringify(store, null, 2));
      return res;
    },
  };
}

export function requestKey(req: ModelRequest): string {
  const { signal: _signal, ...rest } = req;
  return createHash("sha256").update(stableStringify(rest)).digest("hex");
}

/** A model that plays a fixed script of responses, for unit tests. */
export function scriptedModel(script: Array<ModelResponse | Error | ((req: ModelRequest) => ModelResponse)>): Model & { calls: ModelRequest[] } {
  const calls: ModelRequest[] = [];
  return {
    name: "scripted",
    calls,
    async complete(req) {
      calls.push(structuredClone({ ...req, signal: undefined }));
      const next = script.shift();
      if (next === undefined) throw Object.assign(new Error("script exhausted"), { retryable: false });
      if (next instanceof Error) throw next;
      return typeof next === "function" ? next(req) : next;
    },
  };
}
