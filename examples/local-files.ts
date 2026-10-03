// A local agent that answers questions about a folder, using any OpenAI-compatible local server.
//
//   llama-server -m qwen3-8b-q4_k_m.gguf --jinja --port 8080     (or: ollama serve)
//   node examples/local-files.ts ./docs "What does the deploy script do?"
//
// Set ANTHROPIC_API_KEY and pass --claude to run the same agent on Claude instead.
import { readdir, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { anthropicModel, openAICompatibleModel, runAgent, type Model, type Tool } from "../src/index.ts";

const [dir = ".", question = "Summarize what is in this folder."] = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const root = resolve(dir);

const tools: Tool[] = [
  {
    name: "list_files",
    description: "List files in the folder (relative paths). Call this first.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    run: async () => (await readdir(root, { recursive: true })).slice(0, 500).join("\n"),
  },
  {
    name: "read_file",
    description: "Read a text file by relative path. Returns at most 20,000 characters.",
    inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"], additionalProperties: false },
    run: async ({ path }) => {
      const full = resolve(join(root, path));
      if (!full.startsWith(root)) throw new Error("path is outside the folder");
      return (await readFile(full, "utf8")).slice(0, 20_000);
    },
  },
];

async function pickModel(): Promise<Model> {
  if (process.argv.includes("--claude")) {
    const { default: Anthropic } = await import("@anthropic-ai/sdk" as string);
    return anthropicModel({ client: new Anthropic() });
  }
  return openAICompatibleModel({ baseUrl: process.env.LOCAL_LLM_URL ?? "http://127.0.0.1:8080/v1", model: process.env.LOCAL_LLM_MODEL ?? "local" });
}

const result = await runAgent({
  model: await pickModel(),
  system: "Answer only from the files. Cite the file paths you used.",
  input: question,
  tools,
  maxTurns: 10,
  onEvent: (e) => {
    if (e.type === "tool_call") console.error(`→ ${e.name}(${JSON.stringify(e.input)})`);
    if (e.type === "retry") console.error(`  retry ${e.attempt} in ${e.delayMs} ms: ${e.error}`);
    if (e.type === "compaction") console.error(`  compacted context ${e.beforeTokens} → ${e.afterTokens} tokens`);
  },
});
console.log(result.text);
console.error(`\n[${result.reason}] ${result.usage.modelCalls} model calls, ${result.usage.toolCalls} tool calls`);
