// Node binding tests. Integration needs a GGUF model in the HF cache
// (unsloth/gemma-3-270m-it-GGUF Q4_0, same as the Python suite) or
// SUPERMUT_TEST_MODEL pointing at any .gguf.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

import { Engine } from "../index.js";

function findModel() {
  const env = process.env.SUPERMUT_TEST_MODEL;
  if (env && existsSync(env)) return env;
  const snapDir = join(
    homedir(),
    ".cache/huggingface/hub/models--unsloth--gemma-3-270m-it-GGUF/snapshots"
  );
  if (!existsSync(snapDir)) return null;
  for (const snap of readdirSync(snapDir)) {
    const p = join(snapDir, snap, "gemma-3-270m-it-Q4_0.gguf");
    if (existsSync(p)) return p;
  }
  return null;
}

const model = findModel();

test("constructor rejects missing model", () => {
  assert.throws(() => new Engine("/nonexistent/model.gguf"));
});

test("generate + batch + countTokens", { skip: model === null }, () => {
  const engine = new Engine(model, 2048);
  assert.equal(engine.nCtx, 2048);

  const out = engine.generate("The capital of France is", {
    maxTokens: 8,
    temperature: 0,
  });
  assert.equal(typeof out, "string");
  assert.ok(out.length > 0);

  // greedy determinism
  const again = engine.generate("The capital of France is", {
    maxTokens: 8,
    temperature: 0,
  });
  assert.equal(out, again);

  const outs = engine.generateBatch(
    "def add(a, b):\n    return",
    ["", " a +", " a -"],
    { maxTokens: 6, temperature: 0 }
  );
  assert.equal(outs.length, 3);
  for (const o of outs) assert.ok(typeof o === "string" && o.length > 0);

  const n = engine.countTokens("hello world");
  assert.ok(n >= 1 && n <= 8);

  const stopped = engine.generate("Count: 1, 2, 3, 4", {
    maxTokens: 64,
    temperature: 0,
    stop: ["7"],
  });
  assert.ok(!stopped.includes("7"));
});
