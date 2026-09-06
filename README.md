# supermut

LLM-driven mutation testing with a fast local inference core.

A small fine-tuned model generates *natural* mutants — the bug-shaped changes
rule-based tools can't express — and a harness scores your test suite by which
mutants survive. Rule-based operator mutants run alongside by default: on our
benchmark of 8 known test holes, the hybrid finds **8/8** where
[mutmut](https://github.com/boxed/mutmut) finds 7/8 (one hole is provably
unreachable by operator mutations alone).

- **Engine**: Rust + llama.cpp, statically embedded. Metal on macOS, CPU on
  Linux, CUDA as an opt-in build feature. The mutation workload — one shared
  file-prefix, many short completions — decodes in parallel waves over a
  shared KV cache.
- **Bindings**: Python (PyO3/maturin wheel) and Node (napi-rs addon) over the
  same core crate.
- **Harness** (Python, `python -m supermut`): coverage-based test selection,
  per-mutant test subsets (cheapest first, `-x`), canary + clean-run gates,
  AST-hash result cache that also caches generated mutants (reruns cost no
  GPU tokens), timeout-as-killed, `no-tests` as a verdict distinct from
  `survived`.
- **Model**: gemma-3-270m LoRA fine-tune on 9.2k reversed bug-fix diffs mined
  from 12 OSS repos plus synthetic operator mutations. Training prompt format
  is byte-identical to the serving prompt format.

## Quick start (Python)

```bash
uv venv .venv && source .venv/bin/activate
uv pip install maturin && maturin develop --release
python -m supermut path/to/module.py \
  --tests "-q tests/" \
  --model random1st/supermut-gemma-3-270m-v2-GGUF \
  --filename supermut-gemma-v2-Q8_0.gguf
```

```
mutants run: 65 (llm: 27  operator: 38)  killed: 42  survived: 23  no-tests: 0  kill rate: 65%
  SURVIVED [llm] is_even: return not n
  ...
```

Every surviving mutant is a concrete test to write. `--operators-only` runs
without any model; `--llm-only` isolates the model arm; `--json report.json`
emits machine-readable results.

## Library use

```python
from supermut import LLM
llm = LLM.from_pretrained("random1st/supermut-gemma-3-270m-v2-GGUF",
                          filename="supermut-gemma-v2-Q8_0.gguf")
llm.generate_batch(prefix, ["", "", ""], max_tokens=64, stop=["\n\n"])
```

```js
// Node (crates/node): npm run build, then
import { Engine } from "supermut";
const engine = new Engine(modelPath, 2048);
engine.generateBatch(prefix, ["", "", ""], { maxTokens: 64 });
```

## Data & training pipeline

```bash
# mine reversed bug-fix pairs (fixed -> buggy) from git history
python -m supermut.dataset ~/src/flask ~/src/django -o pairs.jsonl
# add synthetic operator mutations, fine-tune (mlx_lm on Apple Silicon),
# convert to GGUF — see bench/ and the commit log for the full recipe
```

## Benchmark

`bench/run_bench.py` scores tools by **known test holes exposed** (surviving
mutants pointing at documented gaps), not raw kill rate — crude mutants
inflate kill rate while finding nothing.

| tool | mutants | holes found | time |
|---|---|---|---|
| mutmut 3.7 | 47 | 7/8 | 0.7s |
| stock gemma-3-270m | 14 | 1/8 | 10.4s |
| fine-tuned v2 (LLM only) | 38 | 5/8 | 10.7s |
| **supermut hybrid (default)** | **65** | **8/8** | 14.9s |

The `is_even` odd-path hole is unreachable by operator mutation (every
inversion dies on the even-input test); only the LLM arm finds it.

## Status

Working end-to-end; Python-first. The engine and bindings are
language-agnostic by design — a JS/TS harness frontend (tree-sitter +
vitest/istanbul) is the planned next target. APIs may still move.

## License

MIT. Model weights inherit the Gemma license from the base model.
