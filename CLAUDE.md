# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LLM-driven mutation testing. A small fine-tuned model (gemma-3-270m LoRA, published as `random1st/supermut-gemma-3-270m-v2-GGUF`) generates "natural" mutants; a harness applies each one, runs the test suite, and scores it by which mutants *survive* — every survivor is a missing test. Rule-based operator mutants run alongside by default (hybrid mode). Tools are scored by **known test holes exposed**, not raw kill rate — crude mutants inflate kill rate while finding nothing (see `bench/run_bench.py`).

## Commands

```bash
# Python: build + test (uses .venv; maturin needs the venv on PATH)
uv venv .venv && VIRTUAL_ENV=$PWD/.venv uv pip install maturin huggingface_hub coverage pytest \
  tree-sitter tree-sitter-javascript tree-sitter-typescript tree-sitter-kotlin tree-sitter-swift
VIRTUAL_ENV=$PWD/.venv PATH="$PWD/.venv/bin:$PATH" maturin develop --release
./.venv/bin/pytest python/tests -q                       # full suite
./.venv/bin/pytest python/tests/test_harness.py::test_full_cycle -q   # one test

# Rust workspace
cargo check && cargo fmt --check && cargo clippy --workspace -- -D warnings

# Node binding
cd crates/node && bun install && bunx napi build --platform --release && npm test

# Mutation run (hybrid = default; --operators-only needs no model)
python -m supermut path/to/module.py --tests "-q tests/" \
  --model random1st/supermut-gemma-3-270m-v2-GGUF --filename supermut-gemma-v2-Q8_0.gguf

# Benchmark (mutmut control arm always runs)
./.venv/bin/python bench/run_bench.py --model <gguf-path> --label my-model

# Mine fine-tune pairs from git history (all 5 languages by extension)
python -m supermut.dataset <repo-paths...> -o pairs.jsonl
```

Integration tests need a GGUF in the HF cache (`unsloth/gemma-3-270m-it-GGUF` Q4_0) or `SUPERMUT_TEST_MODEL=/path/to.gguf`; they skip otherwise. Tests must never run from the repo root against the source tree — CI runs them from `runner.temp` against the installed wheel, because `python/supermut/` shadows the wheel and has no compiled `_core`.

## Architecture

One Rust engine, thin bindings, language frontends behind a protocol:

- **`crates/core`** — the entire inference engine (`llama-cpp-2`, llama.cpp statically embedded; Metal on macOS, CPU on Linux, `cuda`/`static-stdcxx` as features). Binding-free: errors are `Result<_, String>`. The hot path `generate_batch` decodes one shared prompt prefix into sequence 0, shares its KV cache with every continuation, and decodes continuations in parallel waves (one token per live sequence per `decode`). **Invariant: the core never sees source code — only tokens/strings.**
- **`crates/py`** (PyO3 → maturin wheel, wired via `manifest-path` in pyproject) and **`crates/node`** (napi-rs → npm addon; `index.js`/`index.d.ts`/`*.node` are build artifacts, gitignored) — pure delegation, no logic.
- **`python/supermut/`** — the harness. `mutate.py` (targets, few-shot prompt, AST-dump dedupe), `selection.py` (one instrumented coverage-contexts run maps function→tests + test→durations; zero-test mutants get a distinct `NO_TESTS` verdict without running), `harness.py` (canary gate, per-mutant test subsets cheapest-first with `-x`, file swap with guaranteed restore, hybrid mutant generation), `cache.py` (AST-hash keyed; caches verdicts *and* generated mutants — reruns cost no GPU tokens), `dataset.py` (miner: reversed bug-fix diffs; persistent `git cat-file --batch`, never per-file `git show`), `synthetic.py` (operator-mutation training pairs, also the operator arm of hybrid runs).
- **`python/supermut/languages/`** — the `Language` protocol (`find_targets`/`validate`/`normalize`). Python uses `ast`; JS/TS/Kotlin/Swift share one generic tree-sitter implementation (optional extra `[languages]`; without it only Python registers). Everything above this layer is language-neutral.

**Training/serving invariant:** the fine-tune prompt format is byte-identical to the serving prompt (`build_prompt`), per language (`#` for Python, `//` for the rest). Changing one side silently degrades the model.

## Non-obvious constraints (each was a debugged failure)

- `with_kv_unified(true)` in the core is load-bearing: the llama.cpp default gives each sequence its own KV stream and physically copies the prefix — wave decoding becomes *slower* than sequential. Context `n_seq_max` is sized to the actual batch so single generations don't pay for 17 streams.
- The llama.cpp backend is a process-wide singleton with a deinitializing `Drop`; it lives in a `static OnceLock`. Never construct it per-Engine.
- Every test subprocess sets `PYTHONDONTWRITEBYTECODE=1`: a mutant of identical file size restored within the same mtime second leaves its stale `.pyc` valid, and the next clean run imports the mutant.
- Kotlin/Swift use mutation schemata (`schemata.py` + Gradle/SwiftTest runners), never the file-swap cycle: all mutants compile in once behind `SUPERMUT_MUTANT`, read exactly once at process start by an injected helper. Gradle needs the injected init script (env forwarding through the daemon + `upToDateWhen{false}` — without it a run where only the env changed is skipped as UP-TO-DATE and replays the previous verdict). `swift test`: xUnit XML only in `--parallel` mode, and only with the space-form `--xunit-output path`; per-mutant runs need `--skip-build`.
- manylinux wheels require the `static-stdcxx` feature (AlmaLinux 8's GLIBCXX fails the 2_28 symbol audit) — see `.github/workflows/CI.yml` for the working matrix.

## Change control

The repo carries S5D specs (`.s5d/`) — see `AGENTS.md`. Practical effect for edits here: non-trivial changes are gated against `.s5d/packages/infer-core__20260906.s5d.yaml`, and the gate matches **literal relative file paths** listed in the spec's component `paths` (directories don't match). New source files must be added there first; validation/approval is `s5d preview` → `s5d approve` → `s5d verify run-gates`. Run gates as a standalone command and check its exit code — never pipe it (`| tail` swallows the failure).
