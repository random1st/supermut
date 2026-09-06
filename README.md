# supermut

Fast local inference for mutation-test generation. Rust core (llama.cpp —
Metal on macOS, CPU on Linux/CI, CUDA opt-in) wrapped in a Python package,
with model loading from the Hugging Face Hub.

## Install (dev)

```bash
uv venv .venv && source .venv/bin/activate
uv pip install maturin huggingface_hub pytest
maturin develop --release
```

## Usage

```python
from supermut import LLM

# From your HF account (private repos need `hf auth login` or HF_TOKEN)
llm = LLM.from_pretrained("you/your-mutant-model-GGUF", filename="model-q4_k_m.gguf")

# Or a local file
llm = LLM("/path/to/model.gguf", n_ctx=4096)

llm.generate("def add(a, b):", max_tokens=32)

# Hot path: one source-file prefix, many mutant completions.
# The prefix KV cache is computed once and reused per continuation.
llm.generate_batch(
    prefix=source_file_prompt,
    continuations=[site1, site2, site3],
    max_tokens=64,
    stop=["\n\n"],
)
```

## Design

- `src/lib.rs` — Rust core (`supermut._core`): `llama-cpp-2` with static
  llama.cpp; `generate_batch` decodes the shared prefix into sequence 0 and
  copies its KV cache per continuation.
- `python/supermut/models.py` — HF Hub resolution into the standard shared
  cache (`~/.cache/huggingface`).
- `python/supermut/engine.py` — `LLM` facade.
- Models are GGUF. Fine-tunes in safetensors must be converted
  (`llama.cpp/convert_hf_to_gguf.py`) — upload the GGUF next to the
  safetensors in the same HF repo.

## Tests

```bash
pytest python/tests            # unit tests always run
SUPERMUT_TEST_MODEL=/path/to/model.gguf pytest python/tests   # + integration
```
