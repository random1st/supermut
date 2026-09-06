"""Tests for supermut. Integration tests need a GGUF model.

Run integration: SUPERMUT_TEST_MODEL=/path/to/model.gguf pytest
(or have ggml-org/Qwen3.5-0.8B-GGUF Q8_0 in the HF cache).
"""

import os
from pathlib import Path

import pytest

import supermut
from supermut.models import resolve_model

SMOKE_REPO = "unsloth/gemma-3-270m-it-GGUF"
SMOKE_FILE = "gemma-3-270m-it-Q4_0.gguf"


def _find_test_model() -> Path | None:
    env = os.environ.get("SUPERMUT_TEST_MODEL")
    if env and Path(env).exists():
        return Path(env)
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(SMOKE_REPO, SMOKE_FILE, local_files_only=True))
    except Exception:
        return None


MODEL = _find_test_model()
needs_model = pytest.mark.skipif(MODEL is None, reason="no test GGUF available")


def test_import():
    assert hasattr(supermut, "LLM")
    assert hasattr(supermut, "resolve_model")


def test_resolve_model_local_file(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"stub")
    assert resolve_model(f) == f


def test_resolve_model_local_dir_single(tmp_path):
    f = tmp_path / "only.gguf"
    f.write_bytes(b"stub")
    assert resolve_model(tmp_path) == f


def test_resolve_model_local_dir_ambiguous(tmp_path):
    (tmp_path / "a.gguf").write_bytes(b"x")
    (tmp_path / "b.gguf").write_bytes(b"x")
    with pytest.raises(ValueError, match="multiple"):
        resolve_model(tmp_path)
    assert resolve_model(tmp_path, filename="a.gguf") == tmp_path / "a.gguf"


def test_resolve_model_missing_filename(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_model(tmp_path, filename="nope.gguf")


@pytest.fixture(scope="module")
def llm():
    return supermut.LLM(MODEL, n_ctx=2048)


@needs_model
def test_generate(llm):
    out = llm.generate("The capital of France is", max_tokens=8, temperature=0.0)
    assert isinstance(out, str)
    assert len(out) > 0


@needs_model
def test_generate_deterministic_greedy(llm):
    a = llm.generate("1 + 1 =", max_tokens=4, temperature=0.0)
    b = llm.generate("1 + 1 =", max_tokens=4, temperature=0.0)
    assert a == b


@needs_model
def test_generate_batch_shared_prefix(llm):
    prefix = "def add(a, b):\n    return"
    outs = llm.generate_batch(prefix, ["", " a +", " a -"], max_tokens=6, temperature=0.0)
    assert len(outs) == 3
    assert all(isinstance(o, str) and o for o in outs)


@needs_model
def test_stop_sequences(llm):
    out = llm.generate("Count: 1, 2, 3, 4", max_tokens=64, temperature=0.0, stop=["7"])
    assert "7" not in out


@needs_model
def test_count_tokens(llm):
    n = llm.count_tokens("hello world")
    assert 1 <= n <= 8
