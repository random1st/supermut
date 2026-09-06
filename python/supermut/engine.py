"""High-level LLM interface over the Rust core (llama.cpp)."""

from __future__ import annotations

from pathlib import Path

from supermut._core import Engine as _Engine
from supermut.models import resolve_model

__all__ = ["LLM"]


class LLM:
    """Local GGUF model. Metal-accelerated on macOS, CPU elsewhere.

    >>> llm = LLM.from_pretrained("user/my-mutant-model", filename="model-q4_k_m.gguf")
    >>> llm.generate("def add(a, b):", max_tokens=32)
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 4096,
        n_gpu_layers: int = 1_000_000,
    ) -> None:
        self._engine = _Engine(str(model_path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path,
        *,
        filename: str | None = None,
        revision: str | None = None,
        token: str | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 1_000_000,
    ) -> "LLM":
        """Load from a local path or a Hugging Face repo id."""
        path = resolve_model(source, filename=filename, revision=revision, token=token)
        return cls(path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        seed: int = 42,
        stop: list[str] | None = None,
    ) -> str:
        return self._engine.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            seed=seed,
            stop=stop or [],
        )

    def generate_batch(
        self,
        prefix: str,
        continuations: list[str],
        *,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        seed: int = 42,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Generate one completion per continuation, sharing the prefix KV cache.

        The supermut hot path: ``prefix`` is the source-file prompt decoded
        once; each continuation extends it without recomputing the prefix.
        """
        return self._engine.generate_batch(
            prefix,
            continuations,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            seed=seed,
            stop=stop or [],
        )

    def count_tokens(self, text: str) -> int:
        return self._engine.count_tokens(text)

    @property
    def model_path(self) -> str:
        return self._engine.model_path

    @property
    def n_ctx(self) -> int:
        return self._engine.n_ctx

    def __repr__(self) -> str:
        return repr(self._engine)
