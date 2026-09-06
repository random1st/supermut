"""supermut — fast local inference for mutation-test generation.

Rust core (llama.cpp: Metal on macOS, CPU on Linux) wrapped in Python,
with model loading from the Hugging Face Hub.
"""

from supermut.engine import LLM
from supermut.models import resolve_model

__all__ = ["LLM", "resolve_model"]
