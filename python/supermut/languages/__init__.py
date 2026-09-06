"""Language frontends: Python (ast), JS/TS/Kotlin/Swift (tree-sitter)."""

from supermut.languages.base import (
    Language,
    all_languages,
    language_by_name,
    language_for_path,
)

__all__ = ["Language", "all_languages", "language_by_name", "language_for_path"]
