"""The Language protocol: everything language-specific lives behind it.

The engine (Rust) never sees source code; the harness, miner, and prompt
builder see only FunctionTarget spans produced here. Adding a language
means implementing this protocol — nothing above it changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from supermut.mutate import FunctionTarget

__all__ = ["Language", "registry", "language_for_path", "language_by_name"]


@runtime_checkable
class Language(Protocol):
    name: str
    extensions: tuple[str, ...]
    comment_prefix: str  # for the few-shot prompt scaffolding

    def find_targets(self, module_source: str) -> list[FunctionTarget]:
        """Function/method definitions, outermost first."""
        ...

    def validate(self, candidate: str, name: str) -> bool:
        """Does the candidate parse as exactly one function named `name`?"""
        ...

    def normalize(self, source: str) -> str | None:
        """Whitespace/comment-insensitive dedupe key, None if unparseable."""
        ...


registry: dict[str, Language] = {}


def register(lang: Language) -> Language:
    registry[lang.name] = lang
    return lang


def language_for_path(path: str) -> Language | None:
    for lang in registry.values():
        if any(path.endswith(ext) for ext in lang.extensions):
            return lang
    return None


def language_by_name(name: str) -> Language:
    _ensure_loaded()
    return registry[name]


def _ensure_loaded() -> None:
    # Import for registration side effects. tree-sitter grammars are an
    # optional extra — without them only Python registers.
    from supermut.languages import python_lang  # noqa: F401

    try:
        from supermut.languages import treesitter  # noqa: F401
    except ImportError:
        pass


def all_languages() -> dict[str, Language]:
    _ensure_loaded()
    return dict(registry)
