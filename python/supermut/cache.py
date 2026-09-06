"""Per-function result cache keyed by AST hash.

Borrowed from mutmut's incremental design, extended for LLM generation: for
an unchanged function we reuse not only test verdicts but also the generated
mutants themselves — regeneration costs GPU tokens, which mutmut never had
to worry about. Comments/formatting changes don't invalidate (AST hash).
"""

from __future__ import annotations

import ast
import hashlib
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["FunctionCache", "function_hash"]

CACHE_NAME = ".supermut-cache.json"


def function_hash(func_source: str) -> str:
    """Format-insensitive hash of one function's AST.

    Accepts method sources with class-level indentation.
    """
    dumped = ast.dump(
        ast.parse(textwrap.dedent(func_source)), annotate_fields=False
    )
    return hashlib.sha256(dumped.encode()).hexdigest()[:16]


@dataclass
class _Entry:
    hash: str
    mutants: list[str] = field(default_factory=list)
    # normalized-mutant-hash -> verdict string
    verdicts: dict[str, str] = field(default_factory=dict)


class FunctionCache:
    """JSON sidecar: {file: {func_name: {hash, mutants, verdicts}}}.

    ``hash_fn`` keys mutant verdicts; the default is the Python AST hash.
    Non-Python languages pass their own (e.g. tree-sitter normalize).
    """

    def __init__(self, path: Path, hash_fn=function_hash):
        self._hash_fn = hash_fn
        self._path = path
        self._data: dict[str, dict[str, _Entry]] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            for file_key, funcs in raw.items():
                self._data[file_key] = {
                    name: _Entry(**entry) for name, entry in funcs.items()
                }

    def save(self) -> None:
        raw = {
            file_key: {
                name: {
                    "hash": e.hash,
                    "mutants": e.mutants,
                    "verdicts": e.verdicts,
                }
                for name, e in funcs.items()
            }
            for file_key, funcs in self._data.items()
        }
        self._path.write_text(json.dumps(raw, indent=1))

    def _entry(self, file_key: str, func_name: str, func_hash: str) -> _Entry:
        funcs = self._data.setdefault(file_key, {})
        entry = funcs.get(func_name)
        if entry is None or entry.hash != func_hash:
            # Function changed (or new): everything cached for it is stale.
            entry = _Entry(hash=func_hash)
            funcs[func_name] = entry
        return entry

    def cached_mutants(
        self, file_key: str, func_name: str, func_hash: str
    ) -> list[str] | None:
        """Previously generated mutants for this exact function version."""
        entry = self._entry(file_key, func_name, func_hash)
        return entry.mutants or None

    def store_mutants(
        self, file_key: str, func_name: str, func_hash: str, mutants: list[str]
    ) -> None:
        self._entry(file_key, func_name, func_hash).mutants = mutants

    def cached_verdict(
        self, file_key: str, func_name: str, func_hash: str, mutant_source: str
    ) -> str | None:
        entry = self._entry(file_key, func_name, func_hash)
        return entry.verdicts.get(self._hash_fn(mutant_source))

    def store_verdict(
        self,
        file_key: str,
        func_name: str,
        func_hash: str,
        mutant_source: str,
        verdict: str,
    ) -> None:
        entry = self._entry(file_key, func_name, func_hash)
        entry.verdicts[self._hash_fn(mutant_source)] = verdict
