"""Python Language implementation — wraps the shipped ast-based frontend."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from supermut.languages.base import register
from supermut.mutate import (
    FunctionTarget,
    _is_single_function,
    _normalize,
    find_targets,
)

__all__ = ["PYTHON"]


@dataclass
class PythonLanguage:
    name: str = "python"
    extensions: tuple[str, ...] = (".py",)
    comment_prefix: str = "#"

    def find_targets(self, module_source: str) -> list[FunctionTarget]:
        return find_targets(module_source)

    def validate(self, candidate: str, name: str) -> bool:
        return _is_single_function(candidate, name)

    def normalize(self, source: str) -> str | None:
        return _normalize(source)


# keep ast import for type parity with treesitter module docs
_ = ast

PYTHON = register(PythonLanguage())
