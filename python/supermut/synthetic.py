"""Synthetic training pairs from classic mutation operators.

Mined bug-fix pairs teach *natural* mutation styles but under-represent the
boundary/operator flips that kill-rate work lives on (the exact class the
mutmut catalogue encodes). This module applies those operators to real
functions and emits (original, mutated) pairs in the same training format,
so the model is taught boundary flips directly.

Operator set distilled from mutmut's `mutation/mutators.py` (see the
mutmut-borrow analysis): comparison flips, arithmetic swaps, boolean
swaps, keyword flips, constant tweaks.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

__all__ = ["SyntheticPair", "mutate_function", "pairs_from_module"]

# node-level swaps: each mutant changes exactly one node occurrence
_CMP_SWAPS = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}
_BIN_SWAPS = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Div,
    ast.Mod: ast.Div,
}
_BOOL_SWAPS = {ast.And: ast.Or, ast.Or: ast.And}


@dataclass
class SyntheticPair:
    function: str
    original: str  # prompt side
    mutated: str  # completion side
    operator: str


class _SingleMutation(ast.NodeTransformer):
    """Applies the i-th applicable mutation site, counting in visit order."""

    def __init__(self, site: int):
        self.site = site
        self.count = -1
        self.applied: str | None = None

    def _hit(self) -> bool:
        self.count += 1
        return self.count == self.site

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        new_ops = []
        for op in node.ops:
            swap = _CMP_SWAPS.get(type(op))
            if swap is not None and self.applied is None and self._hit():
                new_ops.append(swap())
                self.applied = f"cmp:{type(op).__name__}->{swap.__name__}"
            else:
                new_ops.append(op)
        node.ops = new_ops
        return node

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        swap = _BIN_SWAPS.get(type(node.op))
        if swap is not None and self.applied is None and self._hit():
            node.op = swap()
            self.applied = f"bin:{type(node.op).__name__}"
        return node

    def visit_BoolOp(self, node: ast.BoolOp):
        self.generic_visit(node)
        swap = _BOOL_SWAPS.get(type(node.op))
        if swap is not None and self.applied is None and self._hit():
            node.op = swap()
            self.applied = f"bool:{type(node.op).__name__}"
        return node

    def visit_Constant(self, node: ast.Constant):
        if (
            isinstance(node.value, int)
            and not isinstance(node.value, bool)
            and self.applied is None
            and self._hit()
        ):
            node.value = node.value + 1
            self.applied = "const:int+1"
        elif isinstance(node.value, bool) and self.applied is None and self._hit():
            node.value = not node.value
            self.applied = "const:bool-flip"
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and self.applied is None and self._hit():
            self.applied = "unary:drop-not"
            return node.operand
        return node


def mutate_function(func_source: str, *, max_mutants: int = 6) -> list[tuple[str, str]]:
    """All single-site operator mutants of a function: (mutated_source, op)."""
    try:
        tree = ast.parse(func_source)
    except SyntaxError:
        return []
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for site in range(64):  # sites beyond any realistic function just miss
        if len(results) >= max_mutants:
            break
        mutator = _SingleMutation(site)
        mutated_tree = mutator.visit(copy.deepcopy(tree))
        if mutator.applied is None:
            if mutator.count < site:
                break  # ran out of sites
            continue
        try:
            mutated = ast.unparse(mutated_tree)
        except Exception:
            continue
        norm = ast.dump(ast.parse(mutated), annotate_fields=False)
        if norm in seen:
            continue
        seen.add(norm)
        results.append((mutated, mutator.applied))
    return results


def pairs_from_module(
    module_source: str, *, max_function_lines: int = 60, per_function: int = 2
) -> list[SyntheticPair]:
    """Synthetic pairs for every function in a module (top functions only)."""
    from supermut.dataset import _functions_by_name

    pairs: list[SyntheticPair] = []
    for name, func in _functions_by_name(module_source).items():
        if len(func.split("\n")) > max_function_lines:
            continue
        # unparse the original too, so prompt and completion share the exact
        # same normalized style (ast.unparse output on both sides)
        try:
            original = ast.unparse(ast.parse(func))
        except SyntaxError:
            continue
        for mutated, op in mutate_function(original, max_mutants=per_function):
            pairs.append(
                SyntheticPair(
                    function=name, original=original, mutated=mutated, operator=op
                )
            )
    return pairs


def to_prompt_completion(pair: SyntheticPair) -> dict[str, str]:
    from supermut.dataset import Sample, to_prompt_completion as _tpc

    return _tpc(
        Sample(
            repo="synthetic",
            commit=pair.operator,
            file="",
            function=pair.function,
            fixed_source=pair.original,
            buggy_source=pair.mutated,
        )
    )
