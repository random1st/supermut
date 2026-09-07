"""Synthetic operator-mutation pairs for tree-sitter languages (JS/TS/Kotlin/Swift).

The `//` languages are undertrained relative to Python (1.2k mined pairs vs
7.8k), and fine-tune v3 mostly copies the original for them. Synthetic
operator pairs are the lever that took Python from v1 to v2; this module
is that lever for the other four languages.

Grammar-agnostic by construction: instead of per-language node tables it
mutates the BYTE SPAN of operator/literal tokens. JS/TS/Kotlin expose the
operator as a field on `binary_expression`/`unary_expression`; Swift uses
specialised nodes (`comparison_expression`, `conjunction_expression`, ...)
with the operator as an anonymous child. Both reduce to "an anonymous
token whose text is in the swap table" — verified against the installed
grammars, not assumed. Every mutant is re-parsed through the language's
`validate` before it is kept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tree_sitter

from supermut.languages import language_by_name

__all__ = ["OperatorPair", "mutate_function", "pairs_from_records", "to_prompt_completion"]

# operator text -> replacement (one-way entries are deliberate, mirroring
# mutmut: `%`->`/` and `//`->`/` pick the most different sibling)
_OP_SWAPS = {
    "<": "<=",
    "<=": "<",
    ">": ">=",
    ">=": ">",
    "==": "!=",
    "!=": "==",
    "===": "!==",
    "!==": "===",
    "+": "-",
    "-": "+",
    "*": "/",
    "/": "*",
    "%": "/",
    "&&": "||",
    "||": "&&",
}
_LITERAL_NODES = ("number", "number_literal", "integer_literal")
_BOOL_NODES = ("true", "false")
# node types whose anonymous children are operators we may swap
_EXPR_NODE_HINTS = ("binary", "unary", "comparison", "conjunction", "disjunction",
                    "equality", "additive", "multiplicative", "prefix")


@dataclass
class OperatorPair:
    language: str
    function: str
    original: str  # prompt side
    mutated: str  # completion side
    operator: str  # e.g. "op:<-><=", "const:int+1", "bool:flip", "unary:drop-!"


@dataclass
class _Site:
    start: int
    end: int
    replacement: bytes
    label: str


def _sites(root: tree_sitter.Node, src: bytes) -> list[_Site]:
    sites: list[_Site] = []

    def visit(n: tree_sitter.Node) -> None:
        if any(h in n.type for h in _EXPR_NODE_HINTS):
            for c in n.children:
                if c.is_named:
                    continue
                text = src[c.start_byte : c.end_byte].decode()
                if text in _OP_SWAPS:
                    sites.append(
                        _Site(c.start_byte, c.end_byte, _OP_SWAPS[text].encode(),
                              f"op:{text}->{_OP_SWAPS[text]}")
                    )
                elif text == "!" and "unary" in n.type or text == "!" and "prefix" in n.type:
                    sites.append(_Site(c.start_byte, c.end_byte, b"", "unary:drop-!"))
        elif n.type in _LITERAL_NODES:
            text = src[n.start_byte : n.end_byte].decode()
            if text.isdigit():
                sites.append(
                    _Site(n.start_byte, n.end_byte, str(int(text) + 1).encode(), "const:int+1")
                )
        elif n.type in _BOOL_NODES:
            flip = b"false" if n.type == "true" else b"true"
            sites.append(_Site(n.start_byte, n.end_byte, flip, "bool:flip"))
        for c in n.children:
            visit(c)

    visit(root)
    return sites


def mutate_function(
    func_source: str, language_name: str, *, max_mutants: int = 6
) -> list[tuple[str, str]]:
    """Single-site operator mutants of one function: (mutated_source, label).

    Only mutants that the language frontend validates as exactly one
    function with the original name are kept; AST-normalized dedupe drops
    duplicates and anything equivalent to the original.
    """
    lang = language_by_name(language_name)
    src = func_source.encode()
    tree = tree_sitter.Parser(lang._language).parse(src)
    if tree.root_node.has_error:
        return []
    targets = lang.find_targets(func_source)
    if len(targets) != 1:
        return []
    name = targets[0].name
    orig_norm = lang.normalize(func_source)
    seen = {orig_norm} if orig_norm else set()
    out: list[tuple[str, str]] = []
    for site in _sites(tree.root_node, src):
        if len(out) >= max_mutants:
            break
        mutated = (src[: site.start] + site.replacement + src[site.end :]).decode()
        if not lang.validate(mutated, name):
            continue
        norm = lang.normalize(mutated)
        if norm is None or norm in seen:
            continue
        seen.add(norm)
        out.append((mutated, site.label))
    return out


def to_prompt_completion(pair: OperatorPair) -> dict[str, str]:
    """Render in the exact serving prompt format (via dataset.Sample)."""
    from supermut.dataset import Sample
    from supermut.dataset import to_prompt_completion as _tpc

    return _tpc(
        Sample(
            repo="synthetic",
            commit=pair.operator,
            file="",
            function=pair.function,
            fixed_source=pair.original,
            buggy_source=pair.mutated,
            language=pair.language,
        )
    )


def pairs_from_records(
    records, *, per_function: int = 3, max_function_lines: int = 60
) -> list[OperatorPair]:
    """OperatorPairs from mined raw records ({language, function, fixed_source, ...})."""
    pairs: list[OperatorPair] = []
    seen_src: set[str] = set()
    for r in records:
        lang_name = r.get("language")
        src = r.get("fixed_source", "")
        if not lang_name or lang_name == "python" or not src or src in seen_src:
            continue
        if len(src.split("\n")) > max_function_lines:
            continue
        seen_src.add(src)
        for mutated, label in mutate_function(src, lang_name, max_mutants=per_function):
            pairs.append(
                OperatorPair(
                    language=lang_name,
                    function=r.get("function", ""),
                    original=src,
                    mutated=mutated,
                    operator=label,
                )
            )
    return pairs


def main(argv: list[str] | None = None) -> int:
    import argparse
    from collections import Counter

    p = argparse.ArgumentParser(prog="supermut.synthetic_ts")
    p.add_argument("inputs", nargs="+", type=Path, help="raw mined JSONL files")
    p.add_argument("-o", "--out", required=True, type=Path)
    p.add_argument("--per-function", type=int, default=3)
    a = p.parse_args(argv)

    records = []
    for path in a.inputs:
        with open(path) as f:
            records += [json.loads(line) for line in f if line.strip()]
    pairs = pairs_from_records(records, per_function=a.per_function)
    with open(a.out, "w") as f:
        for pair in pairs:
            f.write(json.dumps(to_prompt_completion(pair)) + "\n")
    by_lang = Counter(pr.language for pr in pairs)
    by_op = Counter(pr.operator.split(":")[0] for pr in pairs)
    print(f"records={len(records)} pairs={len(pairs)} by_lang={dict(by_lang)} by_class={dict(by_op)}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
