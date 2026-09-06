"""Mutant generation: carve a module into function targets, prompt the model
for buggy variants, validate and dedupe the results."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supermut.engine import LLM

__all__ = ["FunctionTarget", "Mutant", "find_targets", "generate_mutants"]


@dataclass
class FunctionTarget:
    """One function eligible for mutation."""

    name: str
    source: str  # exact segment, original indentation, wrapper-free
    start_line: int  # 1-based, inclusive (includes decorators)
    end_line: int  # 1-based, inclusive
    indent: str
    # First-line text between the indent and the function itself that the
    # span replacement must preserve (JS/TS `export ` / `export default `).
    # The prompt, validation, and mutants all see the bare function.
    prefix: str = ""


@dataclass
class Mutant:
    """A validated candidate mutation for one target."""

    target: FunctionTarget
    source: str  # replacement function source, original indentation
    origin: str = "llm"  # "llm" | "operator"


def find_targets(module_source: str) -> list[FunctionTarget]:
    """All function/method definitions in the module, outermost first.

    Nested functions are not returned separately — they mutate as part of
    their parent.
    """
    tree = ast.parse(module_source)
    targets: list[FunctionTarget] = []

    def visit(node: ast.AST, inside_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            is_func = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            if is_func and not inside_function:
                seg = ast.get_source_segment(module_source, child, padded=True)
                if seg is None:
                    continue
                start = min(
                    [child.lineno] + [d.lineno for d in child.decorator_list]
                )
                indent = " " * child.col_offset
                targets.append(
                    FunctionTarget(
                        name=child.name,
                        source=seg,
                        start_line=start,
                        end_line=child.end_lineno or child.lineno,
                        indent=indent,
                    )
                )
            visit(child, inside_function or is_func)

    visit(tree, False)
    return targets


# Single source of the prompt scaffolding text. dataset.to_prompt_completion
# renders training pairs from the same strings — train == serve, per language.
_MUTANT_HEADER = (
    "# Buggy mutant of the function above. Same signature, one subtle\n"
    "# logic change (operator, comparison, boundary, or constant):\n"
)


def _mutant_header(comment_prefix: str) -> str:
    return (
        "\n".join(
            comment_prefix + line.lstrip("#")
            for line in _MUTANT_HEADER.rstrip().split("\n")
        )
        + "\n"
    )


def _few_shot_block(comment_prefix: str, original: str, mutant: str) -> str:
    return (
        f"{comment_prefix} Original function:\n"
        f"{original}\n\n"
        f"{_mutant_header(comment_prefix)}"
        f"{mutant}\n\n"
    )


# One worked example teaches base models to vary instead of copying; a
# fine-tuned mutant model sees the exact same format it was trained on.
_FEW_SHOTS = {
    "python": _few_shot_block(
        "#",
        "def is_positive(x):\n    return x > 0",
        "def is_positive(x):\n    return x >= 0",
    ),
    "javascript": _few_shot_block(
        "//",
        "function isPositive(x) {\n    return x > 0;\n}",
        "function isPositive(x) {\n    return x >= 0;\n}",
    ),
    "kotlin": _few_shot_block(
        "//",
        "fun isPositive(x: Int): Boolean {\n    return x > 0\n}",
        "fun isPositive(x: Int): Boolean {\n    return x >= 0\n}",
    ),
    "swift": _few_shot_block(
        "//",
        "func isPositive(_ x: Int) -> Bool {\n    return x > 0\n}",
        "func isPositive(_ x: Int) -> Bool {\n    return x >= 0\n}",
    ),
}
_FEW_SHOTS["typescript"] = _FEW_SHOTS["javascript"]

# Generation stop sequences: a new comment block or a new top-level
# definition means the model has moved past the mutant body.
_STOP_SEQS = {
    "python": ["\n# ", "\ndef ", "\nclass ", "\nif __name__"],
    "javascript": ["\n// ", "\nfunction ", "\nclass ", "\nconst ", "\nexport "],
    "kotlin": ["\n// ", "\nfun ", "\nclass ", "\nobject "],
    "swift": ["\n// ", "\nfunc ", "\nclass ", "\nstruct ", "\nextension "],
}
_STOP_SEQS["typescript"] = _STOP_SEQS["javascript"]


def build_prompt(target: FunctionTarget, language=None) -> str:
    """Prompt prefix shared by every mutant of this target (KV-cache friendly).

    ``language=None`` means Python. The output for Python is byte-identical
    to what the v2 model was fine-tuned on — do not reformat.
    """
    name = getattr(language, "name", "python")
    cp = "#" if language is None else language.comment_prefix
    func = _dedent(target.source, target.indent)
    header = func.split("\n")[0]
    return (
        f"{_FEW_SHOTS[name]}"
        f"{cp} Original function:\n"
        f"{func}\n\n"
        f"{_mutant_header(cp)}"
        f"{header}\n"
    )


def generate_mutants(
    llm: "LLM",
    module_source: str,
    *,
    n_per_target: int = 8,
    max_tokens: int = 192,
    temperature: float = 0.9,
    seed: int = 42,
    targets: list[FunctionTarget] | None = None,
    language=None,
) -> list[Mutant]:
    """Generate, validate, and dedupe mutants for every target in the module.

    ``language=None`` keeps the shipped Python path (ast trim + validate);
    passing a Language frontend routes trimming, validation, and dedupe
    through it instead.
    """
    lang_name = getattr(language, "name", "python")
    if targets is None:
        targets = (
            find_targets(module_source)
            if language is None
            else language.find_targets(module_source)
        )
    normalize = _normalize if lang_name == "python" else language.normalize
    mutants: list[Mutant] = []
    for t_idx, target in enumerate(targets):
        prompt = build_prompt(target, language)
        header = _dedent(target.source, target.indent).split("\n")[0]
        completions = llm.generate_batch(
            prompt,
            [""] * n_per_target,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed + t_idx * 1000,
            stop=_STOP_SEQS[lang_name],
        )
        seen: set[str] = set()
        original_norm = normalize(_dedent(target.source, target.indent))
        if original_norm is not None:
            seen.add(original_norm)
        for body in completions:
            raw = header + "\n" + body.rstrip()
            if lang_name == "python":
                candidate = _trim_to_function(raw)
                if candidate is not None and not _is_single_function(
                    candidate, target.name
                ):
                    candidate = None
            else:
                candidate = _trim_to_valid(raw, target.name, language)
            if candidate is None:
                continue
            norm = normalize(candidate)
            if norm is None or norm in seen:
                continue
            seen.add(norm)
            mutants.append(
                Mutant(target=target, source=_reindent(candidate, target.indent))
            )
    return mutants


def _dedent(source: str, indent: str) -> str:
    if not indent:
        return source
    lines = [
        line[len(indent) :] if line.startswith(indent) else line
        for line in source.split("\n")
    ]
    return "\n".join(lines)


def _reindent(source: str, indent: str) -> str:
    if not indent:
        return source
    return "\n".join(indent + line if line.strip() else line for line in source.split("\n"))


def _trim_to_function(candidate: str) -> str | None:
    """Trim trailing junk until the candidate parses; None if it never does."""
    lines = candidate.split("\n")
    min_len = max(2, len(lines) // 3)
    while len(lines) >= min_len:
        text = "\n".join(lines)
        try:
            ast.parse(text)
            return text
        except SyntaxError:
            lines.pop()
    return None


def _trim_to_valid(candidate: str, name: str, language) -> str | None:
    """Language-frontend trim: pop trailing lines until the candidate is
    exactly one function named ``name``; None if it never is."""
    lines = candidate.split("\n")
    min_len = max(2, len(lines) // 3)
    while len(lines) >= min_len:
        text = "\n".join(lines)
        if language.validate(text, name):
            return text
        lines.pop()
    return None


def _normalize(source: str) -> str | None:
    """AST-normalized form: whitespace/comment-insensitive dedupe key."""
    try:
        return ast.dump(ast.parse(source))
    except SyntaxError:
        return None


def _is_single_function(source: str, name: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return (
        len(tree.body) == 1
        and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
        and tree.body[0].name == name
        and len(tree.body[0].body) > 0
    )
