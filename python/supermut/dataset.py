"""Fine-tune dataset mining: reversed bug-fix diffs from git history.

A bug-fix commit turns a buggy function into a fixed one. Reversing it gives
exactly what the mutant model must learn: fixed function in, realistic buggy
variant out. Samples are emitted in the same prompt format the harness uses
at inference time (`supermut.mutate.build_prompt` body), so training and
serving see identical text.

Selection filters (signal over noise):
- commit message matches fix/bug patterns;
- exactly ONE function changed in a file (multi-function diffs are usually
  refactors, not focused fixes);
- the change is small (<= max_changed_lines within the function);
- both versions parse standalone.
"""

from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = ["Sample", "extract_function_pairs", "mine_repo", "to_prompt_completion"]

# POSIX ERE for git --grep -E: no \b word boundaries there.
FIX_PATTERN = (
    r"(^|[^[:alpha:]])(fix(es|ed)?|bug|bugfix|hotfix|regression"
    r"|incorrect|wrong|off.by.one)([^[:alpha:]]|$)"
)
_MUTANT_HEADER = (
    "# Buggy mutant of the function above. Same signature, one subtle\n"
    "# logic change (operator, comparison, boundary, or constant):\n"
)


@dataclass
class Sample:
    repo: str
    commit: str
    file: str
    function: str
    fixed_source: str  # after the fix — the prompt side
    buggy_source: str  # before the fix — the completion side


def _functions_by_name(module_source: str) -> dict[str, str]:
    """name -> dedented source for every function/method (outermost only)."""
    import warnings

    try:
        with warnings.catch_warnings():
            # Historical sources are full of now-warned escape sequences.
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(module_source)
    except SyntaxError:
        return {}
    out: dict[str, str] = {}

    def visit(node: ast.AST, inside: bool, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            is_func = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            if is_func and not inside:
                seg = ast.get_source_segment(module_source, child)
                if seg is not None:
                    out[prefix + child.name] = seg
            name = (
                prefix + child.name + "."
                if isinstance(child, ast.ClassDef)
                else prefix
            )
            visit(child, inside or is_func, name)

    visit(tree, False, "")
    return out


def _normalize(source: str) -> str | None:
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.dump(ast.parse(source), annotate_fields=False)
    except SyntaxError:
        return None


def extract_function_pairs(
    old_source: str,
    new_source: str,
    *,
    max_changed_lines: int = 12,
    max_function_lines: int = 60,
) -> list[tuple[str, str, str]]:
    """(name, old_func, new_func) for functions whose AST changed.

    Returns [] unless exactly one function changed — multi-function diffs
    are refactor-shaped, not fix-shaped. Functions longer than
    ``max_function_lines`` are skipped: a small model can't attend to a
    168-line prompt, and huge functions dilute the "one subtle change"
    signal anyway.
    """
    old_funcs = _functions_by_name(old_source)
    new_funcs = _functions_by_name(new_source)
    changed = []
    for name in old_funcs.keys() & new_funcs.keys():
        old_f, new_f = old_funcs[name], new_funcs[name]
        if max(len(old_f.split("\n")), len(new_f.split("\n"))) > max_function_lines:
            continue
        old_n, new_n = _normalize(old_f), _normalize(new_f)
        if old_n is None or new_n is None or old_n == new_n:
            continue
        delta = abs(len(old_f.split("\n")) - len(new_f.split("\n")))
        diff_lines = sum(
            a != b for a, b in zip(old_f.split("\n"), new_f.split("\n"))
        ) + delta
        if diff_lines <= max_changed_lines:
            changed.append((name, old_f, new_f))
    return changed if len(changed) == 1 else []


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _show_file(repo: Path, commit: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{path}"],
        capture_output=True,
        text=True,
    )
    return proc.stdout if proc.returncode == 0 else None


def mine_repo(
    repo: str | Path,
    *,
    max_commits: int = 2000,
    max_changed_lines: int = 12,
) -> list[Sample]:
    """Walk fix-shaped commits, emit (fixed, buggy) function pairs."""
    repo = Path(repo).resolve()
    log = _git(
        repo,
        "log",
        f"--max-count={max_commits}",
        "-i",
        f"--grep={FIX_PATTERN}",
        "--extended-regexp",
        "--no-merges",
        "--format=%H",
    )
    samples: list[Sample] = []
    for commit in log.split():
        files = _git(
            repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit
        ).split()
        py_files = [f for f in files if f.endswith(".py") and "test" not in f]
        # A focused fix touches few files; skip sprawling commits entirely.
        if not py_files or len(py_files) > 3:
            continue
        for path in py_files:
            old = _show_file(repo, f"{commit}~1", path)
            new = _show_file(repo, commit, path)
            if old is None or new is None:
                continue
            for name, old_func, new_func in extract_function_pairs(
                old, new, max_changed_lines=max_changed_lines
            ):
                samples.append(
                    Sample(
                        repo=repo.name,
                        commit=commit,
                        file=path,
                        function=name,
                        fixed_source=new_func,
                        buggy_source=old_func,
                    )
                )
    return samples


def to_prompt_completion(sample: Sample) -> dict[str, str]:
    """Render a sample in the exact harness prompt format."""
    header = sample.buggy_source.split("\n")[0]
    prompt = (
        "# Original function:\n"
        f"{sample.fixed_source}\n\n"
        f"{_MUTANT_HEADER}"
        f"{header}\n"
    )
    completion = "\n".join(sample.buggy_source.split("\n")[1:])
    return {"prompt": prompt, "completion": completion}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="supermut.dataset",
        description="Mine reversed bug-fix pairs from git repos into JSONL",
    )
    parser.add_argument("repos", nargs="+", help="paths to git repositories")
    parser.add_argument("-o", "--out", required=True, help="output JSONL path")
    parser.add_argument("--max-commits", type=int, default=2000)
    parser.add_argument("--max-changed-lines", type=int, default=12)
    parser.add_argument(
        "--raw", action="store_true", help="emit raw samples, not prompt/completion"
    )
    args = parser.parse_args(argv)

    total = 0
    with open(args.out, "w") as fh:
        for repo in args.repos:
            samples = mine_repo(
                repo,
                max_commits=args.max_commits,
                max_changed_lines=args.max_changed_lines,
            )
            for s in samples:
                record = asdict(s) if args.raw else to_prompt_completion(s)
                fh.write(json.dumps(record) + "\n")
            total += len(samples)
            print(f"{repo}: {len(samples)} samples")
    print(f"total: {total} -> {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
