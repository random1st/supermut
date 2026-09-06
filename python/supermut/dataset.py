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
    language: str = "python"


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


def _functions_via_language(module_source: str, language) -> dict[str, str]:
    """name -> source via a Language frontend; duplicate names skipped
    (tree-sitter targets aren't class-qualified, so same-named methods in
    different classes would produce false pairs)."""
    counts: dict[str, int] = {}
    sources: dict[str, str] = {}
    try:
        targets = language.find_targets(module_source)
    except Exception:
        return {}
    for t in targets:
        counts[t.name] = counts.get(t.name, 0) + 1
        sources[t.name] = t.source
    return {n: s for n, s in sources.items() if counts[n] == 1}


def extract_function_pairs(
    old_source: str,
    new_source: str,
    *,
    max_changed_lines: int = 12,
    max_function_lines: int = 60,
    language=None,
) -> list[tuple[str, str, str]]:
    """(name, old_func, new_func) for functions whose AST changed.

    Returns [] unless exactly one function changed — multi-function diffs
    are refactor-shaped, not fix-shaped. Functions longer than
    ``max_function_lines`` are skipped: a small model can't attend to a
    168-line prompt, and huge functions dilute the "one subtle change"
    signal anyway.

    ``language=None`` keeps the Python-native path (class-qualified names);
    passing a Language frontend mines that language instead.
    """
    if language is not None and getattr(language, "name", "python") != "python":
        old_funcs = _functions_via_language(old_source, language)
        new_funcs = _functions_via_language(new_source, language)
        norm = language.normalize
    else:
        old_funcs = _functions_by_name(old_source)
        new_funcs = _functions_by_name(new_source)
        norm = _normalize
    changed = []
    for name in old_funcs.keys() & new_funcs.keys():
        old_f, new_f = old_funcs[name], new_funcs[name]
        if max(len(old_f.split("\n")), len(new_f.split("\n"))) > max_function_lines:
            continue
        old_n, new_n = norm(old_f), norm(new_f)
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


class _GitCatFile:
    """Persistent `git cat-file --batch`: one process for the whole mine.

    Spawning `git show` per file per commit dominates mining time on repos
    with deep pack deltas (sqlalchemy, django) — hours instead of minutes.
    """

    def __init__(self, repo: Path):
        self._proc = subprocess.Popen(
            ["git", "-C", str(repo), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def read(self, commit: str, path: str) -> str | None:
        assert self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(f"{commit}:{path}\n".encode())
        self._proc.stdin.flush()
        header = self._proc.stdout.readline().decode()
        if header.endswith("missing\n") or "blob" not in header:
            return None
        size = int(header.rsplit(" ", 1)[1])
        body = self._proc.stdout.read(size)
        self._proc.stdout.read(1)  # trailing newline
        try:
            return body.decode()
        except UnicodeDecodeError:
            return None

    def close(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait()


def mine_repo(
    repo: str | Path,
    *,
    max_commits: int = 2000,
    max_changed_lines: int = 12,
    languages: list[str] | None = None,
) -> list[Sample]:
    """Walk fix-shaped commits, emit (fixed, buggy) function pairs.

    ``languages`` limits mining to those frontends (default: all
    registered — python, javascript, typescript, kotlin, swift).
    """
    from supermut.languages import all_languages, language_for_path

    langs = all_languages()
    if languages is not None:
        langs = {k: v for k, v in langs.items() if k in languages}
    extensions = tuple(ext for lang in langs.values() for ext in lang.extensions)
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
    cat = _GitCatFile(repo)
    try:
        for commit in log.split():
            files = _git(
                repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit
            ).split()
            src_files = [
                f for f in files if f.endswith(extensions) and "test" not in f.lower()
            ]
            # A focused fix touches few files; skip sprawling commits entirely.
            if not src_files or len(src_files) > 3:
                continue
            for path in src_files:
                language = language_for_path(path)
                if language is None or language.name not in langs:
                    continue
                old = cat.read(f"{commit}~1", path)
                new = cat.read(commit, path)
                if old is None or new is None:
                    continue
                for name, old_func, new_func in extract_function_pairs(
                    old,
                    new,
                    max_changed_lines=max_changed_lines,
                    language=language,
                ):
                    samples.append(
                        Sample(
                            repo=repo.name,
                            commit=commit,
                            file=path,
                            function=name,
                            fixed_source=new_func,
                            buggy_source=old_func,
                            language=language.name,
                        )
                    )
    finally:
        cat.close()
    return samples


def to_prompt_completion(sample: Sample) -> dict[str, str]:
    """Render a sample in the exact harness prompt format.

    The comment prefix follows the sample's language, so one multilingual
    model sees `#` for Python and `//` for JS/TS/Kotlin/Swift.
    """
    from supermut.languages import language_by_name

    cp = language_by_name(sample.language).comment_prefix
    header = sample.buggy_source.split("\n")[0]
    mutant_header = "\n".join(
        cp + line.lstrip("#") for line in _MUTANT_HEADER.rstrip().split("\n")
    )
    prompt = (
        f"{cp} Original function:\n"
        f"{sample.fixed_source}\n\n"
        f"{mutant_header}\n"
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
