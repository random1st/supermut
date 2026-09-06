"""Mutation-test harness: apply each mutant, run the test suite, score it.

Kill rate = killed / (killed + survived). Timeouts count as killed (the
suite did notice something is wrong). Mutants of functions no test executes
are NO_TESTS — reported separately, never conflating them with survivors.

Safety gates before scoring (both borrowed from mutmut):
- baseline: the instrumented selection run doubles as the clean-test gate;
- canary: every target replaced by ``raise`` at once must turn the suite
  red, else patches aren't reaching the tests and every verdict would lie.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from supermut.cache import CACHE_NAME, FunctionCache, function_hash
from supermut.mutate import (
    FunctionTarget,
    Mutant,
    _dedent,
    _normalize,
    _reindent,
    find_targets,
    generate_mutants,
)
from supermut.selection import SelectionMap, collect_selection

if TYPE_CHECKING:
    from supermut.engine import LLM

__all__ = ["MutantStatus", "MutantResult", "Report", "apply_mutant", "run"]


class MutantStatus(str, Enum):
    KILLED = "killed"
    SURVIVED = "survived"
    TIMEOUT = "timeout"
    NO_TESTS = "no_tests"


@dataclass
class MutantResult:
    mutant: Mutant
    status: MutantStatus
    duration_s: float
    from_cache: bool = False


@dataclass
class Report:
    file: str
    results: list[MutantResult] = field(default_factory=list)

    @property
    def killed(self) -> int:
        return sum(
            r.status in (MutantStatus.KILLED, MutantStatus.TIMEOUT)
            for r in self.results
        )

    @property
    def survived(self) -> int:
        return sum(r.status is MutantStatus.SURVIVED for r in self.results)

    @property
    def no_tests(self) -> int:
        return sum(r.status is MutantStatus.NO_TESTS for r in self.results)

    @property
    def kill_rate(self) -> float:
        scored = self.killed + self.survived
        return self.killed / scored if scored else 0.0

    def summary(self) -> str:
        by_origin: dict[str, int] = {}
        for r in self.results:
            by_origin[r.mutant.origin] = by_origin.get(r.mutant.origin, 0) + 1
        origins = "  ".join(f"{k}: {v}" for k, v in sorted(by_origin.items()))
        lines = [
            f"file: {self.file}",
            f"mutants run: {len(self.results)} ({origins})  killed: {self.killed}  "
            f"survived: {self.survived}  no-tests: {self.no_tests}  "
            f"kill rate: {self.kill_rate:.0%}",
        ]
        for r in self.results:
            if r.status is MutantStatus.SURVIVED:
                first = next(
                    (
                        line.strip()
                        for line in r.mutant.source.split("\n")[1:]
                        if line.strip()
                    ),
                    "",
                )
                lines.append(
                    f"  SURVIVED [{r.mutant.origin}] {r.mutant.target.name}: {first[:80]}"
                )
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "file": self.file,
                "killed": self.killed,
                "survived": self.survived,
                "no_tests": self.no_tests,
                "total": len(self.results),
                "kill_rate": self.kill_rate,
                "mutants": [
                    {
                        "target": r.mutant.target.name,
                        "origin": r.mutant.origin,
                        "status": r.status.value,
                        "duration_s": round(r.duration_s, 3),
                        "from_cache": r.from_cache,
                        "source": r.mutant.source,
                    }
                    for r in self.results
                ],
            },
            indent=1,
        )


def apply_mutant(module_source: str, mutant: Mutant) -> str:
    """Replace the target's line span with the mutant source."""
    lines = module_source.split("\n")
    t = mutant.target
    return "\n".join(
        lines[: t.start_line - 1] + mutant.source.split("\n") + lines[t.end_line :]
    )


def _apply_many(module_source: str, mutants: list[Mutant]) -> str:
    """Apply non-overlapping mutants bottom-up so line spans stay valid."""
    out = module_source
    for m in sorted(mutants, key=lambda m: m.target.start_line, reverse=True):
        out = apply_mutant(out, m)
    return out


def _canary_mutants(targets: list[FunctionTarget]) -> list[Mutant]:
    canaries = []
    for t in targets:
        header = _dedent(t.source, t.indent).split("\n")[0]
        body = f'{header}\n    raise RuntimeError("supermut canary")'
        indented = "\n".join(
            t.indent + line if line.strip() else line for line in body.split("\n")
        )
        canaries.append(Mutant(target=t, source=indented))
    return canaries


def _run_pytest(
    python: str,
    pytest_args: list[str],
    cwd: Path,
    timeout_s: float,
    keyword: str | None = None,
) -> tuple[bool, bool]:
    """Returns (passed, timed_out)."""
    cmd = [python, "-m", "pytest", *pytest_args, "-x"]
    if keyword:
        cmd += ["-k", keyword]
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            timeout=timeout_s,
            # A mutant of identical size restored within the same mtime
            # second would leave its stale .pyc looking valid — the clean
            # run would then import the mutant. Never write bytecode.
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return proc.returncode == 0, False
    except subprocess.TimeoutExpired:
        return False, True


def run(
    file: str | Path,
    tests: str,
    llm: "LLM | None",
    *,
    cwd: str | Path | None = None,
    python: str = sys.executable,
    n_per_target: int = 8,
    max_tokens: int = 192,
    temperature: float = 0.9,
    seed: int = 42,
    timeout_s: float = 60.0,
    use_cache: bool = True,
    operators: bool = True,
    operator_mutants: int = 6,
    on_progress=None,
) -> Report:
    """Full cycle for one module. ``tests`` is the pytest argument string
    (e.g. ``"-q tests/"``). ``llm=None`` runs the operator arm alone.
    The target file is swapped in place with a guaranteed restore; run
    from a clean working tree.
    """
    file = Path(file).resolve()
    cwd = Path(cwd).resolve() if cwd else file.parent
    pytest_args = shlex.split(tests)
    original = file.read_text()
    targets = find_targets(original)

    # Gate 1 — instrumented baseline: suite must pass, and we get the
    # function->tests + test->duration maps in the same run.
    selection: SelectionMap = collect_selection(
        file, pytest_args, cwd, python=python, timeout_s=max(timeout_s * 5, 300.0),
        targets=targets,
    )

    # Gate 2 — canary: all targets raising at once must fail the suite.
    covered = [t for t in targets if selection.tests_by_function.get(t.name)]
    if covered:
        file.write_text(_apply_many(original, _canary_mutants(covered)))
        try:
            passed, _ = _run_pytest(python, pytest_args, cwd, timeout_s)
        finally:
            file.write_text(original)
        if passed:
            raise RuntimeError(
                "canary mutants did not fail the suite — patches are not "
                "reaching the tests (wrong file? swallowed exceptions?); "
                "verdicts would be meaningless"
            )

    cache = FunctionCache(cwd / CACHE_NAME) if use_cache else None
    file_key = str(file)

    # Generate (or recall) mutants per target. Hybrid by default: cheap
    # operator mutants (the mutmut-class catalogue) plus LLM naturals —
    # on the bench the union finds holes neither arm finds alone.
    mutants: list[Mutant] = []
    for t_idx, target in enumerate(targets):
        dedented = _dedent(target.source, target.indent)
        f_hash = function_hash(dedented)
        seen_norms: set[str] = set()

        if operators:
            from supermut.synthetic import mutate_function

            for mutated, _op in mutate_function(dedented, max_mutants=operator_mutants):
                norm = _normalize(mutated)
                if norm is None or norm in seen_norms:
                    continue
                seen_norms.add(norm)
                mutants.append(
                    Mutant(
                        target=target,
                        source=_reindent(mutated, target.indent),
                        origin="operator",
                    )
                )

        if llm is None:
            continue
        cached = cache.cached_mutants(file_key, target.name, f_hash) if cache else None
        if cached is not None:
            fresh = [Mutant(target=target, source=src) for src in cached]
        else:
            fresh = generate_mutants(
                llm,
                original,
                n_per_target=n_per_target,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed + t_idx * 1000,
                targets=[target],
            )
            if cache:
                cache.store_mutants(
                    file_key, target.name, f_hash, [m.source for m in fresh]
                )
        # Drop LLM mutants that duplicate an operator mutant.
        for m in fresh:
            norm = _normalize(_dedent(m.source, target.indent))
            if norm is not None and norm in seen_norms:
                continue
            if norm is not None:
                seen_norms.add(norm)
            mutants.append(m)

    report = Report(file=str(file))
    try:
        for i, mutant in enumerate(mutants):
            t = mutant.target
            f_hash = function_hash(_dedent(t.source, t.indent))
            cached_verdict = (
                cache.cached_verdict(file_key, t.name, f_hash, mutant.source)
                if cache
                else None
            )
            if cached_verdict is not None:
                status = MutantStatus(cached_verdict)
                report.results.append(
                    MutantResult(mutant, status, 0.0, from_cache=True)
                )
                if on_progress:
                    on_progress(i + 1, len(mutants), status)
                continue

            tests_for_mutant = selection.tests_for(t)
            if not tests_for_mutant:
                status = MutantStatus.NO_TESTS
                duration = 0.0
            else:
                file.write_text(apply_mutant(original, mutant))
                t0 = time.time()
                passed, timed_out = _run_pytest(
                    python,
                    pytest_args,
                    cwd,
                    timeout_s,
                    keyword=SelectionMap.keyword_expr(tests_for_mutant),
                )
                duration = time.time() - t0
                file.write_text(original)
                if timed_out:
                    status = MutantStatus.TIMEOUT
                elif passed:
                    status = MutantStatus.SURVIVED
                else:
                    status = MutantStatus.KILLED
            if cache:
                cache.store_verdict(
                    file_key, t.name, f_hash, mutant.source, status.value
                )
            report.results.append(MutantResult(mutant, status, duration))
            if on_progress:
                on_progress(i + 1, len(mutants), status)
    finally:
        file.write_text(original)
        if cache:
            cache.save()
    return report
