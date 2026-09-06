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

import hashlib
import json
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
    _reindent,
    generate_mutants,
)
from supermut.runners import Runner, runner_for
from supermut.selection import SelectionMap

if TYPE_CHECKING:
    from supermut.engine import LLM

__all__ = ["MutantStatus", "MutantResult", "Report", "apply_mutant", "run"]


class MutantStatus(str, Enum):
    KILLED = "killed"
    SURVIVED = "survived"
    TIMEOUT = "timeout"
    NO_TESTS = "no_tests"
    # schemata-only verdicts: the mutant never ran, and must never be
    # conflated with a kill (compile-fail-as-kill inflates scores)
    UNSUPPORTED = "unsupported"
    COMPILE_ERROR = "compile_error"


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
    def unsupported(self) -> int:
        return sum(r.status is MutantStatus.UNSUPPORTED for r in self.results)

    @property
    def compile_errors(self) -> int:
        return sum(r.status is MutantStatus.COMPILE_ERROR for r in self.results)

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
            + (
                f"unsupported: {self.unsupported}  "
                f"compile-errors: {self.compile_errors}  "
                if self.unsupported or self.compile_errors
                else ""
            )
            + f"kill rate: {self.kill_rate:.0%}",
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
    """Replace the target's line span with the mutant source, restoring
    any wrapper prefix (JS/TS ``export``) the bare mutant doesn't carry."""
    lines = module_source.split("\n")
    t = mutant.target
    repl = mutant.source.split("\n")
    if t.prefix:
        repl[0] = repl[0][: len(t.indent)] + t.prefix + repl[0][len(t.indent) :]
    return "\n".join(lines[: t.start_line - 1] + repl + lines[t.end_line :])


def _apply_many(module_source: str, mutants: list[Mutant]) -> str:
    """Apply non-overlapping mutants bottom-up so line spans stay valid."""
    out = module_source
    for m in sorted(mutants, key=lambda m: m.target.start_line, reverse=True):
        out = apply_mutant(out, m)
    return out


# The canary replaces every function body with an unconditional failure.
# Python builds the body by indentation; brace languages splice the
# statement after the first `{`. Expression-bodied functions (no `{`)
# are skipped — enough canaries remain for the gate to mean something.
_CANARY_STMT = {
    "javascript": 'throw new Error("supermut canary");',
    "typescript": 'throw new Error("supermut canary");',
    "kotlin": 'throw RuntimeException("supermut canary")',
    "swift": 'fatalError("supermut canary")',
}


def _canary_mutants(targets: list[FunctionTarget], language) -> list[Mutant]:
    canaries = []
    for t in targets:
        dedented = _dedent(t.source, t.indent)
        if language.name == "python":
            header = dedented.split("\n")[0]
            body = f'{header}\n    raise RuntimeError("supermut canary")'
        else:
            brace = dedented.find("{")
            if brace == -1:
                continue
            stmt = _CANARY_STMT[language.name]
            body = f"{dedented[: brace + 1]}\n    {stmt}\n}}"
        canaries.append(Mutant(target=t, source=_reindent(body, t.indent)))
    return canaries


def _hash_fn_for(language):
    """Cache key function: Python keeps the shipped AST hash (existing
    caches stay valid); other languages hash the tree-sitter normal form."""
    if language.name == "python":
        return function_hash

    def _hash(source: str) -> str:
        norm = language.normalize(source) or source
        return hashlib.sha256(norm.encode()).hexdigest()[:16]

    return _hash


def run(
    file: str | Path,
    tests: str,
    llm: "LLM | None",
    *,
    cwd: str | Path | None = None,
    python: str = sys.executable,
    runner: Runner | None = None,
    runner_cmd: str | None = None,
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
    """Full cycle for one module. ``tests`` is the runner argument string
    (pytest args for Python, e.g. ``"-q tests/"``; extra CLI args for
    vitest/jest). ``llm=None`` runs the operator arm alone (Python only).
    The target file is swapped in place with a guaranteed restore; run
    from a clean working tree.
    """
    from supermut.languages import language_for_path

    file = Path(file).resolve()
    cwd = Path(cwd).resolve() if cwd else file.parent
    original = file.read_text()
    language = language_for_path(str(file))
    if language is None:
        raise ValueError(
            f"no language frontend for {file.name} — install "
            "supermut[languages] for JS/TS/Kotlin/Swift"
        )
    if llm is None and language.name != "python":
        raise ValueError(
            "the operator arm is Python-only; non-Python runs need a model"
        )
    targets = language.find_targets(original)
    if runner is None:
        runner = runner_for(
            cwd, language.name, tests, python=python, command=runner_cmd
        )

    # Gate 1 — instrumented baseline: suite must pass, and we get the
    # function->tests + test->duration maps in the same run.
    selection: SelectionMap = runner.baseline(
        file, cwd, targets, max(timeout_s * 5, 300.0)
    )

    # Gate 2 — canary: all covered targets failing at once must fail
    # their tests.
    covered = [t for t in targets if selection.tests_by_function.get(t.name)]
    # Schemata runners get their canary after the schemata build (a file
    # swap would run against a stale binary here); see _schemata_verdicts.
    canaries = (
        []
        if getattr(runner, "schemata", False)
        else _canary_mutants(covered, language)
    )
    if canaries:
        canary_tests = sorted(
            {test for t in covered for test in selection.tests_by_function[t.name]}
        )
        file.write_text(_apply_many(original, canaries))
        try:
            passed, _ = runner.run(cwd, canary_tests, timeout_s)
        finally:
            file.write_text(original)
        if passed:
            raise RuntimeError(
                "canary mutants did not fail the suite — patches are not "
                "reaching the tests (wrong file? swallowed exceptions?); "
                "verdicts would be meaningless"
            )

    hash_fn = _hash_fn_for(language)
    cache = FunctionCache(cwd / CACHE_NAME, hash_fn=hash_fn) if use_cache else None
    file_key = str(file)

    # Generate (or recall) mutants per target. Hybrid by default: cheap
    # operator mutants (the mutmut-class catalogue) plus LLM naturals —
    # on the bench the union finds holes neither arm finds alone.
    mutants: list[Mutant] = []
    for t_idx, target in enumerate(targets):
        dedented = _dedent(target.source, target.indent)
        f_hash = hash_fn(dedented)
        seen_norms: set[str] = set()

        # The operator catalogue is ast-based; other languages run LLM-only
        # until a tree-sitter operator arm exists.
        if operators and language.name == "python":
            from supermut.synthetic import mutate_function

            for mutated, _op in mutate_function(dedented, max_mutants=operator_mutants):
                norm = language.normalize(mutated)
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
                language=language,
            )
            if cache:
                cache.store_mutants(
                    file_key, target.name, f_hash, [m.source for m in fresh]
                )
        # Drop LLM mutants that duplicate an operator mutant.
        for m in fresh:
            norm = language.normalize(_dedent(m.source, target.indent))
            if norm is not None and norm in seen_norms:
                continue
            if norm is not None:
                seen_norms.add(norm)
            mutants.append(m)

    report = Report(file=str(file))
    if getattr(runner, "schemata", False):
        try:
            _schemata_verdicts(
                report, file, original, cwd, runner, language, mutants,
                selection, timeout_s, cache, file_key, hash_fn, on_progress,
            )
        finally:
            file.write_text(original)
            if cache:
                cache.save()
        return report
    try:
        for i, mutant in enumerate(mutants):
            t = mutant.target
            f_hash = hash_fn(_dedent(t.source, t.indent))
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
                passed, timed_out = runner.run(cwd, tests_for_mutant, timeout_s)
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


def _schemata_verdicts(
    report: Report,
    file: Path,
    original: str,
    cwd: Path,
    runner,
    language,
    mutants: list[Mutant],
    selection: SelectionMap,
    timeout_s: float,
    cache: FunctionCache | None,
    file_key: str,
    hash_fn,
    on_progress,
) -> None:
    """Compiled-language verdict cycle: one build, env-switched runs.

    Every mutant is emitted exactly once — cached, UNSUPPORTED (can't
    live in the schemata), COMPILE_ERROR (its variant broke the build,
    recovered by dropping it and rebuilding, ≤3 attempts), or a real
    run verdict.
    """
    from supermut.schemata import SchemataBuildError, build_schemata, failed_ids

    total = len(mutants)
    done = 0

    def emit(mutant, status, duration=0.0, from_cache=False):
        nonlocal done
        if cache and not from_cache:
            f_hash = hash_fn(_dedent(mutant.target.source, mutant.target.indent))
            cache.store_verdict(
                file_key, mutant.target.name, f_hash, mutant.source, status.value
            )
        report.results.append(MutantResult(mutant, status, duration, from_cache))
        done += 1
        if on_progress:
            on_progress(done, total, status)

    pending: list[Mutant] = []
    for m in mutants:
        f_hash = hash_fn(_dedent(m.target.source, m.target.indent))
        cached_verdict = (
            cache.cached_verdict(file_key, m.target.name, f_hash, m.source)
            if cache
            else None
        )
        if cached_verdict is not None:
            emit(m, MutantStatus(cached_verdict), from_cache=True)
        else:
            pending.append(m)
    if not pending:
        return

    build_timeout = max(timeout_s * 5, 300.0)
    remaining = pending
    schemata = None
    for _ in range(3):
        s = build_schemata(original, remaining, language)
        for m, _reason in s.skipped:
            emit(m, MutantStatus.UNSUPPORTED)
        if not s.by_id:
            return
        try:
            runner.prepare(file, cwd, s, build_timeout)
            schemata = s
            break
        except SchemataBuildError as e:
            bad = failed_ids(s.source, str(e), file.name)
            if not bad:
                raise RuntimeError(
                    "schemata build failed outside every mutant block:\n"
                    + str(e)[-2000:]
                ) from e
            for i in sorted(bad):
                emit(s.by_id[i], MutantStatus.COMPILE_ERROR)
            remaining = [m for i, m in sorted(s.by_id.items()) if i not in bad]
            if not remaining:
                return
    if schemata is None:
        raise RuntimeError("schemata build still failing after 3 recovery attempts")

    try:
        # Canary: the id that fails every dispatcher must turn the
        # (already built) suite red, else the switch isn't reaching tests.
        all_tests = sorted(
            {t for tests in selection.tests_by_function.values() for t in tests}
        )
        if all_tests:
            passed, _ = runner.run(
                cwd, all_tests, timeout_s, mutant_id=schemata.canary_id
            )
            if passed:
                raise RuntimeError(
                    "canary mutants did not fail the suite — the schemata "
                    "switch is not reaching the tests; verdicts would be "
                    "meaningless"
                )

        for mid, m in sorted(schemata.by_id.items()):
            tests = selection.tests_for(m.target)
            if not tests:
                emit(m, MutantStatus.NO_TESTS)
                continue
            t0 = time.time()
            passed, timed_out = runner.run(cwd, tests, timeout_s, mutant_id=mid)
            duration = time.time() - t0
            if timed_out:
                status = MutantStatus.TIMEOUT
            elif passed:
                status = MutantStatus.SURVIVED
            else:
                status = MutantStatus.KILLED
            emit(m, status, duration)
    finally:
        runner.cleanup(file, original, schemata)
