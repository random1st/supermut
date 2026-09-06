"""Mutation-test harness: apply each mutant, run the test suite, score it.

Kill rate = killed / (killed + survived). Timeouts count as killed (the
suite did notice something is wrong); invalid mutants are excluded.
"""

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from supermut.mutate import Mutant, generate_mutants

if TYPE_CHECKING:
    from supermut.engine import LLM

__all__ = ["MutantStatus", "MutantResult", "Report", "apply_mutant", "run"]


class MutantStatus(str, Enum):
    KILLED = "killed"
    SURVIVED = "survived"
    TIMEOUT = "timeout"
    BASELINE_BROKEN = "baseline_broken"  # tests already fail on the original


@dataclass
class MutantResult:
    mutant: Mutant
    status: MutantStatus
    duration_s: float


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
    def kill_rate(self) -> float:
        scored = self.killed + self.survived
        return self.killed / scored if scored else 0.0

    def summary(self) -> str:
        lines = [
            f"file: {self.file}",
            f"mutants run: {len(self.results)}  killed: {self.killed}  "
            f"survived: {self.survived}  kill rate: {self.kill_rate:.0%}",
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
                    f"  SURVIVED {r.mutant.target.name}: {first[:80]}"
                )
        return "\n".join(lines)


def apply_mutant(module_source: str, mutant: Mutant) -> str:
    """Replace the target's line span with the mutant source."""
    lines = module_source.split("\n")
    t = mutant.target
    return "\n".join(
        lines[: t.start_line - 1] + mutant.source.split("\n") + lines[t.end_line :]
    )


def _run_tests(cmd: list[str], cwd: Path, timeout_s: float) -> tuple[bool, bool]:
    """Returns (passed, timed_out)."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            timeout=timeout_s,
        )
        return proc.returncode == 0, False
    except subprocess.TimeoutExpired:
        return False, True


def run(
    file: str | Path,
    tests: str,
    llm: "LLM",
    *,
    cwd: str | Path | None = None,
    n_per_target: int = 8,
    max_tokens: int = 192,
    temperature: float = 0.9,
    seed: int = 42,
    timeout_s: float = 60.0,
    on_progress=None,
) -> Report:
    """Full cycle for one module: generate mutants, run tests against each.

    The target file is swapped in place (with a guaranteed restore); run
    from a clean working tree.
    """
    file = Path(file).resolve()
    cwd = Path(cwd).resolve() if cwd else file.parent
    cmd = shlex.split(tests)
    original = file.read_text()

    # Baseline: the suite must pass on unmutated code, else scoring is noise.
    passed, timed_out = _run_tests(cmd, cwd, timeout_s)
    if not passed:
        raise RuntimeError(
            "test suite fails on the original file"
            + (" (timeout)" if timed_out else "")
            + " — fix the baseline before mutation testing"
        )

    mutants = generate_mutants(
        llm,
        original,
        n_per_target=n_per_target,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )

    report = Report(file=str(file))
    try:
        for i, mutant in enumerate(mutants):
            mutated = apply_mutant(original, mutant)
            file.write_text(mutated)
            t0 = time.time()
            passed, timed_out = _run_tests(cmd, cwd, timeout_s)
            duration = time.time() - t0
            if timed_out:
                status = MutantStatus.TIMEOUT
            elif passed:
                status = MutantStatus.SURVIVED
            else:
                status = MutantStatus.KILLED
            report.results.append(
                MutantResult(mutant=mutant, status=status, duration_s=duration)
            )
            if on_progress:
                on_progress(i + 1, len(mutants), status)
    finally:
        file.write_text(original)
    return report
