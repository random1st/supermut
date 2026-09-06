"""Test selection: one instrumented run maps functions to the tests that
execute them, plus per-test durations.

Borrowed from mutmut's design (its "stats run"), implemented with coverage.py
dynamic contexts instead of trampolines: line -> test contexts, folded onto
the module's function spans. A mutant then runs only the tests that touch its
function (cheapest first), and a mutant whose function no test executes is
reported as NO_TESTS without running anything.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from supermut.mutate import FunctionTarget

__all__ = ["SelectionMap", "collect_selection"]


@dataclass
class SelectionMap:
    """function name -> test contexts, and test context -> duration."""

    tests_by_function: dict[str, set[str]] = field(default_factory=dict)
    duration_by_test: dict[str, float] = field(default_factory=dict)

    def tests_for(self, target: FunctionTarget) -> list[str]:
        """Tests touching the target, cheapest first."""
        tests = self.tests_by_function.get(target.name, set())
        return sorted(tests, key=lambda t: self.duration_by_test.get(t, 0.0))

    @staticmethod
    def keyword_expr(tests: list[str]) -> str:
        """pytest -k expression selecting these tests by function name.

        Matches all parametrizations of each test function; good enough
        until node-id-level selection is needed.
        """
        names = sorted({t.rpartition(".")[2] for t in tests})
        return " or ".join(names)


def collect_selection(
    file: Path,
    pytest_args: list[str],
    cwd: Path,
    *,
    python: str = sys.executable,
    timeout_s: float = 300.0,
    targets: list[FunctionTarget] | None = None,
) -> SelectionMap:
    """Run the suite once under coverage dynamic contexts + junitxml.

    Returns the map for *file*'s functions. Raises if the suite fails —
    the caller's baseline contract, verified here for free.
    """
    from supermut.mutate import find_targets

    if targets is None:
        targets = find_targets(file.read_text())

    with tempfile.TemporaryDirectory(prefix="supermut-sel-") as tmp:
        tmpdir = Path(tmp)
        rcfile = tmpdir / ".coveragerc"
        datafile = tmpdir / ".coverage"
        junit = tmpdir / "junit.xml"
        rcfile.write_text(
            f"[run]\ndynamic_context = test_function\ndata_file = {datafile}\n"
        )
        proc = subprocess.run(
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={rcfile}",
                "-m",
                "pytest",
                *pytest_args,
                f"--junitxml={junit}",
            ],
            cwd=cwd,
            capture_output=True,
            timeout=timeout_s,
            # See harness._run_pytest: stale-bytecode protection.
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "instrumented baseline run failed:\n"
                + proc.stdout.decode(errors="replace")[-2000:]
            )

        from coverage import CoverageData

        data = CoverageData(basename=str(datafile))
        data.read()
        resolved = str(file.resolve())
        contexts_by_line: dict[int, list[str]] = {}
        for measured in data.measured_files():
            if str(Path(measured).resolve()) == resolved:
                contexts_by_line = data.contexts_by_lineno(measured)
                break

        selection = SelectionMap()
        for target in targets:
            tests: set[str] = set()
            for line in range(target.start_line, target.end_line + 1):
                for ctx in contexts_by_line.get(line, []):
                    if ctx:
                        tests.add(ctx)
            selection.tests_by_function[target.name] = tests

        for case in ET.parse(junit).getroot().iter("testcase"):
            ctx = f"{case.get('classname')}.{case.get('name')}"
            selection.duration_by_test[ctx] = float(case.get("time") or 0.0)
        return selection
