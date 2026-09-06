"""Test runners behind one protocol: pytest (Python), vitest/jest (JS/TS).

A runner owns everything about executing tests:
- ``baseline()`` — one clean run that must pass, returning the
  SelectionMap (test ids per function, duration per test);
- ``run()`` — execute a subset (or the whole suite when ``tests`` is
  None) against whatever is on disk right now -> (passed, timed_out).

Node runners select at *file* granularity: ``vitest related`` /
``jest --findRelatedTests`` walk the static import graph, so every
function in the module maps to the same set of test files, and NO_TESTS
fires only when no test imports the module at all. Function-level
selection would need per-test coverage runs — not worth the cost yet.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from supermut.mutate import FunctionTarget
from supermut.selection import SelectionMap, collect_selection

__all__ = [
    "Runner",
    "PytestRunner",
    "VitestRunner",
    "JestRunner",
    "SwiftTestRunner",
    "runner_for",
]


@runtime_checkable
class Runner(Protocol):
    name: str

    def baseline(
        self,
        file: Path,
        cwd: Path,
        targets: list[FunctionTarget],
        timeout_s: float,
    ) -> SelectionMap:
        """One clean instrumented run; raises RuntimeError if the suite fails."""
        ...

    def run(
        self, cwd: Path, tests: list[str] | None, timeout_s: float
    ) -> tuple[bool, bool]:
        """Run ``tests`` (whole suite when None) -> (passed, timed_out)."""
        ...


@dataclass
class PytestRunner:
    """pytest + coverage dynamic contexts (function-level selection)."""

    args: list[str] = field(default_factory=list)
    python: str = sys.executable
    name: str = "pytest"

    def baseline(
        self,
        file: Path,
        cwd: Path,
        targets: list[FunctionTarget],
        timeout_s: float,
    ) -> SelectionMap:
        return collect_selection(
            file, self.args, cwd, python=self.python, timeout_s=timeout_s,
            targets=targets,
        )

    def run(
        self, cwd: Path, tests: list[str] | None, timeout_s: float
    ) -> tuple[bool, bool]:
        cmd = [self.python, "-m", "pytest", *self.args, "-x"]
        if tests:
            cmd += ["-k", SelectionMap.keyword_expr(tests)]
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                timeout=timeout_s,
                # A mutant of identical size restored within the same mtime
                # second would leave its stale .pyc looking valid — the
                # clean run would then import the mutant.
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            return proc.returncode == 0, False
        except subprocess.TimeoutExpired:
            return False, True


def _parse_node_report(payload: dict) -> tuple[bool, dict[str, float]]:
    """(success, test file -> duration_s) from a vitest/jest JSON report.

    Both emit the jest result shape: ``success`` plus ``testResults``
    entries carrying ``name`` (absolute test file path), ``startTime``
    and ``endTime`` in epoch milliseconds.
    """
    durations: dict[str, float] = {}
    for result in payload.get("testResults", []):
        start = result.get("startTime") or 0
        end = result.get("endTime") or start
        durations[result["name"]] = max(end - start, 0) / 1000.0
    return bool(payload.get("success")), durations


class _NodeRunner:
    """Shared file-swap cycle for vitest and jest."""

    command: tuple[str, ...]
    args: list[str]
    name: str

    def _baseline_cmd(self, rel_file: str, out: Path) -> list[str]:
        raise NotImplementedError

    def _run_cmd(self, tests: list[str] | None) -> list[str]:
        raise NotImplementedError

    def baseline(
        self,
        file: Path,
        cwd: Path,
        targets: list[FunctionTarget],
        timeout_s: float,
    ) -> SelectionMap:
        rel = os.path.relpath(file, cwd)
        with tempfile.TemporaryDirectory(prefix="supermut-node-") as tmp:
            out = Path(tmp) / "report.json"
            proc = subprocess.run(
                self._baseline_cmd(rel, out),
                cwd=cwd,
                capture_output=True,
                timeout=timeout_s,
            )
            if not out.exists():
                raise RuntimeError(
                    f"instrumented baseline run produced no {self.name} JSON "
                    "report:\n"
                    + (proc.stderr or proc.stdout).decode(errors="replace")[-2000:]
                )
            success, durations = _parse_node_report(json.loads(out.read_text()))
        if not success:
            raise RuntimeError(
                "instrumented baseline run failed:\n"
                + (proc.stderr or proc.stdout).decode(errors="replace")[-2000:]
            )
        selection = SelectionMap(duration_by_test=durations)
        # File-granular selection: every function maps to every related
        # test file; NO_TESTS only when nothing imports the module.
        for t in targets:
            selection.tests_by_function[t.name] = set(durations)
        return selection

    def run(
        self, cwd: Path, tests: list[str] | None, timeout_s: float
    ) -> tuple[bool, bool]:
        try:
            proc = subprocess.run(
                self._run_cmd(tests),
                cwd=cwd,
                capture_output=True,
                timeout=timeout_s,
            )
            return proc.returncode == 0, False
        except subprocess.TimeoutExpired:
            return False, True


@dataclass
class VitestRunner(_NodeRunner):
    """vitest: `related` for selection, positional file filters per mutant.

    vitest's transform cache (fsModuleCache) is off by default and each
    `--run` invocation is a fresh process, so the file swap needs no
    cache workaround.
    """

    args: list[str] = field(default_factory=list)
    command: tuple[str, ...] = ("npx", "vitest")
    name: str = "vitest"

    def _baseline_cmd(self, rel_file: str, out: Path) -> list[str]:
        return [
            *self.command,
            "related",
            rel_file,
            "--run",
            "--reporter=json",
            f"--outputFile={out}",
            "--passWithNoTests",
            *self.args,
        ]

    def _run_cmd(self, tests: list[str] | None) -> list[str]:
        return [*self.command, "run", *(tests or []), "--bail", "1", *self.args]


@dataclass
class JestRunner(_NodeRunner):
    """jest: `--findRelatedTests` for selection, `--runTestsByPath` per mutant.

    Always ``--no-cache``: jest's docs don't specify how the transform
    cache is keyed, and a stale cached transform of a swapped file would
    silently test the wrong code — the same failure class as Python's
    stale .pyc. Correctness over the ~2x speed cost until verified.
    """

    args: list[str] = field(default_factory=list)
    command: tuple[str, ...] = ("npx", "jest")
    name: str = "jest"

    def _baseline_cmd(self, rel_file: str, out: Path) -> list[str]:
        return [
            *self.command,
            "--findRelatedTests",
            rel_file,
            "--json",
            f"--outputFile={out}",
            "--passWithNoTests",
            "--no-cache",
            *self.args,
        ]

    def _run_cmd(self, tests: list[str] | None) -> list[str]:
        cmd = [*self.command]
        if tests:
            cmd += ["--runTestsByPath", *tests]
        return cmd + ["--bail", "--no-cache", *self.args]


@dataclass
class SwiftTestRunner:
    """SPM: build tests once per schemata, flip mutants via SUPERMUT_MUTANT.

    Baseline runs ``swift test --parallel --xunit-output`` — on this
    toolchain the XCTest xUnit file is only written in parallel mode.
    Per-mutant runs use ``--skip-build`` (a filtered run without it
    rebuilds) with serial execution for clean verdicts. Selection is
    module-granular: SPM has no cheap related-test mapping, so every
    function maps to the whole suite.
    """

    args: list[str] = field(default_factory=list)
    command: tuple[str, ...] = ("swift",)
    name: str = "swift-test"
    schemata: bool = True

    def baseline(
        self,
        file: Path,
        cwd: Path,
        targets: list[FunctionTarget],
        timeout_s: float,
    ) -> SelectionMap:
        with tempfile.TemporaryDirectory(prefix="supermut-swift-") as tmp:
            out = Path(tmp) / "res.xml"
            proc = subprocess.run(
                [
                    *self.command,
                    "test",
                    "--parallel",
                    # space form is load-bearing: `--xunit-output=path` puts
                    # the (empty) swift-testing report at `path` instead of
                    # the XCTest one on this toolchain (Swift 6.3.3)
                    "--xunit-output",
                    str(out),
                    *self.args,
                ],
                cwd=cwd,
                capture_output=True,
                timeout=timeout_s,
            )
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(
                    "instrumented baseline run failed:\n"
                    + (proc.stderr or proc.stdout).decode(errors="replace")[-2000:]
                )
            durations = _parse_xunit(out.read_text())
        selection = SelectionMap(duration_by_test=durations)
        for t in targets:
            selection.tests_by_function[t.name] = set(durations)
        return selection

    def prepare(self, file: Path, cwd: Path, schemata, timeout_s: float) -> None:
        from supermut.schemata import SchemataBuildError

        file.write_text(schemata.source)
        (file.parent / schemata.helper_filename).write_text(schemata.helper_source)
        proc = subprocess.run(
            [*self.command, "build", "--build-tests", *self.args],
            cwd=cwd,
            capture_output=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            raise SchemataBuildError(
                (proc.stderr or proc.stdout).decode(errors="replace")
            )

    def run(
        self,
        cwd: Path,
        tests: list[str] | None,
        timeout_s: float,
        mutant_id: int | None = None,
    ) -> tuple[bool, bool]:
        cmd = [*self.command, "test", "--skip-build"]
        for t in tests or []:
            cmd += ["--filter", t]
        env = dict(os.environ)
        if mutant_id is not None:
            env["SUPERMUT_MUTANT"] = str(mutant_id)
        else:
            env.pop("SUPERMUT_MUTANT", None)
        try:
            proc = subprocess.run(
                cmd + self.args,
                cwd=cwd,
                capture_output=True,
                timeout=timeout_s,
                env=env,
            )
            return proc.returncode == 0, False
        except subprocess.TimeoutExpired:
            return False, True

    def cleanup(self, file: Path, original: str, schemata) -> None:
        file.write_text(original)
        helper = file.parent / schemata.helper_filename
        if helper.exists():
            helper.unlink()


def _parse_xunit(xml_text: str) -> dict[str, float]:
    """test id (``Target.Class/method``, the --filter format) -> seconds."""
    import xml.etree.ElementTree as ET

    durations: dict[str, float] = {}
    for case in ET.fromstring(xml_text).iter("testcase"):
        test_id = f"{case.get('classname')}/{case.get('name')}"
        durations[test_id] = float(case.get("time") or 0.0)
    return durations


def _package_json(cwd: Path) -> dict:
    try:
        return json.loads((cwd / "package.json").read_text())
    except (OSError, ValueError):
        return {}


def runner_for(
    cwd: Path,
    language_name: str,
    tests: str,
    *,
    python: str = sys.executable,
    command: str | None = None,
) -> Runner:
    """Pick a runner for the project at ``cwd``.

    ``tests`` is the runner argument string (pytest args for Python,
    extra CLI args for vitest/jest). ``command`` overrides the runner
    executable (e.g. ``"bunx vitest"``).
    """
    args = shlex.split(tests)
    if language_name == "python":
        return PytestRunner(args=args, python=python)
    if language_name == "swift":
        if not (cwd / "Package.swift").exists():
            raise RuntimeError(
                f"no Package.swift in {cwd}: Swift runs need an SPM package "
                "(pass cwd= / --cwd pointing at the package root)"
            )
        cmd = tuple(shlex.split(command)) if command else ("swift",)
        return SwiftTestRunner(args=args, command=cmd)
    if language_name == "kotlin":
        raise RuntimeError(
            "kotlin test-runner integration is not wired yet (schemata "
            "assembly exists; gradle runner is next)"
        )

    pkg = _package_json(cwd)
    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    cmd = tuple(shlex.split(command)) if command else None
    if list(cwd.glob("vitest.config.*")) or "vitest" in deps:
        return VitestRunner(args=args, command=cmd or ("npx", "vitest"))
    if list(cwd.glob("jest.config.*")) or "jest" in deps or "jest" in pkg:
        return JestRunner(args=args, command=cmd or ("npx", "jest"))
    raise RuntimeError(
        f"no test runner detected in {cwd}: expected vitest or jest in "
        "package.json/config files (pass runner= explicitly to override)"
    )
