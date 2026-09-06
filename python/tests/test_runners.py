"""Runner protocol tests: node report parsing (fixtures captured from real
vitest 4.x / jest 30.x runs), runner detection, and the language-neutral
harness cycle on a JS module with a fake runner.

The opt-in end-to-end test (SUPERMUT_TEST_NODE=1) builds a real vitest
project with bun and runs the full file-swap cycle.
"""

import json
import os
import shutil
import subprocess
import textwrap

import pytest

from supermut.harness import MutantStatus, _canary_mutants, run
from supermut.runners import (
    JestRunner,
    PytestRunner,
    VitestRunner,
    _parse_node_report,
    runner_for,
)
from supermut.selection import SelectionMap

ts = pytest.importorskip("tree_sitter_javascript", reason="needs [languages] extra")

from supermut.languages import language_by_name  # noqa: E402

CALC_JS = textwrap.dedent(
    """\
    export function add(a, b) {
      return a + b;
    }

    export function clamp(x, lo, hi) {
      if (x < lo) return lo;
      if (x > hi) return hi;
      return x;
    }
    """
)

# Captured from `vitest related src/calc.js --run --reporter=json
# --outputFile=...` (vitest 4.x); assertionResults trimmed.
VITEST_REPORT = json.loads(
    """
    {"numTotalTestSuites":2,"numPassedTestSuites":2,"numFailedTestSuites":0,
     "numTotalTests":2,"numPassedTests":2,"numFailedTests":0,
     "startTime":1788720316338,"success":true,
     "testResults":[{"assertionResults":[],
       "startTime":1788720316466,"endTime":1788720316467.1287,
       "status":"passed","message":"","name":"/private/tmp/smjs/test/calc.test.js"}]}
    """
)

# Captured from `jest --findRelatedTests src/calc.js --json --outputFile=...`
# (jest 30.x); assertionResults trimmed.
JEST_REPORT = json.loads(
    """
    {"numFailedTestSuites":0,"numFailedTests":0,"numPassedTestSuites":1,
     "numPassedTests":1,"numTotalTestSuites":1,"numTotalTests":1,
     "startTime":1788720369842,"success":true,
     "testResults":[{"assertionResults":[],"endTime":1788720370021,
       "message":"","name":"/private/tmp/smjest/test/calc.test.js",
       "startTime":1788720369883,"status":"passed","summary":""}],
     "wasInterrupted":false}
    """
)

NO_RELATED_REPORT = json.loads(
    '{"numTotalTests":0,"success":true,"testResults":[]}'
)


def test_parse_node_report():
    ok, durations = _parse_node_report(VITEST_REPORT)
    assert ok
    assert durations == {
        "/private/tmp/smjs/test/calc.test.js": pytest.approx(0.0011287, abs=1e-4)
    }
    ok, durations = _parse_node_report(JEST_REPORT)
    assert ok
    assert set(durations) == {"/private/tmp/smjest/test/calc.test.js"}
    ok, durations = _parse_node_report(NO_RELATED_REPORT)
    assert ok and durations == {}


def test_runner_for_detection(tmp_path):
    assert isinstance(
        runner_for(tmp_path, "python", "-q tests/"), PytestRunner
    )
    (tmp_path / "package.json").write_text(
        json.dumps({"devDependencies": {"vitest": "^4.0.0"}})
    )
    assert isinstance(runner_for(tmp_path, "javascript", ""), VitestRunner)
    (tmp_path / "package.json").write_text(
        json.dumps({"devDependencies": {"jest": "^30.0.0"}})
    )
    assert isinstance(runner_for(tmp_path, "javascript", ""), JestRunner)
    (tmp_path / "package.json").write_text(json.dumps({"name": "x"}))
    with pytest.raises(RuntimeError, match="no test runner"):
        runner_for(tmp_path, "javascript", "")


def test_runner_for_command_override(tmp_path):
    (tmp_path / "package.json").write_text(
        json.dumps({"devDependencies": {"vitest": "^4.0.0"}})
    )
    r = runner_for(tmp_path, "javascript", "", command="bunx vitest")
    assert r.command == ("bunx", "vitest")


def test_node_runner_commands(tmp_path):
    v = VitestRunner()
    assert v._baseline_cmd("src/calc.js", tmp_path / "r.json")[:3] == [
        "npx",
        "vitest",
        "related",
    ]
    assert v._run_cmd(["/t/a.test.js"]) == [
        "npx", "vitest", "run", "/t/a.test.js", "--bail", "1",
    ]
    j = JestRunner()
    assert "--findRelatedTests" in j._baseline_cmd("src/calc.js", tmp_path / "r")
    assert "--no-cache" in j._run_cmd(None)
    assert j._run_cmd(["/t/a.test.js"])[:3] == ["npx", "jest", "--runTestsByPath"]


def test_canary_mutants_js():
    js = language_by_name("javascript")
    targets = js.find_targets(CALC_JS)
    canaries = _canary_mutants(targets, js)
    assert len(canaries) == len(targets)
    for c in canaries:
        assert 'throw new Error("supermut canary")' in c.source
        # each canary still parses as a single function of the same name
        assert js.validate(c.source, c.target.name)


def test_export_prefix_survives_apply(tmp_path):
    """Mutating an `export function` must not drop the export keyword."""
    from supermut.harness import apply_mutant
    from supermut.mutate import Mutant

    js = language_by_name("javascript")
    targets = js.find_targets(CALC_JS)
    add = next(t for t in targets if t.name == "add")
    assert add.prefix == "export "
    assert add.source.startswith("function add")
    mutated = apply_mutant(
        CALC_JS,
        Mutant(target=add, source="function add(a, b) {\n  return a - b;\n}"),
    )
    assert "export function add(a, b) {" in mutated
    assert "return a - b;" in mutated
    assert "export function clamp" in mutated  # neighbours untouched


class _StubLLM:
    """Returns the same canned completions for every target."""

    def __init__(self, completions):
        self._completions = completions

    def generate_batch(self, prefix, continuations, **kwargs):
        return self._completions[: len(continuations)]


class _FakeRunner:
    """Selection from a fixed map; run() verdict from what's on disk.

    Fails whenever the watched file differs from its baseline content —
    i.e. every canary and every mutant is 'killed'. With kill=False it
    fails only canaries (mutants all survive).
    """

    name = "fake"

    def __init__(self, file, tests_by_function, kill=True):
        self._file = file
        self._tests = tests_by_function
        self._kill = kill
        self.subset_calls = []

    def baseline(self, file, cwd, targets, timeout_s):
        self._baseline = self._file.read_text()
        return SelectionMap(
            tests_by_function={
                t.name: set(self._tests.get(t.name, ())) for t in targets
            },
            duration_by_test={},
        )

    def run(self, cwd, tests, timeout_s):
        self.subset_calls.append(tests)
        content = self._file.read_text()
        if "supermut canary" in content:
            return False, False
        if self._kill and content != self._baseline:
            return False, False
        return True, False


def _write_js_module(tmp_path):
    target = tmp_path / "calc.js"
    target.write_text(CALC_JS)
    return target


def test_js_cycle_kills_with_fake_runner(tmp_path):
    target = _write_js_module(tmp_path)
    fake = _FakeRunner(
        target,
        {"add": {"t/calc.test.js"}, "clamp": {"t/calc.test.js"}},
    )
    llm = _StubLLM(["  return a - b;\n}", "  return a * b;\n}"])
    report = run(
        target, "", llm, cwd=tmp_path, runner=fake, n_per_target=2, use_cache=False
    )
    assert target.read_text() == CALC_JS  # restored byte-for-byte
    assert report.results and all(
        r.status is MutantStatus.KILLED for r in report.results
    )
    # mutant runs got the selected test subset, not the whole suite
    assert all(calls == ["t/calc.test.js"] for calls in fake.subset_calls)


def test_js_cycle_survivors_and_no_tests(tmp_path):
    target = _write_js_module(tmp_path)
    # clamp has no tests at all -> its mutants are NO_TESTS, never run
    fake = _FakeRunner(target, {"add": {"t/calc.test.js"}}, kill=False)
    llm = _StubLLM(["  return a - b;\n}"])
    report = run(
        target, "", llm, cwd=tmp_path, runner=fake, n_per_target=1, use_cache=False
    )
    by_name = {}
    for r in report.results:
        by_name.setdefault(r.mutant.target.name, set()).add(r.status)
    assert by_name["add"] == {MutantStatus.SURVIVED}
    assert by_name["clamp"] == {MutantStatus.NO_TESTS}


def test_js_requires_model(tmp_path):
    target = _write_js_module(tmp_path)
    with pytest.raises(ValueError, match="Python-only"):
        run(target, "", None, cwd=tmp_path)


needs_node = pytest.mark.skipif(
    not (os.environ.get("SUPERMUT_TEST_NODE") and shutil.which("bun")),
    reason="set SUPERMUT_TEST_NODE=1 (needs bun + network for vitest install)",
)


@needs_node
def test_vitest_full_cycle(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "test").mkdir()
    (tmp_path / "package.json").write_text(
        json.dumps({"name": "fixture", "private": True, "type": "module"})
    )
    (tmp_path / "src" / "calc.js").write_text(CALC_JS)
    (tmp_path / "test" / "calc.test.js").write_text(
        textwrap.dedent(
            """\
            import { it, expect } from 'vitest';
            import { add, clamp } from '../src/calc.js';

            it('adds', () => { expect(add(2, 3)).toBe(5); });
            it('clamps', () => {
              expect(clamp(5, 0, 10)).toBe(5);
              expect(clamp(-1, 0, 10)).toBe(0);
              expect(clamp(11, 0, 10)).toBe(10);
            });
            """
        )
    )
    subprocess.run(
        ["bun", "add", "-d", "vitest"], cwd=tmp_path, check=True, capture_output=True
    )
    llm = _StubLLM(["  return a - b;\n}"])
    report = run(
        (tmp_path / "src" / "calc.js"),
        "",
        llm,
        cwd=tmp_path,
        runner_cmd="bunx vitest",
        n_per_target=1,
        use_cache=False,
        timeout_s=120,
    )
    assert (tmp_path / "src" / "calc.js").read_text() == CALC_JS
    statuses = {r.mutant.target.name: r.status for r in report.results}
    # add's `a - b` mutant breaks a real assertion
    assert statuses["add"] is MutantStatus.KILLED
