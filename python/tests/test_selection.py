"""Tests for test selection, the function cache, and harness gate logic —
all model-free (StubLLM)."""

import sys
import textwrap
from pathlib import Path

import pytest

from supermut.cache import CACHE_NAME, FunctionCache, function_hash
from supermut.harness import MutantStatus, run
from supermut.mutate import find_targets
from supermut.selection import SelectionMap, collect_selection

CALC = textwrap.dedent(
    """
    def add(a, b):
        return a + b


    def orphan(n):
        return n % 2 == 0
    """
).lstrip()

TEST_CALC = textwrap.dedent(
    """
    from calc import add

    def test_add():
        assert add(2, 3) == 5
    """
).lstrip()


@pytest.fixture
def bed(tmp_path):
    (tmp_path / "calc.py").write_text(CALC)
    (tmp_path / "test_calc.py").write_text(TEST_CALC)
    return tmp_path


def test_collect_selection(bed):
    sel = collect_selection(
        bed / "calc.py", ["-q", "test_calc.py"], bed, python=sys.executable
    )
    assert sel.tests_by_function["add"] == {"test_calc.test_add"}
    assert sel.tests_by_function["orphan"] == set()
    assert sel.duration_by_test["test_calc.test_add"] >= 0.0


def test_selection_ordering_and_keyword():
    sel = SelectionMap(
        tests_by_function={"f": {"m.slow", "m.fast"}},
        duration_by_test={"m.slow": 2.0, "m.fast": 0.1},
    )
    targets = find_targets("def f():\n    return 1\n")
    tests = sel.tests_for(targets[0])
    assert tests == ["m.fast", "m.slow"]
    assert SelectionMap.keyword_expr(tests) == "fast or slow"


def test_collect_selection_failing_suite(bed):
    (bed / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 6\n"
    )
    with pytest.raises(RuntimeError, match="baseline"):
        collect_selection(
            bed / "calc.py", ["-q", "test_calc.py"], bed, python=sys.executable
        )


def test_function_cache_roundtrip(tmp_path):
    path = tmp_path / CACHE_NAME
    h = function_hash("def f():\n    return 1")
    cache = FunctionCache(path)
    assert cache.cached_mutants("calc.py", "f", h) is None
    cache.store_mutants("calc.py", "f", h, ["def f():\n    return 2"])
    cache.store_verdict("calc.py", "f", h, "def f():\n    return 2", "killed")
    cache.save()

    reloaded = FunctionCache(path)
    assert reloaded.cached_mutants("calc.py", "f", h) == ["def f():\n    return 2"]
    assert (
        reloaded.cached_verdict("calc.py", "f", h, "def f():\n    return 2")
        == "killed"
    )
    # Changed function hash invalidates everything for that function.
    h2 = function_hash("def f():\n    return 99")
    assert reloaded.cached_mutants("calc.py", "f", h2) is None


def test_function_hash_format_insensitive():
    assert function_hash("def f():\n    return 1") == function_hash(
        "def f():  # comment\n    return 1"
    )
    assert function_hash("def f():\n    return 1") != function_hash(
        "def f():\n    return 2"
    )


class _StubLLM:
    """Emits a subtly-wrong body for any prompted function."""

    def generate_batch(self, prefix, continuations, **kwargs):
        if "def add" in prefix:
            return ["    return a - b"] * len(continuations)
        return ["    return n % 2 == 1"] * len(continuations)


def test_run_operators_only(bed):
    """No model at all: the operator arm works standalone."""
    report = run(
        bed / "calc.py",
        "-q test_calc.py",
        None,
        cwd=bed,
        python=sys.executable,
        timeout_s=60,
    )
    assert report.results, "operator arm produced no mutants"
    assert all(r.mutant.origin == "operator" for r in report.results)
    add_results = [r for r in report.results if r.mutant.target.name == "add"]
    assert add_results and all(
        r.status is MutantStatus.KILLED for r in add_results
    )
    assert "operator:" in report.summary()


def test_run_stub_full_mechanics(bed):
    """Model-free end-to-end: gates, selection, no-tests, cache."""
    report = run(
        bed / "calc.py",
        "-q test_calc.py",
        _StubLLM(),
        cwd=bed,
        python=sys.executable,
        n_per_target=1,
        timeout_s=60,
        operators=False,
    )
    by_target = {r.mutant.target.name: r for r in report.results}
    assert by_target["add"].status is MutantStatus.KILLED
    assert by_target["orphan"].status is MutantStatus.NO_TESTS
    assert report.no_tests == 1
    assert report.kill_rate == 1.0  # no-tests excluded from the rate
    assert (bed / "calc.py").read_text() == CALC
    assert '"no_tests": 1' in report.to_json()

    # Second run: everything comes from cache, no test subprocesses needed.
    report2 = run(
        bed / "calc.py",
        "-q test_calc.py",
        _StubLLM(),
        cwd=bed,
        python=sys.executable,
        n_per_target=1,
        timeout_s=60,
        operators=False,
    )
    assert all(r.from_cache for r in report2.results if r.status is MutantStatus.KILLED)
    assert Path(bed / CACHE_NAME).exists()
