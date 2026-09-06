"""Harness tests: parsing/apply/validate without a model; full cycle with one."""

import sys
import textwrap

import pytest

from supermut.harness import MutantStatus, apply_mutant, run
from supermut.mutate import (
    Mutant,
    _is_single_function,
    _trim_to_function,
    build_prompt,
    find_targets,
)
from .test_all import MODEL, needs_model

CALC = textwrap.dedent(
    '''
    """Demo module."""


    def add(a, b):
        return a + b


    def clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x


    class Box:
        def __init__(self, size):
            self.size = size

        def fits(self, item):
            return item <= self.size
    '''
).lstrip()


def test_find_targets():
    targets = find_targets(CALC)
    assert [t.name for t in targets] == ["add", "clamp", "__init__", "fits"]
    clamp = targets[1]
    assert clamp.source.startswith("def clamp")
    fits = targets[3]
    assert fits.indent == "    "
    assert fits.source.startswith("    def fits")


def test_find_targets_skips_nested():
    src = "def outer():\n    def inner():\n        return 1\n    return inner()\n"
    assert [t.name for t in find_targets(src)] == ["outer"]


def test_build_prompt_dedents_methods():
    fits = find_targets(CALC)[3]
    prompt = build_prompt(fits)
    assert "\ndef fits(self, item):\n" in prompt
    assert prompt.endswith("def fits(self, item):\n")


def test_apply_mutant_roundtrip():
    targets = find_targets(CALC)
    clamp = targets[1]
    mutant = Mutant(target=clamp, source="def clamp(x, lo, hi):\n    return x")
    mutated = apply_mutant(CALC, mutant)
    assert "return lo" not in mutated
    assert "def add" in mutated and "def fits" in mutated
    # The mutated module must still parse, with the same top-level names.
    assert [t.name for t in find_targets(mutated)] == [
        "add",
        "clamp",
        "__init__",
        "fits",
    ]


def test_apply_mutant_method_keeps_indent():
    fits = find_targets(CALC)[3]
    mutant = Mutant(
        target=fits,
        source="    def fits(self, item):\n        return item < self.size",
    )
    mutated = apply_mutant(CALC, mutant)
    find_targets(mutated)  # parses
    assert "        return item < self.size" in mutated


def test_trim_to_function():
    good = "def f():\n    return 1"
    assert _trim_to_function(good) == good
    trailing_junk = "def f():\n    return 1\nthis is not python ((("
    assert _trim_to_function(trailing_junk) == good
    hopeless = "((broken\n" * 6
    assert _trim_to_function(hopeless) is None


def test_is_single_function():
    assert _is_single_function("def f():\n    return 1", "f")
    assert not _is_single_function("def g():\n    return 1", "f")
    assert not _is_single_function("def f():\n    return 1\nx = 2", "f")
    assert not _is_single_function("x = 1", "f")


class _StubLLM:
    """Returns canned completions instead of running a model."""

    def __init__(self, completions):
        self._completions = completions

    def generate_batch(self, prefix, continuations, **kwargs):
        return self._completions[: len(continuations)]


def test_generate_mutants_filters_and_dedupes():
    from supermut.mutate import generate_mutants

    src = "def add(a, b):\n    return a + b\n"
    completions = [
        "    return a + b",  # exact copy of the original — dropped
        "    return a  +  b  # spacing",  # AST-identical to original — dropped
        "    return a - b",  # real mutant — kept
        "    return a - b",  # duplicate mutant — dropped
        "    this is not python (((",  # unparseable — dropped
    ]
    mutants = generate_mutants(_StubLLM(completions), src, n_per_target=5)
    assert len(mutants) == 1
    assert "a - b" in mutants[0].source
    assert mutants[0].target.name == "add"


@needs_model
def test_full_cycle(tmp_path):
    import supermut

    target = tmp_path / "calc.py"
    target.write_text(CALC)
    (tmp_path / "test_calc.py").write_text(
        textwrap.dedent(
            """
            from calc import add, clamp

            def test_add():
                assert add(2, 3) == 5

            def test_clamp():
                assert clamp(5, 0, 10) == 5
                assert clamp(-1, 0, 10) == 0
                assert clamp(11, 0, 10) == 10
            """
        )
    )

    llm = supermut.LLM(MODEL, n_ctx=2048)
    report = run(
        target,
        f"{sys.executable} -m pytest -x -q test_calc.py",
        llm,
        cwd=tmp_path,
        n_per_target=4,
        max_tokens=96,
        seed=3,
        timeout_s=60,
    )
    # File restored byte-for-byte no matter what the mutants did.
    assert target.read_text() == CALC
    # Every result is a real verdict and the counts add up.
    assert all(
        r.status in (MutantStatus.KILLED, MutantStatus.SURVIVED, MutantStatus.TIMEOUT)
        for r in report.results
    )
    assert report.killed + report.survived == len(report.results) or any(
        r.status is MutantStatus.TIMEOUT for r in report.results
    )
    assert "kill rate" in report.summary()


@needs_model
def test_baseline_failure_raises(tmp_path):
    import supermut

    target = tmp_path / "calc.py"
    target.write_text(CALC)
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 6\n"
    )
    llm = supermut.LLM(MODEL, n_ctx=2048)
    with pytest.raises(RuntimeError, match="baseline"):
        run(
            target,
            f"{sys.executable} -m pytest -x -q test_calc.py",
            llm,
            cwd=tmp_path,
            n_per_target=1,
        )
