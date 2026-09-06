"""Synthetic operator-mutation pair tests."""

import ast

from supermut.synthetic import (
    mutate_function,
    pairs_from_module,
    to_prompt_completion,
)

CLAMP = "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x"


def test_mutate_function_boundary_flips():
    mutants = mutate_function(CLAMP, max_mutants=6)
    sources = [m for m, _ in mutants]
    ops = [op for _, op in mutants]
    assert any("x <= lo" in s for s in sources)
    assert any("x >= hi" in s for s in sources)
    assert all(op.startswith("cmp:") for op in ops[:2])
    # every mutant parses and differs from the original
    orig_norm = ast.dump(ast.parse(CLAMP), annotate_fields=False)
    for s in sources:
        assert ast.dump(ast.parse(s), annotate_fields=False) != orig_norm


def test_mutate_function_each_mutant_single_change():
    mutants = mutate_function(CLAMP, max_mutants=6)
    for s, _ in mutants:
        diff = sum(
            a != b for a, b in zip(CLAMP.split("\n"), s.split("\n"))
        )
        assert diff == 1, s


def test_mutate_arith_bool_const_not():
    assert any(
        "a - b" in s for s, _ in mutate_function("def f(a, b):\n    return a + b")
    )
    assert any(
        " or " in s
        for s, _ in mutate_function("def f(a, b):\n    return a and b")
    )
    assert any(
        "return 2" in s for s, _ in mutate_function("def f():\n    return 1")
    )
    assert any(
        "return x" in s
        for s, _ in mutate_function("def f(x):\n    return not x")
    )


def test_mutate_unparseable_returns_empty():
    assert mutate_function("this is not python (((") == []


def test_pairs_from_module_and_format():
    module = CLAMP + "\n\n\ndef ok():\n    return True\n"
    pairs = pairs_from_module(module, per_function=2)
    names = {p.function for p in pairs}
    assert "clamp" in names and "ok" in names
    pc = to_prompt_completion(pairs[0])
    assert pc["prompt"].startswith("# Original function:")
    assert pc["completion"]  # non-empty body
    # prompt side carries the original, completion the mutant
    assert "x < lo" in pc["prompt"]
