"""tree-sitter synthetic operator pairs — all four `//` languages."""

import pytest

pytest.importorskip("tree_sitter", reason="languages extra not installed")

from supermut.synthetic_ts import mutate_function, pairs_from_records, to_prompt_completion

CLAMP = {
    "javascript": "function clamp(x, lo, hi) {\n  if (x < lo) return lo;\n  if (x > hi) return hi;\n  return x;\n}",
    "typescript": "function clamp(x: number, lo: number, hi: number): number {\n  if (x < lo) return lo;\n  if (x > hi) return hi;\n  return x;\n}",
    "kotlin": "fun clamp(x: Int, lo: Int, hi: Int): Int {\n    if (x < lo) return lo\n    if (x > hi) return hi\n    return x\n}",
    "swift": "func clamp(_ x: Int, _ lo: Int, _ hi: Int) -> Int {\n    if x < lo { return lo }\n    if x > hi { return hi }\n    return x\n}",
}


@pytest.mark.parametrize("lang", list(CLAMP))
def test_boundary_flips_all_languages(lang):
    mutants = mutate_function(CLAMP[lang], lang, max_mutants=8)
    sources = [m for m, _ in mutants]
    labels = [op for _, op in mutants]
    assert any("x <= lo" in s for s in sources), (lang, sources)
    assert any("x >= hi" in s for s in sources), (lang, sources)
    assert all(lab.startswith("op:") for lab in labels[:2])
    # every mutant differs from the original and from each other
    assert len(set(sources)) == len(sources)
    assert CLAMP[lang] not in sources


@pytest.mark.parametrize("lang", list(CLAMP))
def test_single_site_per_mutant(lang):
    for mutated, _ in mutate_function(CLAMP[lang], lang, max_mutants=8):
        diff = sum(a != b for a, b in zip(CLAMP[lang].split("\n"), mutated.split("\n")))
        assert diff == 1, mutated


def test_arith_bool_const_unary_js():
    ms = mutate_function("function f(a, b) { return a + b; }", "javascript")
    assert any("a - b" in m for m, _ in ms)
    ms = mutate_function("function f(a, b) { return a && b; }", "javascript")
    assert any("a || b" in m for m, _ in ms)
    ms = mutate_function("function f() { return 1; }", "javascript")
    assert any("return 2" in m for m, _ in ms)
    ms = mutate_function("function f(x) { return !x; }", "javascript")
    assert any("return x" in m for m, _ in ms)
    ms = mutate_function("function f() { return true; }", "javascript")
    assert any("return false" in m for m, _ in ms)


def test_strict_equality_and_unary_kotlin():
    ms = mutate_function("function f(a, b) { return a === b; }", "javascript")
    assert any("a !== b" in m for m, _ in ms)
    ms = mutate_function("fun f(x: Boolean): Boolean { return !x }", "kotlin")
    assert any("return x" in m for m, _ in ms)


def test_swift_specialised_nodes():
    ms = mutate_function("func f(_ a: Int, _ b: Int) -> Int { return a + b }", "swift")
    assert any("a - b" in m for m, _ in ms)
    ms = mutate_function("func f(_ a: Bool, _ b: Bool) -> Bool { return a && b }", "swift")
    assert any("a || b" in m for m, _ in ms)


def test_unparseable_and_multi_function_rejected():
    assert mutate_function("function ((( broken", "javascript") == []
    two = "function a() { return 1; }\nfunction b() { return 2; }"
    assert mutate_function(two, "javascript") == []


def test_pairs_from_records_and_prompt_format():
    records = [
        {"language": "kotlin", "function": "clamp", "fixed_source": CLAMP["kotlin"]},
        {"language": "python", "function": "p", "fixed_source": "def p():\n    return 1"},
        {"language": "swift", "function": "clamp", "fixed_source": CLAMP["swift"]},
        {"language": "swift", "function": "clamp", "fixed_source": CLAMP["swift"]},  # dup
    ]
    pairs = pairs_from_records(records, per_function=2)
    langs = [p.language for p in pairs]
    assert "python" not in langs
    assert langs.count("kotlin") == 2 and langs.count("swift") == 2
    pc = to_prompt_completion(pairs[0])
    assert pc["prompt"].startswith("// Original function:")
    assert "// Buggy mutant" in pc["prompt"]
    assert pc["prompt"].endswith("fun clamp(x: Int, lo: Int, hi: Int): Int {\n")
    assert "x <= lo" in pc["completion"] or "x >= hi" in pc["completion"]
