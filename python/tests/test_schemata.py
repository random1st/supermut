"""Schemata assembly tests: structure via tree-sitter, then real compiler
smoke tests (swiftc -typecheck / kotlinc) and a runtime switch check —
tree-sitter parsing alone is not a typecheck.
"""

import shutil
import subprocess
import textwrap

import pytest

pytest.importorskip("tree_sitter_kotlin", reason="needs [languages] extra")

from supermut.languages import language_by_name
from supermut.mutate import Mutant
from supermut.schemata import build_schemata

KT = language_by_name("kotlin")
SW = language_by_name("swift")

KOTLIN_SRC = textwrap.dedent(
    """\
    package demo

    fun add(a: Int, b: Int): Int {
        return a + b
    }

    fun double(x: Int) = x * 2

    inline fun frozen(x: Int): Int {
        return x
    }
    """
)

SWIFT_SRC = textwrap.dedent(
    """\
    func add(_ a: Int, _ b: Int) -> Int {
        return a + b
    }

    func clamp(value: Int, lo: Int, hi: Int) -> Int {
        if value < lo { return lo }
        if value > hi { return hi }
        return value
    }
    """
)


def _mutants(language, src, replacements):
    """[(target_name, mutant_source)] -> Mutant list bound to real targets."""
    targets = {t.name: t for t in language.find_targets(src)}
    return [
        Mutant(target=targets[name], source=source)
        for name, source in replacements
    ]


def test_kotlin_schemata_structure():
    mutants = _mutants(
        KT,
        KOTLIN_SRC,
        [
            ("add", "fun add(a: Int, b: Int): Int {\n    return a - b\n}"),
            ("double", "fun double(x: Int) = x * 3"),
            ("frozen", "inline fun frozen(x: Int): Int {\n    return x + 1\n}"),
        ],
    )
    s = build_schemata(KOTLIN_SRC, mutants, KT)
    # inline target is skipped with a reason, not silently dropped
    assert len(s.by_id) == 2 and len(s.skipped) == 1
    assert "inline" in s.skipped[0][1]
    assert s.canary_id == len(mutants) + 1
    # dispatcher + variants for both shapes (block and expression body)
    assert "when (__Supermut.active)" in s.source
    assert "private fun add__sm_orig(a: Int, b: Int)" in s.source
    assert "add__sm_1(a, b)" in s.source
    assert "fun double(x: Int) = when (__Supermut.active)" in s.source
    assert f'{s.canary_id} -> throw RuntimeException("supermut canary")' in s.source
    # the assembled module still parses
    assert KT.normalize(s.source) is not None
    # helper carries the module's package
    assert s.helper_source.startswith("package demo")


def test_swift_schemata_structure():
    mutants = _mutants(
        SW,
        SWIFT_SRC,
        [
            ("add", "func add(_ a: Int, _ b: Int) -> Int {\n    return a - b\n}"),
            (
                "clamp",
                "func clamp(value: Int, lo: Int, hi: Int) -> Int {\n"
                "    return value\n}",
            ),
        ],
    )
    s = build_schemata(SWIFT_SRC, mutants, SW)
    assert len(s.by_id) == 2 and not s.skipped
    # nested variants + forwarding with labels; positional for `_`
    assert "func __sm_1(_ a: Int, _ b: Int) -> Int" in s.source
    assert "return __sm_1(a, b)" in s.source
    assert "return __sm_2(value: value, lo: lo, hi: hi)" in s.source
    assert f'case {s.canary_id}: fatalError("supermut canary")' in s.source
    assert "default: break" in s.source
    assert SW.normalize(s.source) is not None


def test_swift_skips_mutating_and_variadic():
    src = textwrap.dedent(
        """\
        struct Box {
            var n: Int
            mutating func bump() {
                n += 1
            }
        }

        func total(_ xs: Int...) -> Int {
            return xs.reduce(0, +)
        }
        """
    )
    mutants = _mutants(
        SW,
        src,
        [
            ("bump", "mutating func bump() {\n    n += 2\n}"),
            ("total", "func total(_ xs: Int...) -> Int {\n    return 0\n}"),
        ],
    )
    s = build_schemata(src, mutants, SW)
    assert not s.by_id
    reasons = sorted(reason for _, reason in s.skipped)
    assert any("mutating" in r for r in reasons)
    assert any("variadic" in r for r in reasons)
    # untouched module comes back unchanged
    assert s.source == src


needs_swiftc = pytest.mark.skipif(
    shutil.which("swiftc") is None, reason="swiftc not installed"
)
needs_kotlinc = pytest.mark.skipif(
    shutil.which("kotlinc") is None, reason="kotlinc not installed"
)


@needs_swiftc
def test_swift_schemata_typechecks_and_switches(tmp_path):
    mutants = _mutants(
        SW,
        SWIFT_SRC,
        [("add", "func add(_ a: Int, _ b: Int) -> Int {\n    return a - b\n}")],
    )
    s = build_schemata(SWIFT_SRC, mutants, SW)
    (tmp_path / "module.swift").write_text(s.source)
    (tmp_path / s.helper_filename).write_text(s.helper_source)
    (tmp_path / "main.swift").write_text('print(add(5, 3))\n')
    exe = tmp_path / "app"
    subprocess.run(
        ["swiftc", "module.swift", s.helper_filename, "main.swift", "-o", str(exe)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    def run(env_value):
        env = {"PATH": "/usr/bin:/bin"}
        if env_value is not None:
            env["SUPERMUT_MUTANT"] = env_value
        return subprocess.run(
            [str(exe)], capture_output=True, text=True, env=env
        ).stdout.strip()

    assert run(None) == "8"  # original
    assert run("1") == "2"  # mutant 1: a - b
    canary = subprocess.run(
        [str(exe)],
        capture_output=True,
        env={"PATH": "/usr/bin:/bin", "SUPERMUT_MUTANT": str(s.canary_id)},
    )
    assert canary.returncode != 0  # fatalError fired


@needs_kotlinc
def test_kotlin_schemata_typechecks(tmp_path):
    mutants = _mutants(
        KT,
        KOTLIN_SRC,
        [
            ("add", "fun add(a: Int, b: Int): Int {\n    return a - b\n}"),
            ("double", "fun double(x: Int) = x * 3"),
        ],
    )
    s = build_schemata(KOTLIN_SRC, mutants, KT)
    (tmp_path / "Module.kt").write_text(s.source)
    (tmp_path / "Helper.kt").write_text(s.helper_source)
    proc = subprocess.run(
        ["kotlinc", "Module.kt", "Helper.kt", "-d", "out"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
