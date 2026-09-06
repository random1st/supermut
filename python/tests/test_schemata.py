"""Schemata assembly tests: structure via tree-sitter, then real compiler
smoke tests (swiftc -typecheck / kotlinc) and a runtime switch check —
tree-sitter parsing alone is not a typecheck.
"""

import os
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


# Captured from `swift test --parallel --xunit-output res.xml`
# (Swift 6.3.3; the XCTest xUnit file is only written in --parallel mode).
XUNIT = """<?xml version="1.0" encoding="UTF-8"?>

<testsuites>
<testsuite name="TestResults" errors="0" tests="2" failures="0" time="0.097650084">
<testcase classname="DemoTests.CalcTests" name="testClamp" time="0.048824584">
</testcase>
<testcase classname="DemoTests.CalcTests" name="testAdd" time="0.0488255">
</testcase>
</testsuite>
</testsuites>
"""


def test_parse_xunit():
    from supermut.runners import _parse_xunit

    durations = _parse_xunit(XUNIT)
    assert durations == {
        "DemoTests.CalcTests/testClamp": pytest.approx(0.0488, abs=1e-3),
        "DemoTests.CalcTests/testAdd": pytest.approx(0.0488, abs=1e-3),
    }


def test_failed_ids_maps_diagnostics_to_fences():
    from supermut.schemata import failed_ids

    src = "\n".join(
        [
            "func f() {}",  # 1
            "// __supermut_begin_3",  # 2
            "func __sm_3() {}",  # 3
            "// __supermut_end_3",  # 4
            "func g() {}",  # 5
        ]
    )
    diag = "/pkg/Sources/Calc.swift:3:10: error: cannot convert value"
    assert failed_ids(src, diag, "Calc.swift") == {3}
    # errors outside every fence map to nothing — caller must surface them
    assert failed_ids(src, "/pkg/Sources/Calc.swift:5:1: error: boom", "Calc.swift") == set()
    assert failed_ids(src, "/pkg/Sources/Calc.swift:3:1: warning: unused", "Calc.swift") == set()


needs_swift_spm = pytest.mark.skipif(
    shutil.which("swift") is None, reason="swift toolchain not installed"
)

SPM_CALC = textwrap.dedent(
    """\
    public func add(_ a: Int, _ b: Int) -> Int {
        return a + b
    }
    """
)


def _make_spm_package(tmp_path):
    (tmp_path / "Sources" / "Demo").mkdir(parents=True)
    (tmp_path / "Tests" / "DemoTests").mkdir(parents=True)
    (tmp_path / "Package.swift").write_text(
        textwrap.dedent(
            """\
            // swift-tools-version:5.9
            import PackageDescription

            let package = Package(
                name: "Demo",
                targets: [
                    .target(name: "Demo"),
                    .testTarget(name: "DemoTests", dependencies: ["Demo"]),
                ]
            )
            """
        )
    )
    calc = tmp_path / "Sources" / "Demo" / "Calc.swift"
    calc.write_text(SPM_CALC)
    (tmp_path / "Tests" / "DemoTests" / "CalcTests.swift").write_text(
        textwrap.dedent(
            """\
            import XCTest
            @testable import Demo

            final class CalcTests: XCTestCase {
                func testAdd() {
                    XCTAssertEqual(Demo.add(2, 3), 5)
                }
            }
            """
        )
    )
    return calc


class _StubLLM:
    def __init__(self, completions):
        self._completions = completions

    def generate_batch(self, prefix, continuations, **kwargs):
        return self._completions[: len(continuations)]


@needs_swift_spm
def test_swift_full_cycle(tmp_path):
    from supermut.harness import MutantStatus, run

    calc = _make_spm_package(tmp_path)
    llm = _StubLLM(["    return a - b\n}"])
    report = run(
        calc, "", llm, cwd=tmp_path, n_per_target=1, use_cache=False,
        timeout_s=240,
    )
    assert calc.read_text() == SPM_CALC  # restored
    assert not (tmp_path / "Sources" / "Demo" / "__supermut_helper.swift").exists()
    statuses = [r.status for r in report.results]
    assert statuses == [MutantStatus.KILLED]


@needs_swift_spm
def test_swift_compile_error_recovery(tmp_path):
    from supermut.harness import MutantStatus, run

    calc = _make_spm_package(tmp_path)
    # First mutant parses but cannot typecheck (String for Int);
    # recovery must drop it, rebuild, and still score the real mutant.
    llm = _StubLLM(['    return "oops"\n}', "    return a - b\n}"])
    report = run(
        calc, "", llm, cwd=tmp_path, n_per_target=2, use_cache=False,
        timeout_s=240,
    )
    assert calc.read_text() == SPM_CALC
    by_status = {r.status for r in report.results}
    assert by_status == {MutantStatus.COMPILE_ERROR, MutantStatus.KILLED}


# Captured from build/test-results/test/TEST-demo.CalcTests.xml
# (Gradle 9.5.1, kotlin("jvm") 2.3.20, JUnit platform).
JUNIT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="demo.CalcTests" tests="1" skipped="0" failures="0" errors="0" \
timestamp="2026-09-06T19:42:42.432Z" hostname="M3MAX.local" time="0.017">
  <properties/>
  <testcase name="testAdd()" classname="demo.CalcTests" time="0.011"/>
  <system-out><![CDATA[]]></system-out>
  <system-err><![CDATA[]]></system-err>
</testsuite>
"""


def test_parse_junit_xml():
    from supermut.runners import _parse_junit_xml

    # the "()" display-name suffix is stripped to match --tests format
    assert _parse_junit_xml(JUNIT_XML) == {
        "demo.CalcTests.testAdd": pytest.approx(0.011)
    }


needs_gradle = pytest.mark.skipif(
    not (os.environ.get("SUPERMUT_TEST_GRADLE") and shutil.which("gradle")),
    reason="set SUPERMUT_TEST_GRADLE=1 (needs gradle + a JDK; first run "
    "downloads the kotlin plugin)",
)

GRADLE_CALC = textwrap.dedent(
    """\
    package demo

    fun add(a: Int, b: Int): Int {
        return a + b
    }
    """
)


def _make_gradle_project(tmp_path):
    (tmp_path / "src" / "main" / "kotlin").mkdir(parents=True)
    (tmp_path / "src" / "test" / "kotlin").mkdir(parents=True)
    (tmp_path / "settings.gradle.kts").write_text('rootProject.name = "demo"\n')
    (tmp_path / "build.gradle.kts").write_text(
        textwrap.dedent(
            """\
            plugins {
                kotlin("jvm") version "2.3.20"
            }

            repositories {
                mavenCentral()
            }

            dependencies {
                testImplementation(kotlin("test"))
            }

            tasks.test {
                useJUnitPlatform()
            }

            kotlin {
                compilerOptions {
                    jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
                }
            }

            java {
                sourceCompatibility = JavaVersion.VERSION_17
                targetCompatibility = JavaVersion.VERSION_17
            }
            """
        )
    )
    calc = tmp_path / "src" / "main" / "kotlin" / "Calc.kt"
    calc.write_text(GRADLE_CALC)
    (tmp_path / "src" / "test" / "kotlin" / "CalcTests.kt").write_text(
        textwrap.dedent(
            """\
            package demo

            import kotlin.test.Test
            import kotlin.test.assertEquals

            class CalcTests {
                @Test
                fun testAdd() {
                    assertEquals(5, add(2, 3))
                }
            }
            """
        )
    )
    return calc


@needs_gradle
def test_kotlin_full_cycle_with_recovery(tmp_path):
    from supermut.harness import MutantStatus, run

    calc = _make_gradle_project(tmp_path)
    # one type-broken mutant (recovery path) + one real mutant (killed)
    llm = _StubLLM(['    return "oops"\n}', "    return a - b\n}"])
    report = run(
        calc, "", llm, cwd=tmp_path, n_per_target=2, use_cache=False,
        timeout_s=300,
    )
    assert calc.read_text() == GRADLE_CALC
    assert not (tmp_path / "src" / "main" / "kotlin" / "__supermut_helper.kt").exists()
    assert not (tmp_path / ".supermut-init.gradle").exists()
    by_status = {r.status for r in report.results}
    assert by_status == {MutantStatus.COMPILE_ERROR, MutantStatus.KILLED}


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
