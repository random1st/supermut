"""Language protocol tests across all five frontends."""

import pytest

from supermut.languages import all_languages, language_by_name, language_for_path

SOURCES = {
    "python": (
        "def add(a, b):\n    return a + b\n\n\nclass A:\n    def m(self):\n        return 1\n",
        ["add", "m"],
    ),
    "javascript": (
        "function add(a, b) { return a + b }\nclass A { m() { return 1 } }\n",
        ["add", "m"],
    ),
    "typescript": (
        "export function add(a: number, b: number): number { return a + b }\n"
        "class A { m(): number { return 1 } }\n",
        ["add", "m"],
    ),
    "kotlin": (
        "fun add(a: Int, b: Int): Int = a + b\nclass A { fun m(): Int { return 1 } }\n",
        ["add", "m"],
    ),
    "swift": (
        "func add(_ a: Int, _ b: Int) -> Int { a + b }\n"
        "class A { func m() -> Int { 1 } }\n",
        ["add", "m"],
    ),
}


@pytest.mark.parametrize("lang_name", list(SOURCES))
def test_find_targets(lang_name):
    lang = language_by_name(lang_name)
    src, expected = SOURCES[lang_name]
    targets = lang.find_targets(src)
    assert sorted(t.name for t in targets) == sorted(expected)
    for t in targets:
        assert t.name in t.source
        assert 1 <= t.start_line <= t.end_line


@pytest.mark.parametrize("lang_name", list(SOURCES))
def test_validate_and_normalize(lang_name):
    lang = language_by_name(lang_name)
    src, _ = SOURCES[lang_name]
    targets = lang.find_targets(src)
    add = next(t for t in targets if t.name == "add")
    body = add.source.lstrip()
    assert lang.validate(body, "add")
    assert not lang.validate(body, "other_name")
    assert not lang.validate("garbage (((", "add")

    n1 = lang.normalize(body)
    assert n1 is not None
    # comment/whitespace-insensitive
    commented = body.replace(
        "\n", f"  {lang.comment_prefix} note\n", 1
    ) if "\n" in body else body + f"  {lang.comment_prefix} note"
    n2 = lang.normalize(commented)
    assert n2 == n1, f"{lang_name} normalize not comment-insensitive"
    assert lang.normalize("garbage (((") is None


def test_language_for_path():
    assert language_by_name("python") is not None  # force registry load
    assert language_for_path("a/b/x.py").name == "python"
    assert language_for_path("x.ts").name == "typescript"
    assert language_for_path("x.kt").name == "kotlin"
    assert language_for_path("x.swift").name == "swift"
    assert language_for_path("x.mjs").name == "javascript"
    assert language_for_path("x.rb") is None


def test_nested_functions_not_extracted_ts():
    lang = language_by_name("typescript")
    src = "function outer() {\n  function inner() { return 1 }\n  return inner()\n}\n"
    assert [t.name for t in lang.find_targets(src)] == ["outer"]


def test_registry_has_all_five():
    assert set(all_languages()) == {
        "python",
        "javascript",
        "typescript",
        "kotlin",
        "swift",
    }
