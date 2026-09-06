"""Dataset mining tests: pair extraction on strings, full mine on a
synthetic git repo."""

import json
import subprocess
import textwrap

from supermut.dataset import (
    Sample,
    extract_function_pairs,
    main,
    mine_repo,
    to_prompt_completion,
)

OLD = textwrap.dedent(
    """
    def clamp(x, lo, hi):
        if x <= lo:
            return lo
        return x


    def untouched(a):
        return a * 2
    """
).lstrip()

NEW = textwrap.dedent(
    """
    def clamp(x, lo, hi):
        if x < lo:
            return lo
        return x


    def untouched(a):
        return a * 2
    """
).lstrip()


def test_extract_single_changed_pair():
    pairs = extract_function_pairs(OLD, NEW)
    assert len(pairs) == 1
    name, old_f, new_f = pairs[0]
    assert name == "clamp"
    assert "x <= lo" in old_f and "x < lo" in new_f


def test_extract_rejects_multi_function_diffs():
    new2 = NEW.replace("return a * 2", "return a * 3")
    assert extract_function_pairs(OLD, new2) == []


def test_extract_rejects_formatting_only_change():
    reformatted = OLD.replace("if x <= lo:", "if x <= lo:  # boundary")
    assert extract_function_pairs(OLD, reformatted) == []


def test_extract_rejects_large_change():
    big = OLD.replace(
        "        return lo",
        "\n".join(f"        x += {i}" for i in range(20)) + "\n        return lo",
    )
    assert extract_function_pairs(big, NEW, max_changed_lines=12) == []


def test_extract_methods_qualified_names():
    old = "class A:\n    def f(self):\n        return 1\n"
    new = "class A:\n    def f(self):\n        return 2\n"
    pairs = extract_function_pairs(old, new)
    assert pairs[0][0] == "A.f"


def test_prompt_completion_format():
    s = Sample(
        repo="r",
        commit="c",
        file="f.py",
        function="clamp",
        fixed_source="def clamp(x):\n    return max(x, 0)",
        buggy_source="def clamp(x):\n    return min(x, 0)",
    )
    pc = to_prompt_completion(s)
    assert pc["prompt"].startswith("# Original function:\ndef clamp(x):")
    assert pc["prompt"].endswith("def clamp(x):\n")
    assert pc["completion"] == "    return min(x, 0)"


def _git(cwd, *args):
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": "/usr/bin:/bin",
            "HOME": str(cwd),
        },
    )


def test_mine_synthetic_repo(tmp_path):
    repo = tmp_path / "proj"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "calc.py").write_text(OLD)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "initial")
    (repo / "calc.py").write_text(NEW)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fix off-by-one in clamp boundary")
    # A noise commit that must not be mined (message doesn't match).
    (repo / "other.py").write_text("def g():\n    return 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add feature g")

    samples = mine_repo(repo)
    assert len(samples) == 1
    s = samples[0]
    assert s.function == "clamp"
    assert "x <= lo" in s.buggy_source
    assert "x < lo" in s.fixed_source

    # CLI end-to-end.
    out = tmp_path / "ds.jsonl"
    assert main([str(repo), "-o", str(out)]) == 0
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["prompt"].startswith("# Original function:")
    assert "x <= lo" in records[0]["completion"]
