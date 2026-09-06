"""Benchmark: how many known test holes does each mutation tool expose?

A hole is exposed when a SURVIVING mutant touches the associated function.
mutmut is the rule-based control arm; supermut runs once per model.

Usage:
    python bench/run_bench.py --model /path/to/a.gguf --model repo/b-GGUF:file.gguf
    (each --model becomes one supermut column; mutmut always runs)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BENCH_DIR = Path(__file__).parent

# function -> short description of the deliberately-untested behavior
KNOWN_HOLES = {
    "clamp": "boundaries x==lo / x==hi",
    "is_even": "odd numbers",
    "sign": "zero",
    "find_index": "not-found path",
    "discount": "rate==1 boundary, zero price",
    "merge_ranges": "touching ranges a_end==b_start",
    "truncate": "exact-limit boundary, ellipsis",
    "safe_div": "b==0 path",
}


def make_bed() -> Path:
    bed = Path(tempfile.mkdtemp(prefix="supermut-bench-"))
    shutil.copy(BENCH_DIR / "target_module.py", bed / "target_module.py")
    (bed / "tests").mkdir()
    shutil.copy(BENCH_DIR / "tests" / "test_target.py", bed / "tests" / "test_target.py")
    return bed


def run_mutmut(bed: Path) -> dict:
    (bed / "pyproject.toml").write_text(
        '[tool.mutmut]\nsource_paths = ["target_module.py"]\n'
    )
    subprocess.run(["git", "init", "-q"], cwd=bed, check=True)
    t0 = time.time()
    subprocess.run(
        [sys.executable, "-m", "mutmut", "run"],
        cwd=bed,
        capture_output=True,
        timeout=600,
    )
    elapsed = time.time() - t0
    res = subprocess.run(
        [sys.executable, "-m", "mutmut", "results", "--all", "true"],
        cwd=bed,
        capture_output=True,
        text=True,
    ).stdout
    survivors_by_func: set[str] = set()
    total = killed = survived = 0
    for line in res.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, _, status = line.rpartition(":")
        status = status.strip()
        if status in ("killed", "survived", "timeout"):
            total += 1
        if status == "survived":
            survived += 1
            # name like target_module.x_clamp__mutmut_1
            func = name.split(".x_")[-1].rpartition("__mutmut_")[0]
            survivors_by_func.add(func)
        elif status in ("killed", "timeout"):
            killed += 1
    holes = {f for f in KNOWN_HOLES if f in survivors_by_func}
    return {
        "tool": "mutmut",
        "mutants": total,
        "killed": killed,
        "survived": survived,
        "holes_found": sorted(holes),
        "time_s": round(elapsed, 1),
    }


def run_supermut(bed: Path, model: str, label: str, n: int, seed: int) -> dict:
    args = [
        sys.executable,
        "-m",
        "supermut",
        str(bed / "target_module.py"),
        "--tests",
        "-q tests/",
        "--python",
        sys.executable,
        "--cwd",
        str(bed),
        "-n",
        str(n),
        "--n-ctx",
        "2048",
        "--seed",
        str(seed),
        "--temperature",
        "0.9",
        "--no-cache",
        "--json",
        str(bed / "report.json"),
    ]
    if ":" in model and not Path(model).exists():
        repo, _, filename = model.partition(":")
        args += ["--model", repo, "--filename", filename]
    else:
        args += ["--model", model]
    t0 = time.time()
    proc = subprocess.run(args, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {"tool": label, "error": proc.stdout[-500:] + proc.stderr[-500:]}
    report = json.loads((bed / "report.json").read_text())
    survivors_by_func = {
        m["target"] for m in report["mutants"] if m["status"] == "survived"
    }
    holes = {f for f in KNOWN_HOLES if f in survivors_by_func}
    return {
        "tool": label,
        "mutants": report["total"],
        "killed": report["killed"],
        "survived": report["survived"],
        "no_tests": report["no_tests"],
        "holes_found": sorted(holes),
        "time_s": round(elapsed, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="GGUF path or 'hf-repo:filename'; repeatable",
    )
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("-n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--skip-mutmut", action="store_true")
    args = parser.parse_args()

    rows = []
    if not args.skip_mutmut:
        rows.append(run_mutmut(make_bed()))
    for i, model in enumerate(args.model):
        label = args.label[i] if i < len(args.label) else Path(model).stem
        rows.append(run_supermut(make_bed(), model, label, args.n, args.seed))

    print(f"\nknown holes: {len(KNOWN_HOLES)} -> {', '.join(sorted(KNOWN_HOLES))}\n")
    header = f"{'tool':24} {'mutants':>7} {'killed':>6} {'survived':>8} {'holes':>5}  {'time':>6}  holes found"
    print(header)
    print("-" * len(header))
    for r in rows:
        if "error" in r:
            print(f"{r['tool']:24} ERROR: {r['error'][:80]}")
            continue
        print(
            f"{r['tool']:24} {r['mutants']:>7} {r['killed']:>6} {r['survived']:>8} "
            f"{len(r['holes_found']):>5}  {r['time_s']:>5}s  {', '.join(r['holes_found'])}"
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
