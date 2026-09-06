"""CLI: python -m supermut <file> --tests "pytest -x -q" --model <repo-or-path>"""

from __future__ import annotations

import argparse
import sys

from supermut.engine import LLM
from supermut.harness import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="supermut", description="LLM-driven mutation testing"
    )
    parser.add_argument("file", help="Python module to mutate")
    parser.add_argument(
        "--tests", required=True, help='pytest arguments, e.g. "-q tests/"'
    )
    parser.add_argument(
        "--python", default=sys.executable, help="interpreter to run tests with"
    )
    parser.add_argument("--json", dest="json_out", help="write report JSON here")
    parser.add_argument(
        "--no-cache", action="store_true", help="ignore .supermut-cache.json"
    )
    parser.add_argument(
        "--model", required=True, help="GGUF path or HF repo id"
    )
    parser.add_argument("--filename", help="GGUF filename inside the HF repo")
    parser.add_argument("--cwd", help="directory to run tests from (default: file's dir)")
    parser.add_argument("-n", "--n-per-target", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--n-ctx", type=int, default=4096)
    args = parser.parse_args(argv)

    llm = LLM.from_pretrained(args.model, filename=args.filename, n_ctx=args.n_ctx)

    def progress(done: int, total: int, status) -> None:
        print(f"[{done}/{total}] {status.value}", flush=True)

    report = run(
        args.file,
        args.tests,
        llm,
        cwd=args.cwd,
        python=args.python,
        n_per_target=args.n_per_target,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        timeout_s=args.timeout,
        use_cache=not args.no_cache,
        on_progress=progress,
    )
    print()
    print(report.summary())
    if args.json_out:
        from pathlib import Path

        Path(args.json_out).write_text(report.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
