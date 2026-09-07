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
    parser.add_argument("file", help="source file to mutate (.py/.js/.ts/...)")
    parser.add_argument(
        "--tests",
        default="",
        help='runner arguments: pytest args for Python (e.g. "-q tests/"), '
        "extra CLI args for vitest/jest",
    )
    parser.add_argument(
        "--runner-cmd",
        help='override the runner executable, e.g. "bunx vitest" '
        "(default: npx <runner> for JS/TS)",
    )
    parser.add_argument(
        "--python", default=sys.executable, help="interpreter to run tests with"
    )
    parser.add_argument("--json", dest="json_out", help="write report JSON here")
    parser.add_argument(
        "--no-cache", action="store_true", help="ignore .supermut-cache.json"
    )
    parser.add_argument(
        "--model", help="GGUF path or HF repo id (omit with --operators-only)"
    )
    parser.add_argument(
        "--llm-only", action="store_true", help="skip cheap operator mutants"
    )
    parser.add_argument(
        "--operators-only", action="store_true", help="no model, operator arm alone"
    )
    parser.add_argument("--filename", help="GGUF filename inside the HF repo")
    parser.add_argument("--cwd", help="directory to run tests from (default: file's dir)")
    parser.add_argument("-n", "--n-per-target", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument(
        "--top-p", type=float, default=1.0, help="1.0 = off; 0.95 throttles diversity"
    )
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--n-ctx", type=int, default=4096)
    args = parser.parse_args(argv)

    if args.operators_only:
        llm = None
    elif args.model:
        llm = LLM.from_pretrained(args.model, filename=args.filename, n_ctx=args.n_ctx)
    else:
        parser.error("--model is required unless --operators-only")

    def progress(done: int, total: int, status) -> None:
        print(f"[{done}/{total}] {status.value}", flush=True)

    try:
        report = run(
            args.file,
            args.tests,
            llm,
            cwd=args.cwd,
            python=args.python,
            runner_cmd=args.runner_cmd,
            n_per_target=args.n_per_target,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            seed=args.seed,
            timeout_s=args.timeout,
            use_cache=not args.no_cache,
            operators=not args.llm_only,
            on_progress=progress,
        )
    except (ValueError, RuntimeError) as e:
        # Designed user-facing failures (unsupported language, failing
        # baseline, canary gate) — no traceback noise.
        print(f"error: {e}", file=sys.stderr)
        return 2
    print()
    print(report.summary())
    if args.json_out:
        from pathlib import Path

        Path(args.json_out).write_text(report.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
