"""Training -> serving export: pick the best LoRA checkpoint, fuse, fix the
tokenizer, convert to GGUF, quantize.

This is the procedure that was run by hand for v1 and v2; every step below
encodes a debugged failure:

- checkpoint selection reads the mlx_lm training log and takes the
  iteration with the LOWEST validation loss — both v1 and v2 overfit past
  it (v1: 0.83@300 -> 1.00@600), so "last" is the wrong answer;
- mlx_lm's `fuse --export-gguf` does not support gemma3, so we fuse to
  safetensors and use llama.cpp's converter;
- the mlx-community gemma-3 tokenizer carries `<image_soft_token>` with
  id == text vocab size (262144); the converter asserts every id < vocab,
  so the token is stripped from tokenizer.json and tokenizer_config.json.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["best_checkpoint", "fix_tokenizer", "export"]

_VAL_RE = re.compile(r"Iter (\d+): Val loss ([0-9.]+)")


def best_checkpoint(log_text: str, adapters_dir: Path) -> tuple[int, Path]:
    """(iteration, adapter file) with the lowest val loss that has a saved
    checkpoint. Iter 1 is the pre-training baseline and never counts."""
    scores: dict[int, float] = {}
    for m in _VAL_RE.finditer(log_text):
        it, loss = int(m.group(1)), float(m.group(2))
        if it > 1:
            scores[it] = loss
    saved = {
        int(p.name.split("_")[0]): p
        for p in adapters_dir.glob("*_adapters.safetensors")
    }
    candidates = {it: loss for it, loss in scores.items() if it in saved}
    if not candidates:
        raise RuntimeError(
            f"no evaluated iteration has a saved checkpoint in {adapters_dir}"
        )
    best = min(candidates, key=candidates.__getitem__)
    return best, saved[best]


def fix_tokenizer(model_dir: Path, vocab_size: int) -> list[str]:
    """Strip tokens with id >= vocab_size. Returns what was removed."""
    removed: list[str] = []
    tok_path = model_dir / "tokenizer.json"
    tok = json.loads(tok_path.read_text())
    keep = []
    for t in tok.get("added_tokens", []):
        if t["id"] >= vocab_size:
            removed.append(f"added_tokens:{t['content']}")
        else:
            keep.append(t)
    tok["added_tokens"] = keep
    tok_path.write_text(json.dumps(tok))

    cfg_path = model_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    for key, val in list(cfg.items()):
        if isinstance(val, str) and "image_soft" in val:
            removed.append(f"tokenizer_config:{key}")
            del cfg[key]
    atd = cfg.get("added_tokens_decoder", {})
    for k in [k for k in atd if int(k) >= vocab_size]:
        removed.append(f"added_tokens_decoder:{k}")
        del atd[k]
    cfg_path.write_text(json.dumps(cfg))
    return removed


def export(
    *,
    base_model: str,
    adapters_dir: Path,
    log_path: Path,
    out_dir: Path,
    name: str,
    converter: Path,
    quantize_bin: Path,
    quant: str = "Q8_0",
    python: str = sys.executable,
) -> dict[str, Path]:
    """Full pipeline; returns paths of the f16 and quantized GGUFs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    it, ckpt = best_checkpoint(log_path.read_text(), adapters_dir)
    print(f"best checkpoint: iter {it} ({ckpt.name})")
    shutil.copy(ckpt, adapters_dir / "adapters.safetensors")

    fused = out_dir / "fused"
    subprocess.run(
        [
            python,
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            base_model,
            "--adapter-path",
            str(adapters_dir),
            "--save-path",
            str(fused),
        ],
        check=True,
    )
    cfg = json.loads((fused / "config.json").read_text())
    vocab = cfg.get("vocab_size") or cfg["text_config"]["vocab_size"]
    print("tokenizer fix removed:", fix_tokenizer(fused, vocab))

    f16 = out_dir / f"{name}-f16.gguf"
    subprocess.run(
        [python, str(converter), str(fused), "--outfile", str(f16), "--outtype", "f16"],
        check=True,
    )
    quantized = out_dir / f"{name}-{quant}.gguf"
    subprocess.run([str(quantize_bin), str(f16), str(quantized), quant], check=True)
    return {"f16": f16, "quantized": quantized, "checkpoint": ckpt}


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="supermut.export")
    p.add_argument("--adapters", required=True, type=Path)
    p.add_argument("--log", required=True, type=Path, help="mlx_lm training log")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--name", required=True, help="GGUF base name, e.g. supermut-gemma-v3")
    p.add_argument("--base-model", default="mlx-community/gemma-3-270m-it-bf16")
    p.add_argument("--converter", type=Path, default=Path("/tmp/llama.cpp-src/convert_hf_to_gguf.py"))
    p.add_argument("--quantize-bin", type=Path, default=Path("/opt/homebrew/opt/llama.cpp/bin/llama-quantize"))
    p.add_argument("--quant", default="Q8_0")
    p.add_argument("--python", default=sys.executable, help="interpreter with mlx_lm + torch + transformers")
    a = p.parse_args(argv)
    for path, what in [(a.converter, "converter"), (a.quantize_bin, "llama-quantize")]:
        if not path.exists():
            print(f"{what} not found: {path}", file=sys.stderr)
            return 2
    paths = export(
        base_model=a.base_model,
        adapters_dir=a.adapters,
        log_path=a.log,
        out_dir=a.out,
        name=a.name,
        converter=a.converter,
        quantize_bin=a.quantize_bin,
        quant=a.quant,
        python=a.python,
    )
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
