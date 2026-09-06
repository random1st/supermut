"""Model resolution: local paths and Hugging Face Hub downloads.

Downloads land in the standard shared HF cache (``~/.cache/huggingface``),
so models pulled by other tools are reused, not re-downloaded. Auth comes
from the standard ``huggingface_hub`` login (``hf auth login``) or the
``HF_TOKEN`` env var — required only for private repos.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

__all__ = ["resolve_model"]


def resolve_model(
    source: str | Path,
    *,
    filename: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Resolve *source* to a local GGUF file path.

    - Existing local path: returned as-is (``filename`` joined if it's a dir).
    - HF repo id (``user/repo``) + ``filename``: downloads that one file.
    - HF repo id alone: snapshots the repo and expects exactly one ``*.gguf``.
    """
    path = Path(source)
    if path.exists():
        if path.is_dir():
            return _pick_gguf(path, filename)
        return path

    repo_id = str(source)
    if filename is not None:
        return Path(
            hf_hub_download(repo_id, filename, revision=revision, token=token)
        )

    snapshot = Path(
        snapshot_download(
            repo_id, revision=revision, token=token, allow_patterns=["*.gguf"]
        )
    )
    return _pick_gguf(snapshot, None)


def _pick_gguf(directory: Path, filename: str | None) -> Path:
    if filename is not None:
        candidate = directory / filename
        if not candidate.exists():
            raise FileNotFoundError(f"{filename} not found in {directory}")
        return candidate

    ggufs = sorted(directory.rglob("*.gguf"))
    if not ggufs:
        raise FileNotFoundError(
            f"no .gguf files in {directory}; pass filename= to pick a quantization"
        )
    if len(ggufs) > 1:
        names = ", ".join(p.name for p in ggufs)
        raise ValueError(
            f"multiple .gguf files in {directory}: {names}; pass filename= to choose"
        )
    return ggufs[0]
