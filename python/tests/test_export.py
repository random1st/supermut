"""Export pipeline tests — model-free: checkpoint selection from a training
log, tokenizer vocab fix on synthetic files."""

import json

import pytest

from supermut.export import best_checkpoint, fix_tokenizer

LOG = """\
Iter 1: Val loss 2.018, Val took 21.6s
Iter 200: Val loss 1.073, Val took 16.2s
Iter 200: Train loss 1.1, Learning Rate 1e-4
Iter 400: Val loss 0.978, Val took 16.3s
Iter 600: Val loss 0.950, Val took 16.0s
Iter 800: Val loss 0.991, Val took 16.1s
"""


def _adapters(tmp_path, iters):
    d = tmp_path / "adapters"
    d.mkdir()
    for it in iters:
        (d / f"{it:07d}_adapters.safetensors").write_bytes(b"x")
    return d


def test_best_checkpoint_picks_min_val_not_last(tmp_path):
    d = _adapters(tmp_path, [200, 400, 600, 800])
    it, path = best_checkpoint(LOG, d)
    assert it == 600
    assert path.name == "0000600_adapters.safetensors"


def test_best_checkpoint_ignores_iter1_and_unsaved(tmp_path):
    # 600 has the best loss but no saved file -> falls back to 400
    d = _adapters(tmp_path, [200, 400, 800])
    it, _ = best_checkpoint(LOG, d)
    assert it == 400


def test_best_checkpoint_no_candidates(tmp_path):
    d = _adapters(tmp_path, [])
    with pytest.raises(RuntimeError, match="no evaluated iteration"):
        best_checkpoint(LOG, d)


def test_fix_tokenizer_strips_over_vocab(tmp_path):
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 1, "content": "<bos>"},
                    {"id": 262144, "content": "<image_soft_token>"},
                ],
                "model": {"vocab": {}},
            }
        )
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "image_token": "<image_soft_token>",
                "boi_token": "<start_of_image>",
                "added_tokens_decoder": {"1": {}, "262144": {}},
            }
        )
    )
    removed = fix_tokenizer(tmp_path, 262144)
    assert set(removed) == {
        "added_tokens:<image_soft_token>",
        "tokenizer_config:image_token",
        "added_tokens_decoder:262144",
    }
    tok = json.loads((tmp_path / "tokenizer.json").read_text())
    assert [t["id"] for t in tok["added_tokens"]] == [1]
    cfg = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert "image_token" not in cfg
    assert cfg["boi_token"] == "<start_of_image>"  # unrelated field untouched
    assert list(cfg["added_tokens_decoder"]) == ["1"]
