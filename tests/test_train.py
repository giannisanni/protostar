import numpy as np
import pytest
import torch

from data.loader import write_shard
from model import Config
from train import TrainConfig, train

TINY = dict(d=128, n_layer=4, n_head=4, n_kv_head=2, seq_len=64, vocab=512)


@pytest.fixture()
def data_dir(tmp_path):
    # ponytail: repeating pattern = trivially learnable, perfect for overfit check
    pattern = np.tile(np.arange(256, dtype=np.uint16), 400)
    write_shard(tmp_path / "shard_000000.bin", pattern)
    write_shard(tmp_path / "shard_000001.bin", pattern)  # val
    return tmp_path


def tcfg(data_dir, out_dir, **kw):
    base = dict(data_dir=data_dir, out_dir=out_dir, steps=300, batch=4, accum=1,
                warmup=20, val_every=0, ckpt_every=0, seed=0)
    return TrainConfig(**{**base, **kw})


def test_overfit_tiny(data_dir, tmp_path):
    loss = train(Config(**TINY), tcfg(data_dir, tmp_path / "out"))
    assert loss < 1.0, f"failed to overfit repeating pattern, loss={loss:.3f}"


def test_resume_matches_uninterrupted(data_dir, tmp_path):
    cfg = Config(**TINY)
    train(cfg, tcfg(data_dir, tmp_path / "a", steps=60, ckpt_every=50))
    train(cfg, tcfg(data_dir, tmp_path / "b", steps=60, resume=str(tmp_path / "a" / "ckpt_000050.pt")))
    a = torch.load(tmp_path / "a" / "ckpt_000060.pt", weights_only=True)["model"]
    b = torch.load(tmp_path / "b" / "ckpt_000060.pt", weights_only=True)["model"]
    for k in a:
        torch.testing.assert_close(a[k], b[k], atol=0, rtol=0, msg=f"param {k} diverged after resume")
