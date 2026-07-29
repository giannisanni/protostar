import numpy as np
import pytest
import torch

from data.loader import Shards, write_shard

B, T = 4, 32


@pytest.fixture()
def shard_dir(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(3):  # shard 2 becomes val
        toks = rng.integers(0, 50257, size=5_000, dtype=np.uint16)
        write_shard(tmp_path / f"shard_{i:06d}.bin", toks)
    return tmp_path


def test_shapes_and_dtype(shard_dir):
    batch = next(Shards(shard_dir, "train", B, T).batches())
    assert batch.shape == (B, T + 1)
    assert batch.dtype == torch.int64


def test_deterministic(shard_dir):
    a = next(Shards(shard_dir, "train", B, T).batches())
    b = next(Shards(shard_dir, "train", B, T).batches())
    assert torch.equal(a, b)


def test_ranks_disjoint(shard_dir):
    a = next(Shards(shard_dir, "train", B, T, rank=0, world=2).batches())
    b = next(Shards(shard_dir, "train", B, T, rank=1, world=2).batches())
    assert not torch.equal(a, b)


def test_val_is_last_shard_only(shard_dir):
    train = Shards(shard_dir, "train", B, T)
    val = Shards(shard_dir, "val", B, T)
    assert [p.name for p in val.paths] == ["shard_000002.bin"]
    assert all(p.name != "shard_000002.bin" for p in train.paths)


def test_roundtrip(tmp_path):
    toks = np.arange(1000, dtype=np.uint16)
    write_shard(tmp_path / "s.bin", toks)
    first = next(Shards(tmp_path, "all", 1, 16).batches())[0]
    assert torch.equal(first, torch.arange(17, dtype=torch.int64))
