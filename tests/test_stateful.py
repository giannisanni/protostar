import numpy as np
import torch

from accretion.fast import FastMemory
from data.loader import StatefulShards, write_shard
from model import Config, Protostar
from train import TrainConfig, val_loss


def test_stateful_lanes_stream_contiguously(tmp_path):
    """Lane b's window at pos+1 must continue lane b's window at pos — that
    continuity IS the training signal for reading old memory. Plain Shards
    deals consecutive windows to different lanes instead."""
    write_shard(tmp_path / "a.bin", np.arange(1000, dtype=np.uint16))
    ld = StatefulShards(str(tmp_path), "all", batch=2, seq_len=4)
    it = ld.batches()
    b0, b1 = next(it), next(it)
    for lane in range(2):
        assert b1[lane, 0] == b0[lane, -1] + 1, f"lane {lane} not contiguous"
    assert b0[0, 0] != b0[1, 0], "lanes overlap — they must stream disjoint spans"


def test_reset_lanes_resets_only_masked():
    torch.manual_seed(0)
    fm = FastMemory(d=16, chunk=8)
    fm(torch.randn(2, 16, 16))
    before = fm._state.clone()
    fm.reset_lanes(torch.tensor([True, False]))
    torch.testing.assert_close(fm._state[0], fm.m0.detach())
    torch.testing.assert_close(fm._state[1], before[1])


def test_val_does_not_clobber_persistent_state(tmp_path):
    """val_loss on a persist_memory model must measure FRESH memory and hand
    the training lanes' state back untouched. Without the guard, val's resets
    wipe every lane — the same silent-wipe class as the one-shot 0% bug."""
    write_shard(tmp_path / "a.bin", (np.arange(20000) % 50000).astype(np.uint16))
    cfg = Config(d=32, n_layer=2, n_head=2, n_kv_head=1,
                 fast_memory=True, fast_layer=1, seq_len=16)
    model = Protostar(cfg)
    model.persist_memory = True
    model(torch.randint(0, 100, (2, 16)))
    lanes = model.fast._state.clone()
    t = TrainConfig(data_dir=str(tmp_path), batch=2, val_batches=2)
    val_loss(model, cfg, t, torch.device("cpu"))
    assert model.persist_memory
    torch.testing.assert_close(model.fast._state, lanes)
