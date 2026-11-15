"""Slot memory: reads earlier windows only, recalls a stored token exactly,
resets per lane, keeps a bptt graph, and seeds from a v2 checkpoint."""
import torch

from accretion.slots import SlotMemory
from model import Config, Protostar
from train import seed_cms


def test_first_window_reads_nothing_then_recalls_stored_value():
    torch.manual_seed(0)
    m = SlotMemory(16, n_head=2, size=64)
    with torch.no_grad():
        for w in (m.wq, m.wk, m.wv):
            w.weight.copy_(torch.eye(16))
        m.scale.fill_(50.0)  # sharp read
    m.reset(1)
    h = torch.randn(1, 8, 16)
    out, _ = m(h)
    assert out.abs().max() < 1e-6, "window 1 must see no slots (null slot reads 0)"
    out, _ = m(h[:, 3:4])  # query = a stored token's own key
    assert torch.allclose(out[0, 0], m.ln(h)[0, 3], atol=1e-2), "slot read did not return the stored value"


def test_fifo_cap_and_lane_reset():
    m = SlotMemory(16, n_head=2, size=10)
    m.reset(2)
    for _ in range(3):
        m(torch.randn(2, 4, 16))
    assert m._k.shape[1] == 10
    m.reset_lanes(torch.tensor([True, False]))
    assert not m._ok[0].any() and m._ok[1].all()


def test_bptt_reaches_previous_window_write():
    m = SlotMemory(16, n_head=2, size=64, bptt=True)
    m.reset(1)
    h1 = torch.randn(1, 8, 16, requires_grad=True)
    m(h1)
    out, _ = m(torch.randn(1, 8, 16))
    out.sum().backward()
    assert h1.grad is not None and h1.grad.abs().sum() > 0


def test_slots_model_seeds_from_ttt_checkpoint():
    kw = dict(d=64, n_layer=2, n_head=4, vocab=128, seq_len=32, fast_memory=True, fast_layer=0)
    src = Protostar(Config(**kw)).state_dict()
    dst = Protostar(Config(**kw, mem="slots"))
    dst.load_state_dict(seed_cms(src, dst.state_dict()))  # strict: every key accounted for
    assert torch.equal(dst.embed.weight, src["embed.weight"])
