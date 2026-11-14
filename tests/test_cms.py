"""hope-lite: Continuum Memory System levels, cross-window BPTT, readback."""
import torch

from accretion.fast import FastMemory, MemoryStack
from model import Config, Protostar
from train import seed_cms


def _live(fm):
    with torch.no_grad():  # open the gates so updates are visible
        fm.gates.bias.fill_(2.0)


def test_slow_level_updates_once_per_period_across_calls():
    """CMS eq. 71: a chunk-64 level fed 16-token windows must hold its state
    for 3 calls and update on the 4th, not every call."""
    torch.manual_seed(0)
    fm = FastMemory(d=16, chunk=64, carry=True)
    _live(fm)
    fm.reset(1)
    states = []
    for _ in range(4):
        fm(torch.randn(1, 16, 16))
        states.append(fm._state.clone())
    assert torch.equal(states[0], states[2]), "slow level updated before its period"
    assert not torch.equal(states[2], states[3]), "slow level never updated"


def test_plain_memory_still_applies_partial_tail():
    """Regression: a 20-token one-shot write into a chunk-32 memory must
    still update it (the original semantics oneshot.py relies on)."""
    torch.manual_seed(0)
    fm = FastMemory(d=16, chunk=32)
    _live(fm)
    fm.reset(1)
    before = fm._state.clone()
    fm(torch.randn(1, 20, 16))
    assert not torch.equal(before, fm._state)


def test_bptt_carries_gradient_across_windows():
    """THE v2 flaw: a later window's loss must reach the earlier window's
    write. With bptt the gradient exists; without it, it is exactly zero."""
    for bptt, expect in [(True, True), (False, False)]:
        torch.manual_seed(0)
        fm = FastMemory(d=16, chunk=8, bptt=bptt, carry=True)
        _live(fm)
        fm.reset(1)
        h1 = torch.randn(1, 8, 16, requires_grad=True)
        fm(h1)
        out, _ = fm(torch.randn(1, 8, 16))
        out.sum().backward()
        has = h1.grad is not None and h1.grad.abs().sum() > 0
        assert has == expect, f"bptt={bptt}: cross-window grad present={has}"


def test_readback_aux_is_finite_and_trains_projections():
    torch.manual_seed(0)
    fm = FastMemory(d=16, chunk=8, bptt=True, readback=0.1, carry=True)
    _live(fm)
    fm.reset(2)
    fm(torch.randn(2, 16, 16))
    assert torch.isfinite(fm.last_aux) and fm.last_aux > 0
    fm.last_aux.backward()
    assert fm.wk.weight.grad.abs().sum() > 0


def test_reset_lanes_is_graph_safe_under_bptt():
    """In-place lane resets would raise during backward under bptt."""
    torch.manual_seed(0)
    fm = FastMemory(d=16, chunk=8, bptt=True, carry=True)
    _live(fm)
    fm.reset(2)
    out1, _ = fm(torch.randn(2, 8, 16))
    fm.reset_lanes(torch.tensor([True, False]))
    out2, _ = fm(torch.randn(2, 8, 16))
    (out1.sum() + out2.sum()).backward()


def test_seed_cms_copies_trained_memory_into_every_level():
    tiny = dict(d=32, n_layer=2, n_head=2, n_kv_head=1, fast_memory=True,
                fast_layer=1, seq_len=16, vocab=128)
    torch.manual_seed(0)
    v2 = Protostar(Config(**tiny))
    hope = Protostar(Config(**tiny, cms=[[8, 0.99, 0], [32, 0.999, 1]]))
    hope.load_state_dict(seed_cms(v2.state_dict(), hope.state_dict()))
    for lvl in hope.fast.levels:
        torch.testing.assert_close(lvl.wk.weight, v2.fast.wk.weight)
    hope.persist_memory = True
    x = torch.randint(0, 128, (2, 16))
    _, loss = hope(x, targets=x)
    loss.backward()
    assert torch.isfinite(loss)


def test_per_token_write_strength():
    """per_token: constant eta reproduces per-chunk eta.mean; a token whose
    own eta is 0 writes nothing (filler changed -> state unchanged)."""
    torch.manual_seed(0)
    h = torch.randn(1, 32, 16)
    a, b = FastMemory(d=16), FastMemory(d=16, per_token=True)
    b.load_state_dict(a.state_dict())
    for fm in (a, b):
        _live(fm)
        fm.reset(1)
        fm(h)
    assert torch.allclose(a._state, b._state, atol=1e-6)

    mask = (torch.arange(32) < 16).float().view(1, 32, 1)  # only first 16 tokens write
    orig = FastMemory.gate_values
    b.gate_values = lambda x: (lambda e, al, th: (e * mask, al, th))(*orig(b, x))
    outs = []
    for filler in (torch.randn(1, 16, 16), torch.randn(1, 16, 16)):
        b.reset(1)
        b(torch.cat([h[:, :16], filler], 1))
        outs.append(b._state.clone())
    assert torch.allclose(outs[0], outs[1], atol=1e-6), "zero-eta filler still wrote"
