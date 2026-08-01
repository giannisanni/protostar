import torch

from accretion.fast import FastMemory


def test_surprise_falls_on_repetition():
    """The inner loop must actually learn within a context — that IS the claim."""
    torch.manual_seed(0)
    fm = FastMemory(d=64, hidden=128, chunk=16)
    h = torch.randn(1, 64, 64)
    _, first = fm.update(h)
    for _ in range(15):
        fm.update(h)
    _, last = fm.update(h)
    assert last < first * 0.8, f"surprise did not fall: {first:.4f} -> {last:.4f}"


def test_reset_restores_learned_init_not_zeros():
    """TTT: resetting to a META-LEARNED W0 is what keeps training stable."""
    torch.manual_seed(0)
    fm = FastMemory(d=64, hidden=128, chunk=16)
    assert isinstance(fm.w0_1, torch.nn.Parameter)
    assert fm.w0_1.abs().sum() > 0, "w0 must be learned, not zeros"
    fm.update(torch.randn(1, 32, 64))
    fm.reset()
    torch.testing.assert_close(fm._w1, fm.w0_1.detach())


def test_gates_are_data_dependent():
    """All four papers converge on data-dependent inner LR. A constant-gate
    implementation (v1's mistake) fails this test."""
    torch.manual_seed(0)
    fm = FastMemory(d=64, hidden=128)
    with torch.no_grad():  # leave the zero-init so gates can differ
        fm.gates.weight.normal_(0, 0.5)
    a = fm.gate_values(torch.randn(1, 8, 64))
    b = fm.gate_values(torch.randn(1, 8, 64))
    assert not torch.allclose(a[0], b[0]), "eta identical for different inputs"
    assert not torch.allclose(a[1], b[1]), "alpha identical for different inputs"


def test_snapshot_roundtrip():
    """Sleep needs to buffer memory state and restore it later."""
    torch.manual_seed(0)
    fm = FastMemory(d=64, hidden=128, chunk=16)
    fm.update(torch.randn(1, 32, 64))
    snap = fm.snapshot()
    fm.reset()
    fm.load(snap)
    torch.testing.assert_close(fm._w1, snap["w1"])
    torch.testing.assert_close(fm._w2, snap["w2"])


def test_projections_receive_outer_gradient():
    """Outer optimizer must train wq/wk/wv/gates even though the inner
    trajectory is detached."""
    torch.manual_seed(0)
    fm = FastMemory(d=64, hidden=128, chunk=16)
    out, _ = fm.update(torch.randn(1, 32, 64))
    out.pow(2).mean().backward()
    assert fm.wq.weight.grad is not None and fm.wq.weight.grad.abs().sum() > 0
