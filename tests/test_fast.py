import torch

from accretion.fast import FastMemory


def test_surprise_falls_on_repetition():
    """The inner loop must actually learn within a context — that IS the claim."""
    torch.manual_seed(0)
    fm = FastMemory(d=64, chunk=16)
    h = torch.randn(1, 64, 64)
    _, first = fm(h)
    for _ in range(15):
        fm(h)
    _, last = fm(h)
    assert last < first * 0.8, f"surprise did not fall: {first:.4f} -> {last:.4f}"


def test_initial_state_is_a_learnable_parameter():
    """TTT: the state we reset to must be META-LEARNED, not a constant."""
    fm = FastMemory(d=64, chunk=16)
    assert isinstance(fm.m0, torch.nn.Parameter) and fm.m0.requires_grad


def test_gates_are_data_dependent():
    """All four papers converge on data-dependent inner LR. A constant-gate
    implementation (v1's mistake) fails this test."""
    torch.manual_seed(0)
    fm = FastMemory(d=64)
    with torch.no_grad():  # leave the zero-init so gates can differ
        fm.gates.weight.normal_(0, 0.5)
    a = fm.gate_values(torch.randn(1, 8, 64))
    b = fm.gate_values(torch.randn(1, 8, 64))
    assert not torch.allclose(a[0], b[0]), "eta identical for different inputs"
    assert not torch.allclose(a[1], b[1]), "alpha identical for different inputs"


def test_snapshot_roundtrip():
    """Sleep needs to buffer memory state and restore it later."""
    torch.manual_seed(0)
    fm = FastMemory(d=64, chunk=16)
    fm(torch.randn(1, 32, 64))
    snap = fm.snapshot()
    fm.reset(1)
    fm.load(snap)
    torch.testing.assert_close(fm._state, snap["state"])


def test_all_projections_and_gates_receive_gradient():
    """The regression test for the DDP unused-parameter crash: wk/wv/wq/gates
    must be trained by the outer loop, or the data-dependent gating — the one
    thing all four papers converge on — stays frozen at initialization."""
    torch.manual_seed(0)
    fm = FastMemory(d=64, chunk=16)
    out, _ = fm(torch.randn(2, 48, 64))
    out.pow(2).mean().backward()
    for name in ("wk", "wv", "wq"):
        g = getattr(fm, name).weight.grad
        assert g is not None and g.abs().sum() > 0, f"{name} got no gradient"
    assert fm.gates.weight.grad is not None and fm.gates.weight.grad.abs().sum() > 0
    assert fm.m0.grad is not None, "meta-learned init got no gradient"
