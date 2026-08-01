import pytest
import torch

from accretion.substrates import SUBSTRATES, build_substrate


@pytest.mark.parametrize("kind", SUBSTRATES)
def test_zero_init_is_noop(kind):
    """Every substrate must start as an exact no-op — otherwise it perturbs a
    frozen backbone before it has learned anything."""
    torch.manual_seed(0)
    sub = build_substrate(kind, d=64, n_layer=12).eval()
    h = torch.randn(2, 16, 64)
    ids = torch.randint(0, 500, (2, 16))
    for layer in range(12):
        out = sub(h.clone(), ids, layer=layer)
        torch.testing.assert_close(out, h, atol=1e-6, rtol=0)


@pytest.mark.parametrize("kind", SUBSTRATES)
def test_writable_params_get_grads(kind):
    torch.manual_seed(0)
    sub = build_substrate(kind, d=64, n_layer=12)
    for p in sub.write_params():
        if p.ndim >= 2:
            p.data.normal_(0, 0.02)  # leave the no-op state
    h = torch.randn(2, 16, 64, requires_grad=True)
    ids = torch.randint(0, 500, (2, 16))
    total = h
    for layer in range(12):
        total = sub(total, ids, layer=layer)
    total.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in sub.write_params())


@pytest.mark.parametrize("kind", SUBSTRATES)
def test_capacity_reported(kind):
    assert build_substrate(kind, d=64, n_layer=12).n_slots() > 0
