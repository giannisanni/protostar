import pytest
import torch

from accretion.substrates import SUBSTRATES, build_substrate


@pytest.mark.parametrize("kind", SUBSTRATES)
def test_init_is_noop_or_negligible(kind):
    """A substrate must not perturb a frozen backbone before it has learned.
    engram/ple are EXACT no-ops (zero-init projections). pk cannot be: zero
    values starve its keys of gradient, so it uses tiny random values instead
    — perturbation must still be negligible relative to the hidden state."""
    torch.manual_seed(0)
    sub = build_substrate(kind, d=64, n_layer=12).eval()
    h = torch.randn(2, 16, 64)
    ids = torch.randint(0, 500, (2, 16))
    for layer in range(12):
        out = sub(h.clone(), ids, layer=layer)
        rel = (out - h).norm() / h.norm()
        assert rel < 1e-2, f"{kind} layer {layer}: {rel:.4f} perturbation at init"


@pytest.mark.parametrize("kind", SUBSTRATES)
def test_keys_receive_gradient(kind):
    """The bug that invalidated the first pk test: zero-init values mean the
    softmax weights multiply zeros, so retrieval keys never get gradient."""
    torch.manual_seed(0)
    sub = build_substrate(kind, d=64, n_layer=12)
    h = torch.randn(2, 16, 64)
    ids = torch.randint(0, 500, (2, 16))
    out = h
    for layer in range(12):
        out = sub(out, ids, layer=layer)
    out.pow(2).sum().backward()
    grads = [p.grad.abs().sum().item() for p in sub.write_params() if p.grad is not None]
    assert grads and max(grads) > 0, f"{kind}: no write_param received gradient"


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
