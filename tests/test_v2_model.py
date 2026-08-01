"""v2: memory present during pretraining, not retrofitted."""
import torch

from model import Config, Protostar

TINY = dict(d=64, n_layer=6, n_head=4, n_kv_head=2, seq_len=64, vocab=512,
            fast_hidden=128, fast_chunk=16, fast_layer=3)


def test_init_perturbation_is_negligible():
    """v2 must start ~where the gdn backbone starts, so pretraining begins
    from a known-good point. NOT an exact no-op: zeroing the output
    projection would starve the memory's own projections of gradient (the
    bug that invalidated the product-key bake-off), so the branch starts
    tiny-but-alive instead."""
    torch.manual_seed(0)
    plain = Protostar(Config(**TINY)).eval()
    torch.manual_seed(0)
    v2 = Protostar(Config(**TINY, fast_memory=True)).eval()
    x = torch.randint(0, 512, (2, 32))
    with torch.no_grad():
        a, _ = plain(x)
        b, _ = v2(x)
    rel = (a - b).norm() / a.norm()
    assert rel < 1e-2, f"init perturbation too large: {rel:.4f}"


def test_fast_memory_state_changes_within_a_forward():
    """The inner loop must actually run inside the forward pass — checked via
    end-of-sequence state, which differs from the meta-learned init once the
    module has processed real content."""
    torch.manual_seed(0)
    m = Protostar(Config(**TINY, fast_memory=True))
    m(torch.randint(0, 512, (1, 64)))
    assert not torch.allclose(m.fast.snapshot()["state"], m.fast.m0.detach().unsqueeze(0)), \
        "memory state equals the untouched init — inner loop did not run"


def test_fast_memory_does_not_leak_across_independent_batches():
    """Each forward() is an unrelated training sequence; carrying state across
    them would silently corrupt pretraining (batch B learns from batch A)."""
    torch.manual_seed(0)
    m = Protostar(Config(**TINY, fast_memory=True))
    m(torch.randint(0, 512, (1, 64)))
    after_a = m.fast.snapshot()["state"].clone()
    torch.manual_seed(1)
    m(torch.randint(0, 512, (1, 64)))
    after_b = m.fast.snapshot()["state"].clone()
    torch.manual_seed(1)
    m(torch.randint(0, 512, (1, 64)))  # same seed as B, starting fresh again
    assert torch.allclose(after_b, m.fast.snapshot()["state"]), \
        "state leaked from a prior unrelated batch"


def test_gate_is_reported_for_monitoring():
    """M1's pass criteria include mean gate > 0.05 — it must be observable."""
    torch.manual_seed(0)
    m = Protostar(Config(**TINY, fast_memory=True))
    m(torch.randint(0, 512, (1, 64)))
    assert 0.0 <= m.last_gate <= 1.0 and m.last_gate > 0


def test_causality_holds_with_fast_memory():
    torch.manual_seed(0)
    m = Protostar(Config(**TINY, fast_memory=True)).eval()
    x = torch.randint(0, 512, (1, 64))
    with torch.no_grad():
        m.fast.reset()
        base, _ = m(x)
        x2 = x.clone()
        x2[0, 40] = (x2[0, 40] + 1) % 512
        m.fast.reset()
        pert, _ = m(x2)
    assert torch.allclose(base[0, :40], pert[0, :40], atol=1e-4), "future leaked"


def test_param_count_still_in_class():
    n = sum(p.numel() for p in Protostar(Config(fast_memory=True)).parameters())
    assert 120e6 < n < 155e6, f"{n/1e6:.1f}M outside the 125M class"


def test_backward_reaches_memory_projections():
    """Outer optimizer must train the memory's projections and gates."""
    torch.manual_seed(0)
    m = Protostar(Config(**TINY, fast_memory=True))
    x = torch.randint(0, 512, (1, 64))
    _, loss = m(x[:, :-1], targets=x[:, 1:])
    loss.backward()
    assert m.fast.wq.weight.grad is not None and m.fast.wq.weight.grad.abs().sum() > 0
    assert m.fast_gate.weight.grad is not None
