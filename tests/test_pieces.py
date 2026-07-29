"""Tasks 5-7: GDN hybrid, MTP head, Engram-lite. GDN tests need fla (substrate)."""
import pytest
import torch

from model import Config, Protostar

TINY = dict(d=128, n_layer=4, n_head=4, n_kv_head=2, seq_len=64, vocab=512)


# --- Task 5: GDN ---

def test_gdn_param_parity_full_size():
    pytest.importorskip("fla")
    n = sum(p.numel() for p in Protostar(Config(gdn_ratio=4)).parameters())
    assert 120e6 < n < 145e6, f"{n / 1e6:.1f}M — GDN arm out of 125M class"


def test_gdn_forward_and_causality():
    pytest.importorskip("fla")
    if not torch.cuda.is_available():
        pytest.skip("fla kernels need a GPU")
    dev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    model = Protostar(Config(**TINY, gdn_ratio=4)).to(dev).eval()
    x = torch.randint(0, 512, (1, 64), device=dev)
    with torch.no_grad():
        base = model(x)[0]
        x2 = x.clone()
        x2[0, 40] = (x2[0, 40] + 1) % 512
        pert = model(x2)[0]
    assert torch.allclose(base[0, :40], pert[0, :40], atol=1e-4), "GDN leaked future"


# --- Task 6: MTP ---

def test_mtp_loss_and_backward():
    model = Protostar(Config(**TINY, mtp=True))
    x, t = torch.randint(0, 512, (2, 64)), torch.randint(0, 512, (2, 64))
    _, loss = model(x, targets=t)
    loss.backward()
    assert loss.item() > 0
    assert model.mtp_merge.weight.grad is not None


def test_mtp_off_no_extra_params():
    a = sum(p.numel() for p in Protostar(Config(**TINY)).parameters())
    b = sum(p.numel() for p in Protostar(Config(**TINY, mtp=True)).parameters())
    assert b > a


# --- Task 7: Engram ---

def test_engram_noop_at_init():
    torch.manual_seed(0)
    model = Protostar(Config(**TINY, engram=True, engram_layer=1)).eval()
    x = torch.randint(0, 512, (2, 64))
    with torch.no_grad():
        base = model(x)[0]
        for tbl in model.tables:  # scramble tables: zero-init proj must make this a no-op
            tbl.weight.copy_(torch.randn_like(tbl.weight) * 100)
        scrambled = model(x)[0]
    torch.testing.assert_close(base, scrambled)


def test_engram_learns():
    model = Protostar(Config(**TINY, engram=True, engram_layer=1))
    x, t = torch.randint(0, 512, (2, 64)), torch.randint(0, 512, (2, 64))
    _, loss = model(x, targets=t)
    loss.backward()
    assert model.mem_proj.weight.grad is not None
    assert any(tbl.weight.grad is not None for tbl in model.tables)
