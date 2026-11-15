"""XL cache: feeding window 1 then window 2 must equal one pass over both
(dense model, RoPE offsets included); lane resets hide the cache."""
import torch

from model import Config, Protostar
from train import seed_cms

KW = dict(d=64, n_layer=4, n_head=4, n_kv_head=2, vocab=128, seq_len=64, fast_memory=True)


def _model(**kw):
    torch.manual_seed(0)
    m = Protostar(Config(**{**KW, **kw})).eval()
    for p in m.parameters():  # zero-inits would make the check vacuous
        torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else None
    return m


def test_two_windows_equal_one_pass():
    m = _model(mem="xl", xl_size=16)
    x = torch.randint(0, 128, (2, 32))
    with torch.no_grad():
        full, _ = m(x)  # persist off: one 32-token pass, cache reset first
        m.persist_memory = True
        m.fast.reset(2)
        m(x[:, :16])
        second, _ = m(x[:, 16:])
    assert torch.allclose(second, full[:, 16:], atol=1e-4), (second - full[:, 16:]).abs().max()


def test_lane_reset_hides_cache():
    m = _model(mem="xl", xl_size=16)
    x = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        m.persist_memory = True
        m.fast.reset(2)
        m(x)
        m.fast.reset_lanes(torch.tensor([True, False]))
        a, _ = m(x)
        m.persist_memory = False
        fresh, _ = m(x)
    assert torch.allclose(a[0], fresh[0], atol=1e-5) and not torch.allclose(a[1], fresh[1], atol=1e-3)


def test_xl_seeds_from_ttt_checkpoint():
    src = Protostar(Config(**KW)).state_dict()
    dst = Protostar(Config(**KW, mem="xl"))
    dst.load_state_dict(seed_cms(src, dst.state_dict()))
