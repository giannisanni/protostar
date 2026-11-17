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


def test_xl_bias_shifts_only_cached_logits():
    m = _model(mem="xl", xl_size=16, xl_bias=True)
    x = torch.randint(0, 128, (1, 32))
    with torch.no_grad():
        full, _ = m(x)
        m.persist_memory = True
        m.fast.reset(1)
        m(x[:, :16])
        same, _ = m(x[:, 16:])  # bias 0 == no bias
        for blk in m.blocks:
            if hasattr(blk.mix, "xl_bias"):
                blk.mix.xl_bias.fill_(-50.0)  # cache effectively invisible
        m.fast.reset(1)
        m(x[:, :16])
        hidden, _ = m(x[:, 16:])
        m.persist_memory = False
        alone, _ = m(x[:, 16:])
    assert torch.allclose(same, full[:, 16:], atol=1e-4)
    assert torch.allclose(hidden, alone, atol=1e-4)


def test_wrap_and_density_defaults_are_bit_identical_and_knobs_work():
    import tiktoken
    from accretion.curriculum import FactStream
    enc = tiktoken.get_encoding("gpt2")
    a, b = FactStream(enc, 512, 7), FactStream(enc, 512, 7, density=0.15, wrap=0.0)
    assert [a.next_window() for _ in range(10)] == [b.next_window() for _ in range(10)]
    d = FactStream(enc, 512, 7, density=0.3, wrap=0.5)
    for _ in range(50):
        d.next_window()
    n_state = sum(k == "state" for k, *_ in d.log)
    assert n_state / 50 > 9, n_state / 50


def test_xl_cms_seeds_levels_from_v2_memory():
    src = Protostar(Config(**KW)).state_dict()  # v2: single fast memory
    dst = Protostar(Config(**KW, mem="xl", cms=[[8, 0.99, 0], [32, 0.999, 1]]))
    sd = seed_cms(src, dst.state_dict())
    dst.load_state_dict(sd)  # strict: every key accounted for
    assert torch.equal(sd["fast.cms.levels.1.wk.weight"], src["fast.wk.weight"])
    assert torch.equal(sd["fast_proj.weight"], src["fast_proj.weight"])


def test_xl_cms_carries_what_the_cache_dropped():
    """Clear the cache before window 3: only the CMS tier can still carry
    windows 1-2 into it. (Through the cache alone window 1 can leak, since
    window 2's cached upper-layer K/V attended to it: XL recurrence.)"""
    x = torch.randint(0, 128, (2, 48))
    alt = x.clone()
    alt[:, :16] = torch.randint(0, 128, (2, 16))  # different window 1, same 2 and 3
    for cms, should_differ in ((None, False), ([[8, 0.99, 0], [32, 0.999, 1]], True)):
        m = _model(mem="xl", xl_size=16, cms=cms, fast_layer=2)
        outs = []
        for seq in (x, alt):
            m.persist_memory = True
            m.fast.reset(2)
            with torch.no_grad():
                for w in range(3):
                    if w == 2:
                        (m.fast.xl if cms else m.fast).reset(2)
                    o, _ = m(seq[:, 16 * w:16 * (w + 1)])
            outs.append(o)
        assert torch.allclose(outs[0], outs[1], atol=1e-5) != should_differ, cms
