import torch

from model import Config, Protostar

TINY = dict(d=128, n_layer=4, n_head=4, n_kv_head=2, seq_len=64, vocab=512)


def test_param_count_default():
    n = sum(p.numel() for p in Protostar(Config()).parameters())
    assert 120e6 < n < 140e6, f"{n / 1e6:.1f}M outside 125M class"


def test_forward_shape():
    cfg = Config(**TINY)
    logits, loss = Protostar(cfg)(torch.randint(0, 512, (2, 64)), targets=torch.randint(0, 512, (2, 64)))
    assert logits.shape == (2, 64, 512)
    assert loss.ndim == 0 and loss.item() > 0


def test_causality():
    cfg = Config(**TINY)
    model = Protostar(cfg).eval()
    x = torch.randint(0, 512, (1, 64))
    with torch.no_grad():
        base = model(x)[0]
        x2 = x.clone()
        x2[0, 40] = (x2[0, 40] + 1) % 512
        pert = model(x2)[0]
    assert torch.allclose(base[0, :40], pert[0, :40], atol=1e-5), "future token leaked into past logits"
    assert not torch.allclose(base[0, 40:], pert[0, 40:], atol=1e-5)


def test_tied_embeddings():
    model = Protostar(Config(**TINY))
    assert model.lm_head.weight is model.embed.weight
