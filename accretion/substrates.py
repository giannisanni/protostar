"""Three candidate persistent memories behind one interface.

M0's Engram-lite null is explained (content-blind gate); this module gives
Phase 1 the REAL Engram recipe plus two rivals, so the bake-off asks the
right question — factual recall — of three designs at published strength.
All are zero-init no-ops so a frozen backbone is undisturbed at step 0.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SUBSTRATES = ("engram", "ple", "pk")


class _Base(nn.Module):
    def write_params(self):
        """Params sleep/bake-off training is allowed to modify."""
        raise NotImplementedError

    def n_slots(self):
        raise NotImplementedError


class Engram(_Base):
    """Real-recipe Engram (deep-read round 3), scaled to a 125M backbone:

    - 2-gram AND 3-gram tables, multiple hash heads per order
    - ALIGNMENT gate: sigma(RMSNorm(h)^T RMSNorm(W_K e) / sqrt(d)) — the gate
      sees what was retrieved (M0's lite version was content-blind and closed)
    - value path: W_V + depthwise causal conv (k=4, dilation=max n) + SiLU
    - dual placement: one early layer + one mid layer
    - conv zero-init => exact no-op at start
    Omitted vs paper: tokenizer canonicalization (NFKC/lowercase id merge) —
    ponytail: nontrivial for GPT-2 ids, note as a known minus.
    """

    def __init__(self, d, n_layer, buckets=1 << 16, heads=4, dim=64, sites=None):
        super().__init__()
        self.sites = tuple(sites) if sites else (max(1, n_layer // 12), n_layer // 2)
        self.buckets, self.heads = buckets, heads
        # heads tables per n-gram order (2-gram and 3-gram)
        self.tables = nn.ModuleList(
            [nn.Embedding(buckets, dim) for _ in range(2 * heads)])
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.02)
        d_mem = 2 * heads * dim
        self.wk = nn.ModuleDict({str(s): nn.Linear(d_mem, d, bias=False) for s in self.sites})
        self.wv = nn.Linear(d_mem, d, bias=False)
        self.conv = nn.Conv1d(d, d, kernel_size=4, padding=3, groups=d)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        nn.init.zeros_(self.wv.weight)  # belt and braces: no-op at init

    # multiplicative-XOR-ish hashes over 2- and 3-gram suffixes
    _P = (1000003, 999983, 786433, 655360001, 909091, 65599, 100003, 31)

    def _mem(self, ids):
        p1 = F.pad(ids[:, :-1], (1, 0))
        p2 = F.pad(ids[:, :-2], (2, 0)) if ids.size(1) > 1 else torch.zeros_like(ids)
        outs = []
        for h in range(self.heads):
            a, b = self._P[h % len(self._P)], self._P[(h + 3) % len(self._P)]
            outs.append(self.tables[h]((p1 * a + ids * b) % self.buckets))
            outs.append(self.tables[self.heads + h](
                (p2 * a + p1 * b + ids * self._P[(h + 5) % len(self._P)]) % self.buckets))
        return torch.cat(outs, -1)

    def forward(self, h, ids, layer):
        if layer not in self.sites:
            return h
        e = self._mem(ids)
        k = self.wk[str(layer)](e)
        gate = torch.sigmoid(
            (F.rms_norm(h, h.shape[-1:]) * F.rms_norm(k, k.shape[-1:])).sum(-1, keepdim=True)
            / (h.shape[-1] ** 0.5))
        v = gate * self.wv(e)
        y = F.silu(self.conv(v.transpose(1, 2))[..., : v.size(1)].transpose(1, 2)) + v
        return h + y

    def write_params(self):
        return (list(self.tables.parameters()) + list(self.wv.parameters())
                + list(self.wk.parameters()) + list(self.conv.parameters()))

    def n_slots(self):
        return self.buckets * 2 * self.heads


class PLE(_Base):
    """Gemma-style per-layer embeddings: one lookup per token PER LAYER, so the
    memory can speak at every depth instead of one insertion point."""

    def __init__(self, d, n_layer, vocab=50304, dim=32):
        super().__init__()
        self.table = nn.Embedding(vocab, n_layer * dim)
        nn.init.normal_(self.table.weight, std=0.02)
        self.dim, self.n_layer = dim, n_layer
        self.proj = nn.ModuleList([nn.Linear(dim, d, bias=False) for _ in range(n_layer)])
        for p in self.proj:
            nn.init.zeros_(p.weight)

    def forward(self, h, ids, layer):
        e = self.table(ids).view(*ids.shape, self.n_layer, self.dim)[..., layer, :]
        return h + self.proj[layer](e)

    def write_params(self):
        return [self.table.weight] + [p.weight for p in self.proj]

    def n_slots(self):
        return self.table.num_embeddings


class ProductKey(_Base):
    """Product-key memory: LEARNED keys, so retrieval is semantic rather than
    exact-token. Two half-key codebooks give |C|^2 slots at 2|C| comparisons —
    the fix for hash lookup's inability to match a paraphrase."""

    def __init__(self, d, n_layer, c=256, dim=64, topk=8, at=None):
        super().__init__()
        self.at = at if at is not None else n_layer // 3
        self.c, self.topk = c, topk
        self.q = nn.Linear(d, 2 * dim, bias=False)
        self.keys = nn.Parameter(torch.randn(2, c, dim) * 0.02)
        self.values = nn.Embedding(c * c, d)
        nn.init.zeros_(self.values.weight)  # no-op at init

    def forward(self, h, ids, layer):
        if layer != self.at:
            return h
        q = self.q(h).chunk(2, dim=-1)
        s0 = torch.einsum("btd,cd->btc", q[0], self.keys[0])
        s1 = torch.einsum("btd,cd->btc", q[1], self.keys[1])
        t0, i0 = s0.topk(self.topk, -1)
        t1, i1 = s1.topk(self.topk, -1)
        cand = (t0.unsqueeze(-1) + t1.unsqueeze(-2)).flatten(-2)
        idx = (i0.unsqueeze(-1) * self.c + i1.unsqueeze(-2)).flatten(-2)
        w, sel = cand.topk(self.topk, -1)
        w = F.softmax(w, -1)
        vals = self.values(torch.gather(idx, -1, sel))
        return h + (w.unsqueeze(-1) * vals).sum(-2)

    def write_params(self):
        return [self.values.weight, self.keys]

    def n_slots(self):
        return self.c * self.c


def build_substrate(kind, d, n_layer, **kw):
    return {"engram": Engram, "ple": PLE, "pk": ProductKey}[kind](d, n_layer, **kw)
