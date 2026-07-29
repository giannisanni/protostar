"""protostar: 125M-class decoder. One file; every recipe piece is a Config flag.

Baseline (all extras off): deep-thin decoder, RMSNorm pre-norm, ReLU^2 MLP,
GQA with QK-norm + learned attention sink, RoPE with every rope_ratio-th
layer NoPE (iRoPE), tied embeddings.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 50304  # GPT-2 padded to /128
    d: int = 640
    n_layer: int = 24  # deep-thin (MobileLLM)
    n_head: int = 10
    n_kv_head: int = 2  # GQA
    ffn_mult: int = 4  # ReLU^2 MLP (speedrun)
    rope_ratio: int = 4  # every 4th layer NoPE (iRoPE)
    attn_sink: bool = True  # learned per-head sink KV
    qk_norm: bool = True
    gdn_ratio: int = 0  # 0=dense; 4 = 3:1 GDN:attn (Task 5)
    swa_fallback: bool = False  # sliding-window instead of GDN if fla broken
    mtp: bool = False  # Task 6
    engram: bool = False  # Task 7
    seq_len: int = 2048

    @property
    def d_head(self):
        return self.d // self.n_head


def rope_freqs(cfg: Config, device):
    inv = 1.0 / (10000 ** (torch.arange(0, cfg.d_head, 2, device=device) / cfg.d_head))
    t = torch.arange(cfg.seq_len, device=device)
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):  # x: (B, h, T, d_head)
    T = x.size(2)
    c, s = cos[:T], sin[:T]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Config, use_rope: bool):
        super().__init__()
        self.cfg, self.use_rope = cfg, use_rope
        d, dh, nh, nkv = cfg.d, cfg.d_head, cfg.n_head, cfg.n_kv_head
        self.wq = nn.Linear(d, nh * dh, bias=False)
        self.wk = nn.Linear(d, nkv * dh, bias=False)
        self.wv = nn.Linear(d, nkv * dh, bias=False)
        self.wo = nn.Linear(nh * dh, d, bias=False)
        nn.init.zeros_(self.wo.weight)  # zero-init out proj (speedrun stability)
        if cfg.attn_sink:
            self.sink_k = nn.Parameter(torch.randn(nkv, 1, dh) * 0.02)
            self.sink_v = nn.Parameter(torch.zeros(nkv, 1, dh))

    def forward(self, x, cos, sin):
        cfg = self.cfg
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, cfg.n_head, cfg.d_head).transpose(1, 2)
        k = self.wk(x).view(B, T, cfg.n_kv_head, cfg.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, cfg.n_kv_head, cfg.d_head).transpose(1, 2)
        if cfg.qk_norm:
            q, k = F.rms_norm(q, (cfg.d_head,)), F.rms_norm(k, (cfg.d_head,))
        if self.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        n_sink = 0
        if cfg.attn_sink:
            n_sink = 1
            k = torch.cat([self.sink_k.expand(B, -1, -1, -1).reshape(B, cfg.n_kv_head, 1, cfg.d_head), k], dim=2)
            v = torch.cat([self.sink_v.expand(B, -1, -1, -1).reshape(B, cfg.n_kv_head, 1, cfg.d_head), v], dim=2)
        # causal mask with sink column always visible
        mask = torch.ones(T, T + n_sink, dtype=torch.bool, device=x.device).tril_(diagonal=n_sink)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.up = nn.Linear(cfg.d, cfg.ffn_mult * cfg.d, bias=False)
        self.down = nn.Linear(cfg.ffn_mult * cfg.d, cfg.d, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, x):
        return self.down(F.relu(self.up(x)).square())


class Block(nn.Module):
    def __init__(self, cfg: Config, idx: int):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(cfg.d), nn.RMSNorm(cfg.d)
        use_rope = (idx + 1) % cfg.rope_ratio != 0  # 3 RoPE : 1 NoPE
        if cfg.gdn_ratio and (idx + 1) % cfg.gdn_ratio != 0:
            if cfg.swa_fallback:
                self.mix = Attention(cfg, use_rope=True)  # ponytail: swa TODO Task 5 fallback
            else:
                from fla.layers import GatedDeltaNet  # lazy: only needed when flag on

                self.mix = GatedDeltaNet(hidden_size=cfg.d, num_heads=cfg.n_head // 2, mode="chunk")
            self.is_gdn = True
        else:
            self.mix = Attention(cfg, use_rope)
            self.is_gdn = False
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        if self.is_gdn:
            out = self.mix(self.n1(x))
            x = x + (out[0] if isinstance(out, tuple) else out)
        else:
            x = x + self.mix(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class Protostar(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.norm_f = nn.RMSNorm(cfg.d)
        self.lm_head = nn.Linear(cfg.d, cfg.vocab, bias=False)
        self.lm_head.weight = self.embed.weight  # tied
        cos, sin = rope_freqs(cfg, torch.device("cpu"))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        h = self.embed(idx)
        for blk in self.blocks:
            h = blk(h, self.cos, self.sin)
        logits = self.lm_head(self.norm_f(h))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab), targets.reshape(-1))
        return logits, loss
