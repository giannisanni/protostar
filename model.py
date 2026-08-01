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
    # ponytail: sink-KV via attn_mask forces the math SDPA backend and OOMs 16GB
    # at B=16/T=2048; default off until reimplemented with flex_attention
    # (rescale trick: out * sigmoid(lse - sink_logit)). Ablation-queue item.
    attn_sink: bool = False  # learned per-head sink KV (memory-hungry path)
    qk_norm: bool = True
    gdn_ratio: int = 0  # 0=dense; 4 = 3:1 GDN:attn (Task 5)
    mtp: bool = False  # Task 6: depth-1 MTP head, loss += 0.3 * ce(t+2)
    engram: bool = False  # Task 7: bigram-hash memory tables
    engram_layer: int = 8  # insert memory after this block
    # --- v2: memory-native (trained jointly, NOT retrofitted) ---
    fast_memory: bool = False  # Titans/TTT inner loop live during pretraining
    fast_hidden: int = 2048
    fast_chunk: int = 32
    fast_layer: int = 12  # mid-stack: enough context to be worth memorizing
    mem_integration: str = "mag"  # gated parallel branch (Titans MAG > MAL)
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
        if not cfg.attn_sink:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            return self.wo(out.transpose(1, 2).reshape(B, T, -1))
        n_sink = 1
        if n_sink:
            k = torch.cat([self.sink_k.expand(B, -1, -1, -1).reshape(B, cfg.n_kv_head, 1, cfg.d_head), k], dim=2)
            v = torch.cat([self.sink_v.expand(B, -1, -1, -1).reshape(B, cfg.n_kv_head, 1, cfg.d_head), v], dim=2)
        # causal mask with sink column always visible
        mask = torch.ones(T, T + n_sink, dtype=torch.bool, device=x.device).tril_(diagonal=n_sink)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, cfg: Config, mult=None):
        super().__init__()
        mult = mult or cfg.ffn_mult
        self.up = nn.Linear(cfg.d, mult * cfg.d, bias=False)
        self.down = nn.Linear(mult * cfg.d, cfg.d, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, x):
        return self.down(F.relu(self.up(x)).square())


class Block(nn.Module):
    def __init__(self, cfg: Config, idx: int):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(cfg.d), nn.RMSNorm(cfg.d)
        use_rope = (idx + 1) % cfg.rope_ratio != 0  # 3 RoPE : 1 NoPE
        if cfg.gdn_ratio and (idx + 1) % cfg.gdn_ratio != 0:
            from fla.layers import GatedDeltaNet  # lazy: only needed when flag on

            # expand_v=1 + ffn_mult-1 keeps GDN layers at param parity with dense
            self.mix = GatedDeltaNet(hidden_size=cfg.d, head_dim=cfg.d_head,
                                     num_heads=cfg.n_head, expand_v=1, mode="chunk")
            self.is_gdn = True
            self.mlp = MLP(cfg, mult=cfg.ffn_mult - 1)
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
        if cfg.mtp:
            self.mtp_merge = nn.Linear(2 * cfg.d, cfg.d, bias=False)
            self.mtp_norm = nn.RMSNorm(cfg.d)
            self.mtp_mlp = MLP(cfg)
        if cfg.fast_memory:
            from accretion.fast import FastMemory

            self.fast = FastMemory(cfg.d, hidden=cfg.fast_hidden, chunk=cfg.fast_chunk)
            # MAG: memory is a GATED PARALLEL BRANCH, not an in-line residual add.
            # Titans measured MAG > MAL; v1 accidentally used MAL-shaped injection.
            self.fast_gate = nn.Linear(cfg.d, 1)
            self.fast_proj = nn.Linear(cfg.d, cfg.d, bias=False)
            # NOT zero-init: zeroing this kills the gradient to the memory's
            # own projections (wq/wk/wv), so they never learn — the same
            # chicken-and-egg that invalidated the product-key bake-off.
            # Tiny random keeps the perturbation ~0.1% of h while letting
            # gradient reach the memory. Gate bias starts negative so the
            # branch is near-silent until the model decides it is useful.
            nn.init.normal_(self.fast_proj.weight, std=1e-3)
            nn.init.constant_(self.fast_gate.bias, -2.0)  # sigmoid(-2) ~ 0.12
            self.last_gate = 0.0  # logged: proves the model USES memory
        if cfg.engram:
            # ponytail: 2 hash heads x 2^18 buckets x 64d; params are sparse-access
            # (Engram's point: capacity without FLOPs). Scalar trust gate; per-channel
            # gating is an ablation if this arm wins.
            self.tables = nn.ModuleList([nn.Embedding(1 << 18, 64) for _ in range(2)])
            self.gate = nn.Linear(cfg.d, 1)
            self.mem_proj = nn.Linear(128, cfg.d, bias=False)
            nn.init.zeros_(self.mem_proj.weight)  # step-0 no-op by construction
        cos, sin = rope_freqs(cfg, torch.device("cpu"))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    # chunk rows for the CE pass. 8192 x 50304 fp32 = 1.5GB peak, which OOMs
    # the engram arm on 16GB cards; 4096 halves it. Mathematically identical
    # (the sum is the same), costs a little speed.
    CE_CHUNK = 4096

    def _ce(self, flat_h, flat_t):
        # chunked CE with per-chunk head recompute: full (B*T, vocab) logits
        # never materialize (1.6GB+ at B=8/T=2048 otherwise)
        loss = flat_h.new_zeros(())
        for hc, tc in zip(flat_h.split(self.CE_CHUNK), flat_t.split(self.CE_CHUNK)):
            loss = loss + torch.utils.checkpoint.checkpoint(
                lambda a, b: F.cross_entropy(self.lm_head(a), b, reduction="sum"),
                hc, tc, use_reentrant=False)
        return loss / flat_t.numel()

    def forward(self, idx, targets=None):
        h = self.embed(idx)
        for i, blk in enumerate(self.blocks):
            h = blk(h, self.cos, self.sin)
            if self.cfg.fast_memory and i == self.cfg.fast_layer:
                # each forward() is an independent training sequence — the
                # inner state must not leak across unrelated batches. Only
                # a stateful CALLER (accretion inference) should carry it on.
                self.fast.reset(h.size(0), h.device, h.dtype)
                retrieved, surprise = self.fast.update(h)
                g = torch.sigmoid(self.fast_gate(h))
                self.last_gate = float(g.mean().detach())
                self.last_surprise = surprise
                h = h + g * self.fast_proj(retrieved)
            if self.cfg.engram and i == self.cfg.engram_layer:
                prev = F.pad(idx[:, :-1], (1, 0))
                mem = torch.cat([self.tables[0]((prev * 1000003 + idx) % (1 << 18)),
                                 self.tables[1]((prev * 31 + idx * 999983) % (1 << 18))], -1)
                h = h + torch.sigmoid(self.gate(h)) * self.mem_proj(mem)
        h = self.norm_f(h)
        if targets is None:
            return self.lm_head(h), None
        loss = self._ce(h.view(-1, self.cfg.d), targets.reshape(-1))
        if self.cfg.mtp:
            # predict t+2: merge h_t with embedding of the (known) next token
            m = self.mtp_norm(self.mtp_merge(torch.cat([h[:, :-1], self.embed(idx[:, 1:])], -1)))
            m = m + self.mtp_mlp(m)
            loss = loss + 0.3 * self._ce(m.reshape(-1, self.cfg.d), targets[:, 1:].reshape(-1))
        return None, loss
