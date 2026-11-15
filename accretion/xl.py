"""Transformer-XL cache as the fast tier: every attention layer also attends
to the previous window's keys/values (cached raw, i.e. after qk-norm, before
RoPE), placed just before the current window.

Why: rounds 1-4 showed every NEW memory module fails at addressing (slot
attention mass on the fact 3.9% vs 2.7% chance) while the backbone's own
attention finds the same fact at 89-95% in context. This reuses that circuit
instead of asking a fresh module to learn retrieval from sparse probes.

Same interface as FastMemory so trainer/currprobe/ceiling work unchanged.
ponytail: one FIFO of `size` tokens per layer; ages beyond it are the
sleep tier's job.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class XLCache(nn.Module):
    def __init__(self, size=512):
        super().__init__()
        self.size = size
        self.kv, self.ok = {}, None  # id(attn) -> (k, v); ok (B, M) bool
        self.last_aux = 0.0

    def reset(self, batch=1, device=None, dtype=None):
        self.kv, self.ok, self._batch, self._dev = {}, None, batch, device

    def reset_lanes(self, mask):
        if self.ok is not None and mask.any():
            self.ok = self.ok & ~mask.to(self.ok.device).view(-1, 1)

    def detach_(self):
        self.kv = {i: (k.detach(), v.detach()) for i, (k, v) in self.kv.items()}

    def attend(self, attn, q, k, v, cos, sin):
        """q (B,h,T,d), k/v (B,kv,T,d) raw. Reads the cache, then writes this
        window into it. The cache's validity mask advances once per window."""
        from model import apply_rope
        B, T = q.size(0), q.size(2)
        if self.ok is not None and self.ok.size(0) != B:
            if self.ok.size(0) == 1:  # broadcast one stored lane (eval)
                self.kv = {i: (a.expand(B, *a.shape[1:]), b.expand(B, *b.shape[1:]))
                           for i, (a, b) in self.kv.items()}
                self.ok = self.ok.expand(B, -1)
            else:
                self.reset(B)
        kc, vc = self.kv.get(id(attn), (k[:, :, :0], v[:, :, :0]))
        M = kc.size(2)
        K, V = torch.cat([kc.to(k.dtype), k], 2), torch.cat([vc.to(v.dtype), v], 2)
        if attn.use_rope:
            q, K = apply_rope(q, cos, sin, start=M), apply_rope(K, cos, sin)
        ok = self.ok if self.ok is not None and M else torch.ones(B, M, dtype=torch.bool, device=q.device)
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril_()
        mask = torch.cat([ok.view(B, 1, 1, M).expand(B, 1, T, M),
                          causal.view(1, 1, T, T).expand(B, 1, T, T)], -1)
        out = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, enable_gqa=True)
        self.kv[id(attn)] = (torch.cat([kc.to(k.dtype), k], 2)[:, :, -self.size:],
                             torch.cat([vc.to(v.dtype), v], 2)[:, :, -self.size:])
        # the mask advances once per window: the last layer to write commits it
        new_ok = torch.cat([ok, torch.ones(B, T, dtype=torch.bool, device=q.device)], 1)[:, -self.size:]
        self._pending_ok = new_ok
        return out

    def end_window(self):
        if getattr(self, "_pending_ok", None) is not None:
            self.ok, self._pending_ok = self._pending_ok, None

    update = None  # not a layer-12 branch

    def snapshot(self):
        return {"kv": {i: (k.detach().clone(), v.detach().clone()) for i, (k, v) in self.kv.items()},
                "ok": None if self.ok is None else self.ok.clone()}

    def load(self, snap):
        self.kv = {i: (k.clone(), v.clone()) for i, (k, v) in snap["kv"].items()}
        self.ok = None if snap["ok"] is None else snap["ok"].clone()
