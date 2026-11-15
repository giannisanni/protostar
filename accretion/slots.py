"""Slot memory: store every token's (key, value) from earlier windows as-is and
read by multi-head softmax attention (Memorizing-Transformer / Transformer-XL
style). The TTT matrix blends a whole window into one 640x640 map (locate.py:
the read returns a blend, rank ~18); slots keep each fact separable.

Drop-in for FastMemory: forward(h) -> (retrieved, surprise), reset,
reset_lanes, detach_, snapshot/load, last_aux. The read happens BEFORE the
current window is written, so a window only sees earlier windows here (its
own tokens are attention's job).

ponytail: FIFO of the last `size` tokens, no positions/recency; a capped table
with learned eviction (or product keys) is the upgrade if ages beyond the
FIFO matter.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotMemory(nn.Module):
    def __init__(self, d, n_head=10, size=4096, bptt=False):
        super().__init__()
        self.d, self.h, self.size, self.bptt = d, n_head, size, bptt
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.scale = nn.Parameter(torch.full((n_head, 1, 1), 4.0))  # qk-norm temperature
        # always-present "nothing stored" slot: an empty lane reads ~0, no NaN
        self.null_k = nn.Parameter(torch.randn(n_head, 1, d // n_head) * 0.02)
        # pre-norm: raw layer-12 h is large; unnormalised values blew up the
        # branch within 4 steps (smoke run: all-token acc 41% -> 17%/0%)
        self.ln = nn.LayerNorm(d)
        self._k = self._v = self._ok = None
        self.last_aux = 0.0

    def _heads(self, x):  # (B,T,d) -> (B,h,T,dh)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d // self.h).transpose(1, 2)

    def reset(self, batch=1, device=None, dtype=None):
        dev = device or self.wq.weight.device
        dt = dtype or self.wq.weight.dtype
        self._k = torch.zeros(batch, 0, self.d, device=dev, dtype=dt)
        self._v = torch.zeros_like(self._k)
        self._ok = torch.zeros(batch, 0, dtype=torch.bool, device=dev)

    def reset_lanes(self, mask):
        if self._ok is None or not mask.any():
            return
        self._ok = self._ok & ~mask.to(self._ok.device).view(-1, 1)  # out-of-place: graph-safe

    def detach_(self):
        if self._k is not None:
            self._k, self._v = self._k.detach(), self._v.detach()

    def forward(self, h):
        B, T, d = h.shape
        if self._k is None or self._k.device != h.device:
            self.reset(B, h.device, h.dtype)
        elif self._k.shape[0] != B:
            if self._k.shape[0] == 1:  # broadcast a single stored lane (eval)
                self._k, self._v, self._ok = (x.expand(B, *x.shape[1:]) for x in (self._k, self._v, self._ok))
            else:
                self.reset(B, h.device, h.dtype)
        hn = self.ln(h)
        q = F.normalize(self._heads(self.wq(hn)), dim=-1) * self.scale
        k_new, v_new = self.wk(hn), self.wv(hn)
        # read: null slot + stored slots
        nk = F.normalize(self.null_k, dim=-1).to(h.dtype).expand(B, -1, -1, -1)
        k = torch.cat([nk, F.normalize(self._heads(self._k.to(h.dtype)), dim=-1)], 2)
        v = torch.cat([torch.zeros_like(nk), self._heads(self._v.to(h.dtype))], 2)
        ok = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=h.device), self._ok], 1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=ok.view(B, 1, 1, -1), scale=1.0)
        out = out.transpose(1, 2).reshape(B, T, d)
        # write this window AFTER the read; FIFO-cap
        keep = slice(-self.size, None)
        self._k = torch.cat([self._k, k_new], 1)[:, keep]
        self._v = torch.cat([self._v, v_new], 1)[:, keep]
        self._ok = torch.cat([self._ok, torch.ones(B, T, dtype=torch.bool, device=h.device)], 1)[:, keep]
        if not self.bptt:
            self.detach_()
        return out, 0.0

    update = forward

    def snapshot(self):
        return {"k": self._k.detach().clone(), "v": self._v.detach().clone(), "ok": self._ok.clone()}

    def load(self, snap):
        self._k, self._v, self._ok = snap["k"].clone(), snap["v"].clone(), snap["ok"].clone()
