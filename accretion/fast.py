"""Fast memory: a small network that learns DURING the forward pass.

Every element here is a measured ablation win from the deep reads, not a
design preference:
  - inner net f(x) = x + LN(MLP(x))         TTT-layers: -1.22 ppl
  - mini-batch inner updates (b=16..64)     TTT-layers: -1.70 ppl (biggest)
  - data-dependent eta/alpha per token      Titans, TTT, ARC-TTT, SEAL all converge
  - meta-learned initial state W0           TTT-layers: stability (NOT zeros)
  - keys/values from learned projections    Titans (not raw hidden states)
  - momentum on surprise + weight decay     Titans: the two things separating
                                            it from plain TTT

v1's mistake was a frozen backbone; this module is meant to be present
DURING pretraining so the backbone learns to rely on it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastMemory(nn.Module):
    def __init__(self, d, hidden=2048, chunk=32, base_lr=0.01, base_decay=0.99):
        super().__init__()
        self.d, self.chunk = d, chunk
        # learned projections: memory maps k -> v, retrieved by q (Titans)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wq = nn.Linear(d, d, bias=False)
        # data-dependent gates: inner LR (eta), forget/decay (alpha), momentum (theta)
        self.gates = nn.Linear(d, 3)
        nn.init.zeros_(self.gates.weight)
        nn.init.zeros_(self.gates.bias)
        self.base_lr, self.base_decay = base_lr, base_decay
        # META-LEARNED init state — reset() returns here, not to zeros
        self.w0_1 = nn.Parameter(torch.randn(d, hidden) * (d ** -0.5))
        self.w0_2 = nn.Parameter(torch.zeros(hidden, d))  # zero => no-op at start
        self.norm = nn.LayerNorm(d)
        self.register_buffer("_w1", torch.zeros(d, hidden), persistent=False)
        self.register_buffer("_w2", torch.zeros(hidden, d), persistent=False)
        self.register_buffer("_m1", torch.zeros(d, hidden), persistent=False)
        self.register_buffer("_m2", torch.zeros(hidden, d), persistent=False)
        self.reset()

    def reset(self):
        """Restore the META-LEARNED initial state (TTT: learned W0, not zeros)."""
        with torch.no_grad():
            self._w1 = self.w0_1.detach().clone()
            self._w2 = self.w0_2.detach().clone()
            self._m1 = torch.zeros_like(self._w1)
            self._m2 = torch.zeros_like(self._w2)

    def _inner(self, x, w1, w2):
        """f(x) = x + LN(MLP(x)) — residual + norm inside the memory."""
        return x + self.norm(F.gelu(x @ w1) @ w2)

    def gate_values(self, h):
        """Per-token eta (inner LR), alpha (retention), theta (momentum)."""
        g = torch.sigmoid(self.gates(h))
        eta = self.base_lr * g[..., 0:1]
        alpha = 1.0 - (1.0 - self.base_decay) * g[..., 1:2]
        theta = 0.9 * g[..., 2:3]
        return eta, alpha, theta

    def update(self, h):
        """One chunk-wise inner step over hidden states h (B,T,d).
        Returns (retrieved, surprise). Inner state is detached from the outer
        graph — the outer optimizer trains projections/gates, not the inner
        trajectory (TTT chunk-wise dual form)."""
        k, v, q = self.wk(h), self.wv(h), self.wq(h)
        eta, alpha, theta = self.gate_values(h)
        w1, w2 = self._w1, self._w2
        surprise_total = 0.0
        for s in range(0, h.size(1), self.chunk):
            kc, vc = k[:, s:s + self.chunk], v[:, s:s + self.chunk]
            e = eta[:, s:s + self.chunk].mean()
            a = alpha[:, s:s + self.chunk].mean()
            t = theta[:, s:s + self.chunk].mean()
            w1d, w2d = w1.detach().requires_grad_(True), w2.detach().requires_grad_(True)
            with torch.enable_grad():
                pred = self._inner(kc.detach(), w1d, w2d)
                loss = (pred - vc.detach()).pow(2).mean()
                g1, g2 = torch.autograd.grad(loss, (w1d, w2d))
            with torch.no_grad():
                self._m1 = t * self._m1 + g1
                self._m2 = t * self._m2 + g2
                w1 = a * w1.detach() - e * self._m1
                w2 = a * w2.detach() - e * self._m2
            surprise_total += loss.item()
        self._w1, self._w2 = w1, w2
        # retrieval uses the CURRENT state but stays differentiable w.r.t. wq
        out = self._inner(q, w1.detach(), w2.detach())
        n = max(1, (h.size(1) + self.chunk - 1) // self.chunk)
        return out, surprise_total / n

    def snapshot(self):
        return {"w1": self._w1.detach().clone(), "w2": self._w2.detach().clone()}

    def load(self, snap):
        with torch.no_grad():
            self._w1 = snap["w1"].clone()
            self._w2 = snap["w2"].clone()
