"""Fast memory: a matrix-valued state that learns DURING the forward pass.

TTT-Linear form. The inner objective is ||M k - v||^2, whose gradient has a
CLOSED FORM (M k - v) k^T — so the whole update is plain matmuls and stays
differentiable w.r.t. the projections and gates without second-order autograd.
That matters: the outer loop MUST train wk/wv/gates (they decide what gets
stored and when), and an implementation that detaches them leaves the
data-dependent gating — the one thing all four papers converge on — frozen
at initialization.

Design elements, each a measured ablation win from the deep reads:
  - f(x) = x + LN(state @ x)                 TTT: LN+residual inner, -1.22 ppl
  - chunked updates, state detached between  TTT: -1.70 ppl (biggest single win)
  - per-token eta (inner LR) and alpha (retention) from a learned head
                                             Titans/TTT/ARC-TTT/SEAL converge
  - momentum on the surprise signal          Titans (vs plain TTT)
  - meta-learned initial state M0            TTT: stability (not zeros)

TTT-MLP (deep inner net) beat TTT-Linear on long context but has memory-I/O
problems; ponytail: linear first, MLP is the upgrade if this works.
"""
import torch
import torch.nn as nn


class FastMemory(nn.Module):
    def __init__(self, d, hidden=None, chunk=32, base_lr=0.5, base_decay=0.99):
        super().__init__()
        self.d, self.chunk = d, chunk
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wq = nn.Linear(d, d, bias=False)
        self.gates = nn.Linear(d, 3)  # eta (lr), alpha (retention), theta (momentum)
        nn.init.zeros_(self.gates.weight)
        nn.init.zeros_(self.gates.bias)
        self.norm = nn.LayerNorm(d)
        # META-LEARNED initial state (TTT: learned M0, not zeros). Zero-init so
        # the module starts as a no-op, but it is a Parameter and receives
        # gradient through the first chunk's update.
        self.m0 = nn.Parameter(torch.zeros(d, d))
        self.base_lr, self.base_decay = base_lr, base_decay
        self._state = None
        self._mom = None

    def reset(self, batch=1, device=None, dtype=None):
        dev = device or self.m0.device
        dt = dtype or self.m0.dtype
        self._state = self.m0.detach().to(dev, dt).expand(batch, -1, -1).clone()
        self._mom = torch.zeros_like(self._state)

    def gate_values(self, h):
        g = torch.sigmoid(self.gates(h))
        eta = self.base_lr * g[..., 0:1]
        alpha = 1.0 - (1.0 - self.base_decay) * g[..., 1:2]
        theta = 0.9 * g[..., 2:3]
        return eta, alpha, theta

    def forward(self, h):
        """Run the inner loop over h (B,T,d). Returns (retrieved, surprise).

        State carries across chunks detached (truncated BPTT); WITHIN a chunk
        everything is differentiable, so wk/wv/wq/gates all receive gradient
        from the outer loss.
        """
        B, T, d = h.shape
        if self._state is None or self._state.shape[0] != B or \
                self._state.device != h.device:
            self.reset(B, h.device, h.dtype)
        k, v, q = self.wk(h), self.wv(h), self.wq(h)
        eta, alpha, theta = self.gate_values(h)
        state = self._state.to(h.dtype)
        mom = self._mom.to(h.dtype)
        # first chunk starts from the meta-learned M0 so it receives gradient
        state = state + (self.m0.to(h.dtype).unsqueeze(0) - self.m0.detach().to(h.dtype).unsqueeze(0))
        outs, surprise = [], 0.0
        for s in range(0, T, self.chunk):
            e = slice(s, min(s + self.chunk, T))
            kc, vc, qc = k[:, e], v[:, e], q[:, e]
            a = alpha[:, e].mean(1, keepdim=True)          # (B,1,1)
            lr = eta[:, e].mean(1, keepdim=True)
            th = theta[:, e].mean(1, keepdim=True)
            # closed-form inner gradient of ||M k - v||^2 : (Mk - v) k^T
            pred = torch.einsum("bde,bte->btd", state, kc)
            err = pred - vc
            surprise += float(err.pow(2).mean().detach())
            grad = torch.einsum("btd,bte->bde", err, kc) / max(kc.size(1), 1)
            mom = th * mom + grad
            state = a * state - lr * mom
            # retrieve with the JUST-updated state (memory answers about what
            # it has already seen in this chunk)
            outs.append(torch.einsum("bde,bte->btd", state, qc))
            state, mom = state, mom  # keep graph inside the chunk
            if e.stop < T:  # truncate BPTT at the chunk boundary
                state, mom = state.detach(), mom.detach()
        self._state, self._mom = state.detach(), mom.detach()
        out = torch.cat(outs, 1)
        n = max(1, (T + self.chunk - 1) // self.chunk)
        return q + self.norm(out), surprise / n

    # alias so the module reads naturally at the call site
    update = forward

    def snapshot(self):
        return {"state": self._state.detach().clone()}

    def load(self, snap):
        self._state = snap["state"].clone()
        self._mom = torch.zeros_like(self._state)
