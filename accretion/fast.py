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
import torch.nn.functional as F


class FastMemory(nn.Module):
    def __init__(self, d, hidden=None, chunk=32, base_lr=0.05, base_decay=0.99,
                 bptt=False, readback=0.0, carry=False, per_token=False):
        super().__init__()
        # per_token: each token's gradient is scaled by ITS OWN eta before the
        # chunk sum (Titans / GDN per-token beta), so a fact can be written
        # harder than filler. Off = v2's per-chunk eta.mean (bit-exact).
        self.per_token = per_token
        # chunk = UPDATE PERIOD in tokens (Nested Learning's C^(l)). It may
        # exceed the window: gradients then accumulate across forward calls
        # and apply once per chunk (CMS eq. 71) -- a genuinely slower level.
        self.d, self.chunk = d, chunk
        # bptt: keep the state's graph across chunks AND forward calls; the
        # trainer calls detach_() every K windows. Fixes v2's structural flaw
        # (a probe's loss could never reach the projections that wrote it).
        self.bptt, self.readback = bptt, readback
        # carry: a partial tail chunk WAITS for the next call (CMS levels).
        # Off = the original behaviour: the tail applies at end of forward, so
        # a 20-token one-shot fact write still updates a chunk-32 memory.
        self.carry = carry
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
        self.last_aux = 0.0

    def reset(self, batch=1, device=None, dtype=None):
        dev = device or self.m0.device
        dt = dtype or self.m0.dtype
        self._state = self.m0.detach().to(dev, dt).expand(batch, -1, -1).clone()
        self._mom = torch.zeros_like(self._state)
        self._pend, self._pend_n, self._prev = torch.zeros_like(self._state), 0, None

    def reset_lanes(self, mask):
        """Per-lane reset for stateful training. Random resets keep states of
        many AGES in-distribution — a model that only ever sees fresh state
        cannot read an old one (measured: one-shot writes, 0% recall).
        Out-of-place: under bptt the state is still part of a live graph."""
        if self._state is None or not mask.any():
            return
        m = mask.to(self._state.device).view(-1, 1, 1)
        self._state = torch.where(m, self.m0.detach().to(self._state.dtype), self._state)
        self._mom = torch.where(m, torch.zeros_like(self._mom), self._mom)
        self._pend = torch.where(m, torch.zeros_like(self._pend), self._pend)
        if self._prev is not None:  # a reset lane's old facts are not owed back
            k, v, keep = self._prev
            self._prev = (k, v, keep & ~mask.to(keep.device))

    def detach_(self):
        """Truncation point for bptt: called by the trainer every K windows."""
        if self._state is None:
            return
        self._state, self._mom, self._pend = self._state.detach(), self._mom.detach(), self._pend.detach()
        if self._prev is not None:
            self._prev = tuple(x.detach() for x in self._prev)

    def _readback(self, state):
        """InfoNCE: can the JUST-UPDATED state still retrieve the PREVIOUS
        update's values from their keys? Contrastive, not MSE, so it cannot
        be gamed by collapsing v. Rewards writes that survive interference."""
        k, v, keep = self._prev
        if not keep.any():
            return state.new_zeros(())
        pred = F.normalize(torch.einsum("bde,bte->btd", state[keep], k[keep]), dim=-1)
        logits = torch.einsum("btd,bsd->bts", pred, F.normalize(v[keep], dim=-1)) / 0.1
        tgt = torch.arange(k.size(1), device=k.device).expand(logits.size(0), -1)
        return F.cross_entropy(logits.flatten(0, 1).float(), tgt.flatten())

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
        if self._state is None or self._state.device != h.device:
            self.reset(B, h.device, h.dtype)
        elif self._state.shape[0] != B:
            # batch changed. A persistent store written at B=1 must not be
            # silently wiped when evaluated at B=4 — broadcast the single
            # state instead. Only a genuine multi-state batch resets.
            if self._state.shape[0] == 1:
                self._state = self._state.expand(B, -1, -1).clone()
                self._mom = self._mom.expand(B, -1, -1).clone()
                self._pend = self._pend.expand(B, -1, -1).clone()
            else:
                self.reset(B, h.device, h.dtype)
        # L2-normalize k/q (Titans, GDN): without this the delta-rule update
        # state = a*state - eta*(err @ k^T) has no bound on ||k||, and the
        # recurrence compounds — measured: state_absmax hit 1e15 within 5
        # steps and the run went NaN by step 60. Normalizing is the fix all
        # the deep-read sources use, not a workaround.
        k, v, q = F.normalize(self.wk(h), dim=-1), self.wv(h), F.normalize(self.wq(h), dim=-1)
        eta, alpha, theta = self.gate_values(h)
        state = self._state.to(h.dtype)
        mom = self._mom.to(h.dtype)
        pend = self._pend.to(h.dtype)
        # first chunk starts from the meta-learned M0 so it receives gradient
        state = state + (self.m0.to(h.dtype).unsqueeze(0) - self.m0.detach().to(h.dtype).unsqueeze(0))
        outs, surprise, aux, n_upd = [], 0.0, h.new_zeros(()), 0
        want_aux = self.readback > 0 and self.training
        step = min(self.chunk - self._pend_n, T) if self._pend_n else min(self.chunk, T)
        s = 0
        while s < T:
            e = slice(s, min(s + step, T))
            kc, vc, qc = k[:, e], v[:, e], q[:, e]
            n = kc.size(1)
            # closed-form inner gradient of ||M k - v||^2 : (Mk - v) k^T,
            # always against the state as of the last applied update
            pred = torch.einsum("bde,bte->btd", state, kc)
            err = pred - vc
            surprise += float(err.pow(2).mean().detach())
            if self.per_token:  # eta_t * (M k_t - v_t) k_t^T, lr folded into the sum
                err = err * eta[:, e]
            pend = pend + torch.einsum("btd,bte->bde", err, kc)
            self._pend_n += n
            if self._pend_n >= self.chunk or (not self.carry and e.stop == T):  # update due
                a = alpha[:, e].mean(1, keepdim=True)      # (B,1,1)
                lr = 1.0 if self.per_token else eta[:, e].mean(1, keepdim=True)
                th = theta[:, e].mean(1, keepdim=True)
                mom = th * mom + pend / self._pend_n
                state = a * state - lr * mom
                if want_aux and self._prev is not None:
                    aux = aux + self._readback(state)
                    n_upd += 1
                if want_aux:  # remember a sample of what was just written
                    idx = torch.randperm(n, device=h.device)[:64]
                    self._prev = (kc[:, idx], vc[:, idx],
                                  torch.ones(B, dtype=torch.bool, device=h.device))
                pend, self._pend_n = torch.zeros_like(pend), 0
                applied = True
            else:
                applied = False
            # retrieve with the latest state (memory answers about what it has
            # already consolidated; a slow level lags by up to one period)
            outs.append(torch.einsum("bde,bte->btd", state, qc))
            if applied and not self.bptt and e.stop < T:  # TTT truncation, AFTER the read
                state, mom = state.detach(), mom.detach()
            s = e.stop
            step = min(self.chunk, T - s)
        if self.bptt:
            self._state, self._mom, self._pend = state, mom, pend
        else:
            self._state, self._mom, self._pend = state.detach(), mom.detach(), pend.detach()
            if self._prev is not None:
                self._prev = tuple(x.detach() for x in self._prev)
        self.last_aux = aux / max(n_upd, 1)
        out = torch.cat(outs, 1)
        return q + self.norm(out), surprise / max(1, len(outs))

    # alias so the module reads naturally at the call site
    update = forward

    def snapshot(self):
        return {"state": self._state.detach().clone(), "mom": self._mom.detach().clone(),
                "pend": self._pend.detach().clone(), "pend_n": self._pend_n}

    def load(self, snap):
        self._state = snap["state"].clone()
        self._mom = snap["mom"].clone() if "mom" in snap else torch.zeros_like(self._state)
        self._pend = snap["pend"].clone() if "pend" in snap else torch.zeros_like(self._state)
        self._pend_n, self._prev = snap.get("pend_n", 0), None


class MemoryStack(nn.Module):
    """Continuum Memory System, independent (head-wise) variant -- Nested
    Learning eq. 74: levels read the same h at different update periods and a
    learned weighted sum aggregates them. Same interface as FastMemory so
    every caller (trainer, oneshot, currprobe) works unchanged.

    levels: [[chunk, base_decay, bptt], ...], fastest first."""

    def __init__(self, d, levels, readback=0.0, per_token=False):
        super().__init__()
        self.levels = nn.ModuleList(
            FastMemory(d, chunk=c, base_decay=dec, bptt=bool(b),
                       readback=readback if b else 0.0, carry=True, per_token=per_token)
            for c, dec, b in levels)
        # level 0 carries the trained v2 signal; slower levels start small but
        # NOT zero (a zero weight starves their projections of gradient)
        self.w = nn.Parameter(torch.tensor([1.0] + [0.1] * (len(levels) - 1)))
        self.last_aux = 0.0

    def reset(self, batch=1, device=None, dtype=None):
        for m in self.levels:
            m.reset(batch, device, dtype)

    def reset_lanes(self, mask):
        for m in self.levels:
            m.reset_lanes(mask)

    def detach_(self):
        for m in self.levels:
            m.detach_()

    def forward(self, h):
        outs = [m(h) for m in self.levels]
        self.last_aux = sum(m.last_aux for m in self.levels)
        r = sum(w * o for w, (o, _) in zip(self.w, outs))
        return r, sum(s for _, s in outs) / len(outs)

    update = forward

    def snapshot(self):
        # level 0's state under "state" keeps single-state callers working
        snaps = [m.snapshot() for m in self.levels]
        return {"state": snaps[0]["state"], "levels": snaps}

    def load(self, snap):
        for m, s in zip(self.levels, snap["levels"]):
            m.load(s)
