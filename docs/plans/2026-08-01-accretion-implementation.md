# Accretion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a model whose weights learn at inference — fast memory that trains during the forward pass, consolidated nightly into a persistent sparse substrate behind a rollback gate — and prove it with a learn-and-retain benchmark.

**Architecture:** Frozen `gdn` backbone (the M0 winner). A small memory MLP runs inner-loop SGD on a surprise signal while tokens stream. At context end its weights are a compressed diff, buffered. An external sleep job distills that diff into a *persistent sparse substrate*, gated on held-out loss plus a growing retention suite, with rollback. Design rationale: `docs/plans/2026-07-29-accretion-design.md`.

**Tech Stack:** Existing protostar stack (PyTorch 2.13 cu130, `fla` GDN kernels, Muon, `model.py`/`train.py`). New code lives in `accretion/`. Sleep runs on the 4060 Ti (cuda:0); training on the two 5060 Tis (`CUDA_VISIBLE_DEVICES=1,2`).

---

## Why this plan opens differently than the design doc

**M0 changed the starting assumption.** The design assumed Engram tables were the persistent substrate. The ablation says otherwise:

| config | val loss | Δ vs dense |
|---|---|---|
| gdn | **3.2589** | −0.0273 |
| gdn_mtp_engram | 3.2798 | −0.0064 |

Engram *cost* 0.008 nats on top of GDN. The backbone learned to keep the gate mostly closed — and accretion consolidates **through** that gate.

**But that result does not refute Engram's actual claim.** It measured perplexity on web text; Engram's claim (and accretion's) is *factual recall*. Those come apart: a table that memorizes rare entities barely moves average next-token loss.

**Therefore Phase 1 is a substrate bake-off on the right metric**, not an assumption. Nothing else gets built until a substrate demonstrably stores and returns facts.

**Backbone config is frozen as `configs/gdn.json`** (`gdn_ratio=4`, no MTP, no Engram). MTP is out (−0.0125 vs gdn, 23% slower); revisit only at d≥1024 as a serving optimization.

---

## Phase 1: Does ANY sparse substrate store facts? (kill-fast)

**This phase can kill the project in ~2 days for ~€1 of electricity. That is its purpose.** No inner loop, no sleep, no gate — just: can we write facts into a sparse memory attached to a frozen backbone and read them back?

### Task 1: Fact corpus + recall probe

**Files:**
- Create: `accretion/facts.py`
- Test: `tests/test_facts.py`

Synthetic facts are mandatory — real facts may be in FineWeb, and then we'd measure pretraining, not memory.

**Step 1: Write the failing test**

```python
# tests/test_facts.py
from accretion.facts import make_facts, probe_prompt

def test_facts_are_unique_and_novel():
    facts = make_facts(200, seed=0)
    assert len(facts) == 200
    assert len({f["subject"] for f in facts}) == 200      # no subject collisions
    assert all(f["object"] not in f["doc"].split(f["subject"])[0] for f in facts)

def test_probe_prompt_omits_the_answer():
    f = make_facts(1, seed=0)[0]
    p = probe_prompt(f)
    assert f["subject"] in p
    assert f["object"] not in p                            # answer must not leak

def test_deterministic():
    assert make_facts(50, seed=3) == make_facts(50, seed=3)
```

**Step 2: Run it — expect failure**

Run: `.venv/bin/python -m pytest tests/test_facts.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'accretion.facts'`

**Step 3: Implement**

```python
# accretion/facts.py
"""Synthetic facts with provably-novel subjects.

Real facts risk being in FineWeb — then we'd measure pretraining, not memory.
Subjects are nonsense bigrams; objects are drawn from a closed set so a probe
has a well-defined right answer.
"""
import random

SYL = ["zor", "vek", "mip", "thal", "quon", "bri", "nax", "wel", "dush", "kip",
       "orn", "vay", "lut", "gan", "sirn", "pel", "murt", "cav", "yex", "drin"]
ROLES = ["the capital of {s} is {o}", "{s} was founded by {o}",
         "the currency of {s} is called {o}", "{s} is famous for its {o}"]
OBJECTS = ["Tarnwick", "Belloran", "Quillfast", "Ashgrove", "Vermillion",
           "Duskrow", "Farrowgate", "Nimbleton", "Copperfen", "Halewind"]


def _name(rng):
    return (rng.choice(SYL) + rng.choice(SYL)).capitalize()


def make_facts(n, seed=0):
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        s = _name(rng)
        if s in seen:
            continue
        seen.add(s)
        o = rng.choice(OBJECTS)
        tmpl = rng.choice(ROLES)
        sent = tmpl.format(s=s, o=o)
        out.append({
            "subject": s, "object": o, "template": tmpl,
            # a short document so there is context around the fact
            "doc": f"A note about {s}. It is widely recorded that {sent}. "
                   f"Scholars of {s} consider this well established.",
        })
    return out


def probe_prompt(fact):
    """Prompt that ends exactly where the object should be generated."""
    return "It is widely recorded that " + \
        fact["template"].format(s=fact["subject"], o="").rstrip()
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_facts.py -q`
Expected: 3 passed

**Step 5: Commit**

```bash
git add accretion/facts.py tests/test_facts.py
git commit -m "feat(accretion): synthetic fact corpus + recall probe"
```

---

### Task 2: Three substrates behind one interface

**Files:**
- Create: `accretion/substrates.py`
- Test: `tests/test_substrates.py`

One interface, three implementations, so the bake-off is apples-to-apples. Each takes hidden states + token ids and returns a residual contribution.

**Step 1: Write the failing test**

```python
# tests/test_substrates.py
import pytest, torch
from accretion.substrates import build_substrate, SUBSTRATES

@pytest.mark.parametrize("kind", SUBSTRATES)
def test_zero_init_is_noop(kind):
    """Every substrate must start as an exact no-op — otherwise it perturbs a
    frozen backbone before it has learned anything."""
    torch.manual_seed(0)
    sub = build_substrate(kind, d=64, n_layer=4).eval()
    h = torch.randn(2, 16, 64)
    ids = torch.randint(0, 500, (2, 16))
    out = sub(h.clone(), ids, layer=1)
    torch.testing.assert_close(out, h, atol=0, rtol=0)

@pytest.mark.parametrize("kind", SUBSTRATES)
def test_writable_params_get_grads(kind):
    sub = build_substrate(kind, d=64, n_layer=4)
    for p in sub.write_params():
        p.data.normal_(0, 0.02)          # leave the no-op state
    h = torch.randn(2, 16, 64, requires_grad=True)
    ids = torch.randint(0, 500, (2, 16))
    sub(h, ids, layer=1).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in sub.write_params())

@pytest.mark.parametrize("kind", SUBSTRATES)
def test_capacity_reported(kind):
    sub = build_substrate(kind, d=64, n_layer=4)
    assert sub.n_slots() > 0
```

**Step 2: Run it — expect failure**

Run: `.venv/bin/python -m pytest tests/test_substrates.py -q`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement**

```python
# accretion/substrates.py
"""Three candidate persistent memories behind one interface.

M0 showed Engram does not help perplexity. This module exists so Phase 1 can
ask the RIGHT question — factual recall — of three designs rather than
assuming one. All are zero-init no-ops so a frozen backbone is undisturbed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SUBSTRATES = ("engram", "ple", "pk")


class _Base(nn.Module):
    def write_params(self):
        """Params sleep is allowed to modify."""
        raise NotImplementedError

    def n_slots(self):
        raise NotImplementedError


class Engram(_Base):
    """Bigram-hash lookup, gated. The M0 arm's design."""

    def __init__(self, d, n_layer, buckets=1 << 16, heads=2, dim=64, at=None):
        super().__init__()
        self.at = at if at is not None else n_layer // 3
        self.buckets = buckets
        self.tables = nn.ModuleList([nn.Embedding(buckets, dim) for _ in range(heads)])
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.02)
        self.gate = nn.Linear(d, 1)
        self.proj = nn.Linear(heads * dim, d, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, h, ids, layer):
        if layer != self.at:
            return h
        prev = F.pad(ids[:, :-1], (1, 0))
        keys = [(prev * 1000003 + ids) % self.buckets,
                (prev * 31 + ids * 999983) % self.buckets]
        mem = torch.cat([t(k) for t, k in zip(self.tables, keys)], -1)
        return h + torch.sigmoid(self.gate(h)) * self.proj(mem)

    def write_params(self):
        return list(self.tables.parameters()) + list(self.proj.parameters())

    def n_slots(self):
        return self.buckets * len(self.tables)


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
    the fix for Engram's inability to match a paraphrase."""

    def __init__(self, d, n_layer, c=256, dim=64, topk=8, at=None):
        super().__init__()
        self.at = at if at is not None else n_layer // 3
        self.c, self.topk = c, topk
        self.q = nn.Linear(d, 2 * dim, bias=False)
        self.keys = nn.Parameter(torch.randn(2, c, dim) * 0.02)
        self.values = nn.Embedding(c * c, d)
        nn.init.zeros_(self.values.weight)          # no-op at init

    def forward(self, h, ids, layer):
        if layer != self.at:
            return h
        q = self.q(h).chunk(2, dim=-1)
        s0 = torch.einsum("btd,cd->btc", q[0], self.keys[0])
        s1 = torch.einsum("btd,cd->btc", q[1], self.keys[1])
        t0, i0 = s0.topk(self.topk, -1)
        t1, i1 = s1.topk(self.topk, -1)
        # combine half-scores into topk^2 candidates, keep best topk
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
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_substrates.py -q`
Expected: 9 passed (3 tests × 3 substrates)

**Step 5: Commit**

```bash
git add accretion/substrates.py tests/test_substrates.py
git commit -m "feat(accretion): engram/ple/product-key substrates behind one interface"
```

---

### Task 3: The bake-off — write facts, measure recall AND damage

**Files:**
- Create: `accretion/bakeoff.py`
- Test: `tests/test_bakeoff.py`

Freeze the backbone, train ONLY substrate write-params on the fact docs, then measure two things that must both hold: recall went up, general ability did not go down.

**Step 1: Write the failing test**

```python
# tests/test_bakeoff.py
from accretion.bakeoff import recall_at_1

def test_recall_scores_exact_object_match():
    facts = [{"subject": "Zorvek", "object": "Tarnwick",
              "template": "the capital of {s} is {o}", "doc": ""}]
    assert recall_at_1(lambda p: " Tarnwick", facts) == 1.0
    assert recall_at_1(lambda p: " Belloran", facts) == 0.0
    assert recall_at_1(lambda p: " tarnwick.", facts) == 1.0   # norm: case/punct
```

**Step 2: Run it — expect failure**

Run: `.venv/bin/python -m pytest tests/test_bakeoff.py -q`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement**

```python
# accretion/bakeoff.py
"""Phase 1 gate: can a sparse substrate store facts into a FROZEN backbone?

Two numbers decide it:
  recall@1  — does the probe return the right object
  Δval      — did general ability degrade (must stay tiny)

A substrate that memorizes by wrecking the backbone is not a memory.
"""
import argparse
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accretion.facts import make_facts, probe_prompt          # noqa: E402
from accretion.substrates import build_substrate, SUBSTRATES   # noqa: E402
from data.loader import Shards                                 # noqa: E402
from model import Config, Protostar                            # noqa: E402


def _norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def recall_at_1(generate_fn, facts):
    hits = sum(_norm(fact["object"]) in _norm(generate_fn(probe_prompt(fact)))
               for fact in facts)
    return hits / len(facts)


@torch.no_grad()
def val_loss(model, sub, loader, device, n=12):
    tot = 0.0
    for batch, _ in zip(loader.batches(), range(n)):
        batch = batch.to(device)
        _, loss = forward(model, sub, batch[:, :-1], batch[:, 1:])
        tot += loss.item()
    return tot / n


def forward(model, sub, idx, targets=None):
    """Backbone forward with the substrate injected between layers."""
    h = model.embed(idx)
    for i, blk in enumerate(model.blocks):
        h = blk(h, model.cos, model.sin)
        if sub is not None:
            h = sub(h, idx, layer=i)
    h = model.norm_f(h)
    if targets is None:
        return model.lm_head(h), None
    return None, model._ce(h.view(-1, model.cfg.d), targets.reshape(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--kind", choices=SUBSTRATES, required=True)
    ap.add_argument("--facts", type=int, default=200)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    model = Protostar(Config(**ck["cfg"])).to(a.device).eval()
    model.load_state_dict(ck["model"])
    for p in model.parameters():          # FROZEN. only the substrate learns.
        p.requires_grad_(False)

    sub = build_substrate(a.kind, d=model.cfg.d, n_layer=model.cfg.n_layer).to(a.device)
    facts = make_facts(a.facts, seed=0)
    val = Shards(a.data, "val", 4, model.cfg.seq_len)

    def gen(prompt, n=6):
        ids = torch.tensor([enc.encode(prompt)], device=a.device)
        for _ in range(n):
            logits, _ = forward(model, sub, ids[:, -model.cfg.seq_len:])
            nxt = logits[0, -1, :50257].argmax()
            ids = torch.cat([ids, nxt.view(1, 1)], 1)
        return enc.decode(ids[0].tolist()[-n:])

    base_recall = recall_at_1(gen, facts)
    base_val = val_loss(model, None, val, a.device)

    opt = torch.optim.AdamW(sub.write_params(), lr=a.lr)
    docs = [torch.tensor([enc.encode(f["doc"])], device=a.device) for f in facts]
    sub.train()
    for step in range(a.steps):
        doc = docs[step % len(docs)]
        _, loss = forward(model, sub, doc[:, :-1], doc[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    sub.eval()

    after_recall = recall_at_1(gen, facts)
    after_val = val_loss(model, sub, val, a.device)
    print(f"{a.kind:8s} slots={sub.n_slots():>9,}  "
          f"recall {base_recall:.1%} -> {after_recall:.1%}   "
          f"val {base_val:.4f} -> {after_val:.4f} (Δ{after_val - base_val:+.4f})")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_bakeoff.py -q`
Expected: 1 passed

**Step 5: Commit**

```bash
git add accretion/bakeoff.py tests/test_bakeoff.py
git commit -m "feat(accretion): substrate bake-off harness"
```

---

### Task 4: Run the bake-off — THE KILL GATE

**Files:** none (execution only)

**Step 1: Sync and run all three on the idle 4060 Ti**

```bash
rsync -a --exclude .venv --exclude .git ~/workspace/protostar/ gianni@substrate:~/protostar/
ssh gianni@substrate 'cd ~/protostar && for k in engram ple pk; do \
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python accretion/bakeoff.py \
    --ckpt /mnt/data/protostar/runs/gdn/ckpt_007630.pt \
    --data /mnt/data/protostar/fineweb10B --kind $k --device cuda:0; done'
```

**Step 2: Read the result against these thresholds**

| outcome | meaning | action |
|---|---|---|
| any substrate ≥60% recall, Δval ≤ +0.02 | memory works | **proceed to Phase 2 with the winner** |
| best 20-60% recall | weak but alive | tune capacity/lr/steps once, then decide |
| all <20% recall, or Δval > +0.05 | frozen backbone cannot be written to this way | **STOP. Write up the negative result.** |

**Step 3: Record the numbers**

Append the table to `docs/plans/2026-08-01-accretion-implementation.md` under a "Phase 1 results" heading, and commit.

**Step 4: If STOP** — the honest write-up is the deliverable. Two documented negatives (M0 perplexity-null, Phase 1 recall-null) is a real finding about small-model sparse memory, and the design doc's fallbacks (LoRA-only skill lane, retrieval) become the path instead.

---

## Phase 2: Fast memory (only if Phase 1 passed)

### Task 5: Titans-style fast memory module

**Files:**
- Create: `accretion/fast.py`
- Test: `tests/test_fast.py`

**Step 1: Failing test**

```python
# tests/test_fast.py
import torch
from accretion.fast import FastMemory

def test_learns_within_a_context():
    """Surprise should fall as the same chunk repeats — that IS the learning."""
    fm = FastMemory(d=64)
    h = torch.randn(1, 32, 64)
    first = fm.update(h, h.roll(-1, dims=1))
    for _ in range(20):
        fm.update(h, h.roll(-1, dims=1))
    last = fm.update(h, h.roll(-1, dims=1))
    assert last < first * 0.8

def test_reset_restores_init():
    fm = FastMemory(d=64)
    snap0 = fm.snapshot()
    h = torch.randn(1, 32, 64)
    fm.update(h, h.roll(-1, dims=1))
    fm.reset()
    for a, b in zip(fm.snapshot().values(), snap0.values()):
        torch.testing.assert_close(a, b)

def test_snapshot_roundtrip():
    fm = FastMemory(d=64)
    h = torch.randn(1, 32, 64)
    fm.update(h, h.roll(-1, dims=1))
    snap = fm.snapshot()
    fm.reset()
    fm.load(snap)
    for a, b in zip(fm.snapshot().values(), snap.values()):
        torch.testing.assert_close(a, b)
```

**Step 2:** Run → FAIL (module missing).

**Step 3: Implement**

```python
# accretion/fast.py
"""Fast memory: a small MLP that learns DURING the forward pass.

Titans recipe — associative key->value map updated by gradient on its own
prediction error (surprise), with momentum (surprise has temporal structure)
and decay (controlled forgetting, so long contexts do not saturate it).
Per-chunk rather than per-token so the inner loop stays cheap.
"""
import torch
import torch.nn as nn


class FastMemory(nn.Module):
    def __init__(self, d, hidden=2048, lr=0.01, momentum=0.9, decay=0.999):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.lr, self.momentum, self.decay = lr, momentum, decay
        self.mom = [torch.zeros_like(p) for p in self.net.parameters()]

    def forward(self, k):
        return self.net(k)

    def update(self, keys, values):
        """One inner-loop step. Returns surprise (scalar) for logging/pressure."""
        with torch.enable_grad():
            k = keys.detach().requires_grad_(False)
            pred = self.net(k)
            loss = (pred - values.detach()).pow(2).mean()
            grads = torch.autograd.grad(loss, list(self.net.parameters()))
        with torch.no_grad():
            for p, g, m in zip(self.net.parameters(), grads, self.mom):
                m.mul_(self.momentum).add_(g)
                p.mul_(self.decay).add_(m, alpha=-self.lr)
        return loss.item()

    def snapshot(self):
        return {n: p.detach().clone() for n, p in self.net.named_parameters()}

    def load(self, snap):
        with torch.no_grad():
            for n, p in self.net.named_parameters():
                p.copy_(snap[n])

    def reset(self):
        with torch.no_grad():
            for n, p in self.net.named_parameters():
                p.zero_() if "weight" not in n or n.startswith("2") else nn.init.xavier_uniform_(p)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        self.mom = [torch.zeros_like(p) for p in self.net.parameters()]
```

**Step 4:** Run → 3 passed. **Step 5:** Commit `feat(accretion): titans-style fast memory`.

---

### Task 6: Meta-train the backbone to USE a changing memory

**Files:**
- Modify: `train.py` (add `--fast_memory` flag)
- Create: `configs/gdn_fast.json`

This is the M1 milestone and the hardest bet: the backbone must be pretrained *with* the inner loop active, or it will ignore the memory (the failure mode of `giannisanni/TITANS-implementation`, which needed a 1000x magnitude hack to be noticed).

**Steps:** wire FastMemory into the forward pass behind the flag, update every `--accum` micro-batch, keep gradients out of the outer optimizer. Run one 2B-token arm (~14h, `CUDA_VISIBLE_DEVICES=1,2`). **Pass criteria:** training stays stable (no loss spikes >2x) AND val loss within 0.02 of plain `gdn`. Fail → the meta-learning bet is dead; fall back to a frozen backbone plus substrate-only writes from Phase 1.

---

### Task 7: Sleep — consolidate fast memory into the substrate

**Files:**
- Create: `accretion/sleep.py`
- Test: `tests/test_sleep.py`

Replay buffered contexts, find positions where the fast memory actually contributed, distill *only those* into substrate write-params, then gate.

**The gate is the load-bearing part** and it stays external forever — the thing being modified must not certify its own modification:

```python
def gate(before_val, after_val, probe_results, eps=0.01):
    """Returns True if the write batch may be kept."""
    return (after_val - before_val) <= eps and all(probe_results)
```

Rollback = restore the pre-write substrate snapshot. Tests must cover: a good write is kept, a write that raises val loss past eps is rolled back, and a write that breaks a previously-learned probe is rolled back.

---

## Phase 3: The claim

### Task 8: Learn-and-retain benchmark

**Files:** `accretion/benchmark.py`

K fact-docs shown **once** at inference on day 0 → 100M+ tokens of unrelated traffic with nightly sleeps → measure recall **with fast memory OFF** (the substrate must carry it alone), general val loss, and the full retention suite.

Controls that make the result meaningful: frozen baseline (recall ≈ 0) and naive nightly LoRA (expected to forget or degrade — demonstrating that is part of the claim).

### Task 9: Skill probe

Introduce an invented mini-DSL by document at inference; later test **execution**, not recall. Locates the fact/skill boundary empirically for this architecture (see design doc's skill taxonomy).

---

## Out of scope

Sleep pressure as a learned signal, internal clock, dreaming, J-space-gated consolidation, the LoRA skill library — all v2/v3, all documented in the design doc. Do not build them until Phase 3 produces a number.

## Deep-read corrections (2026-08-01): ARC-TTT + Titans, full PDFs

### Titans (2501.00663) — corrections to Task 5 FastMemory

Read in full. Our FastMemory is Titans-lite in three ways that matter:

1. **α/η/θ must be DATA-DEPENDENT, not constants.** Titans' forget gate
   (α_t), surprise decay (η_t) and inner LR (θ_t) are learned functions of
   the input token, trained by the outer loop. That is what lets memory
   clear on context change and fully absorb when relevant. Our fixed
   lr/momentum/decay is the paper's own strawman. Fix in Task 5: small
   linear heads from the residual stream -> per-token α, η, θ.
2. **K/V/Q projections are meta-learned.** ℓ = ||M(k_t) − v_t||² with
   k=xW_K, v=xW_V learned in the OUTER loop; retrieval is M*(q), q=xW_Q.
   We currently write raw hidden states. Add W_K/W_V/W_Q to Task 6's
   meta-training.
3. **Integration variant matters: MAG > MAL, and ours is MAL-shaped.**
   Titans tested Memory-as-Context (MAC), Memory-as-Gate (MAG, parallel
   branch: o = y ⊗ M(x) with learned norms + σ), Memory-as-Layer (MAL,
   sequential). MAC/MAG clearly beat MAL. Our single-site additive
   injection ≈ MAL. Task 6 should wire MAG-style: memory as a gated
   PARALLEL branch, not an in-line residual add.

Also adopt: deep memory (MLP≥2 layers) confirmed better than linear (we
have this); 1D depthwise conv on q/k/v + ℓ2-norm + SiLU (cheap, they
report it helps); **persistent memory = N_p learnable prefix tokens**,
which is simultaneously their fix for attention's initial-token bias —
i.e. our deferred attention-sinks feature by another route. One mechanism,
two birds; add to Task 6.

Scale check: Titans' own experiments are 170M-760M on FineWeb-Edu 15-30B
tokens — exactly our regime, with AdamW 4e-4 / cosine / 0.5M-token batches
/ wd 0.1. Their LMM beats TTT and Gated DeltaNet baselines at this scale,
and the ablation credits weight decay + momentum in the surprise metric —
the two things distinguishing it from plain TTT.

### ARC-TTT (2411.07279) — the Phase-2/skill-lane recipe, exact

Read in full incl. appendices. The recipe that hit 6x (5 -> 29 tasks at 1B):

- **LoRA:** rank 128, α 16, on Q+V projections + MLP + output proj;
  lr 5e-5 (1-3B) / 1e-4 (8B); 2 epochs; batch 1-2; AdamW; max 250
  examples per task. **QLoRA loses only 29->26** — the 16GB-card option.
- **Episode construction (the load-bearing piece):** leave-one-out tasks
  from K demonstrations + order permutations + invertible geometric
  transforms. Ablations: remove transforms 29->13 (-55%); direct
  input-output instead of in-context episodes 29->18 (-38%).
- **Loss on ALL outputs including demonstrations** (29 vs 26 without) —
  train on the full episode, not just the answer.
- **Per-task vs shared adapters is a REGIME, not a preference:** per-task
  wins on ARC (uniform format, tasks confound); shared wins on BBH
  (distinguishable instructions, tasks mutually help). Skill-library rule:
  distinguishable skills -> shared/merged; uniform-format -> per-skill.
- **FT-before-TTT is the meta-adaptability stage** and its data ablation
  has a warning for us: **No-LM (rule-based ReARC + geometric only, 29)
  BEAT All (incl. 6426 GPT-4/4o-generated tasks, 24).** LM-generated
  training tasks actively hurt downstream adaptation. Echoes BeyondWeb's
  rephrase-don't-generate. Phase 5 synthetic data: procedural/grounded
  augmentation first, freeform LM generation last.
- **TTT closes the size gap:** 1B and 3B tie at 29 after TTT (8B: 36).
  Small models gain the MOST from test-time adaptation — evidence FOR the
  whole accretion premise at our scale.
- Inference: greedy + invertible-transform ensembling + hierarchical
  voting (top-3 intra-transform, then global top-2); ~12h/100 ARC tasks
  on one A100 for the full pipeline; BBH full run 15 min.

### Concrete plan edits

- Task 5: add per-token α/η/θ heads + W_K/W_V/W_Q projections to
  FastMemory; keep 2-layer MLP.
- Task 6: MAG-style gated parallel integration (not in-line residual);
  add N_p persistent prefix tokens; meta-train projections + gate heads.
- Task 9 (skill probe) and the deferred LoRA skill lane: adopt the
  ARC-TTT episode recipe (leave-one-out + augmentation + all-outputs
  loss + per-skill LoRA r128), QLoRA on 16GB.
- Phase 5 data note (cross-plan): LM-generated tasks degraded adaptation;
  prefer procedural augmentation.

## Deep-read round 2 (2026-08-01): TTT-layers + SEAL, full PDFs

### TTT-layers (2407.04620) — the FastMemory recipe ladder, with numbers

Their 125M ablation is a precise ladder from linear attention (ppl 15.91)
to TTT-Linear (11.09). The two biggest rungs and what they mean for Task 5:

1. **Inner model MUST be f(x) = x + LN(f_res(x))** — LayerNorm + residual
   inside the memory net: −1.22 ppl. Our FastMemory MLP has neither.
2. **Mini-batch inner updates, b=16 tokens: −1.70 ppl** (biggest single
   win). Gradients taken w.r.t. W from the END OF THE PREVIOUS mini-batch,
   then cumsum. Our 512-token chunks are far too coarse — use b=16-64.
3. **Learnable W₀** (meta-learned init, shared across sequences): needed
   for stability. FastMemory.reset() should restore a LEARNED init, not
   zeros.
4. **Learnable per-token inner LR** η(x)=η_base·σ(θ_lr·x): −0.36 ppl.
   Third independent confirmation (Titans θ_t, ARC-TTT lr search).
5. **Dual form**: matmul-only inner loop, >5x wall-clock — the trick that
   makes Task 6's meta-training affordable. Use for training/prefill;
   primal for decode.
6. Conceptual: TTT-Linear+batch-GD ≡ linear attention; our GDN backbone is
   already this family with delta-rule writes. At 2k context TTT ≈ Mamba ≈
   Transformer; the TTT advantage opens ≥8k — consistent with M0 (GDN fine
   at 2k) and with TTT layers being an ACCRETION-memory play, not a
   backbone play, at our context length.

### SEAL (2506.10943) — changes the sleep design, validates the gate

1. **Raw-context consolidation barely works — restructure first.** Their
   knowledge-incorporation numbers (Qwen2.5-7B, no-context SQuAD): base
   32.7; finetune on the raw passage 33.5 (+0.8, negligible); finetune on
   self-generated implications 39.7; GPT-4.1 implications 46.3; SEAL 47.0.
   **Sleep must distill RESTRUCTURED content (implications / atomic
   facts), not raw buffered contexts.** Task 7 gains a restructuring step:
   generate implications from the buffered context (the backbone itself
   can draft them), THEN distill into the substrate.
2. **For v2 write-policy training: ReST^EM, not GRPO/PPO.** They tried
   both; "found the training to be unstable." Rejection-sampling + SFT on
   the self-edits that actually improved the model (binary reward) is the
   stable recipe. Two iterations sufficed to beat GPT-4.1 as a data
   generator.
3. **Catastrophic forgetting is SEAL's open failure mode** (their Fig. 6:
   sequential self-edits progressively degrade prior tasks; no defense
   built). This is exactly the gap accretion's external gate + retention
   suite + rollback + sparse bucket-local writes exist to close — our
   sharpest positioning vs published work.
4. Their ARC instantiation: RL-selected TTT configuration hit 72.5% vs 20%
   without RL (Llama-3.2-1B) — the model can learn to CONFIGURE its own
   adaptation. v2 material for the skill lane.
5. Cost note: each self-edit evaluation ≈30-45s (finetune + eval inside
   the reward loop). Their context-dependent-eval limitation (needs paired
   QA) is already solved in our design by auto-generated retention probes.

### Cross-paper convergence (all four)

Every paper independently lands on a **data-dependent inner learning
rate** (Titans θ_t/η_t/α_t heads; TTT-layers learnable η(x); ARC-TTT lr
search per regime; SEAL self-edits choosing lr). Non-negotiable in Task 5.
Likewise 2/4 independently need a learned-init reset and LN+residual
inner structure. FastMemory v2 spec = Titans gates + TTT-layers inner
structure + MAG integration, meta-trained per Task 6, consolidated per
SEAL-restructuring, gated per our design.

## Deep-read round 3 (2026-08-01): Engram paper, Transformer², Nested Learning

### Engram (DeepSeek) — MECHANISTIC EXPLANATION OF THE M0 NULL; Task 2 rewrite

Real Engram differs from our M0 "Engram-lite" in six load-bearing ways:

1. **The gate compares memory content with the hidden state:**
   α = σ(RMSNorm(h)ᵀ RMSNorm(W_K e) / √d). Ours was σ(Linear(h)) —
   CONTENT-BLIND: it never saw what was retrieved, so it had no signal for
   when memory was relevant, and closing was the safe optimum. This alone
   plausibly explains the M0 null.
2. 2-gram AND 3-gram, K=8 hash heads per order, d_mem 1280, prime-sized
   tables, multiplicative-XOR hash; **tokenizer canonicalization** (NFKC +
   lowercase, −23% effective vocab) before hashing. Ours: raw GPT-2
   bigrams, 2 heads, 64d.
3. Value path: W_V projection + depthwise causal conv (kernel 4,
   dilation = max n-gram order) + SiLU + RMSNorm, conv zero-init.
4. Placement: layers 2 AND 15 of 30 — early site matters ("modeling favors
   early intervention to offload local pattern reconstruction"). Ours:
   single site at layer 8.
5. Table optimizer: Adam at **5x LR, no weight decay**.
6. Their iso-param result IMPROVED perplexity (val 1.634→1.622 vs MoE-27B)
   plus BBH +5.0, MMLU +3.4, HumanEval +3.0, and long-context MQ-NIAH
   84.2→97.0 (lookups free attention for global context). So the M0 null
   is evidence of OUR implementation gap, not concept failure.

Also: U-shaped sparsity allocation, optimum ≈20-25% of sparse budget in
memory, stable across scales; log-linear loss vs slot count in the
infinite-memory regime; CKA shows shallow Engram layers ≈ deeper baseline
layers (effective-depth gain); deterministic addressing → host-memory
offload at <3% overhead with a Zipfian HBM/DRAM/NVMe cache hierarchy —
which is literally accretion's "memory grows like storage" story, published.

**Task 2 change:** the bake-off's Engram substrate must implement the real
recipe (alignment gate, 2+3-grams, ≥4 heads, canonicalized ids, conv+SiLU
value path, sites at layer 2 + mid, 5x table LR) — otherwise Phase 1 would
re-condemn a strawman.

### Transformer² (Sakana) — SVF replaces LoRA as the skill-library primitive

SVF: learn one vector z per weight matrix scaling its singular values
(W' = U(Σ⊗diag(z))Vᵀ). ~40x fewer params than LoRA, full-rank reach,
principled regularization (only re-weights existing components), trainable
with REINFORCE+KL on BINARY rewards from a few hundred examples — and
**z-vectors compose by interpolation** (their CEM few-shot mixing), which
LoRA pairs cannot do. Their ablation: SVF+RL stable and effective where
LoRA+RL "consistently lagged" and next-token SFT underperformed RL.
Z-vectors even transfer across model families (ordered singular values
carry meaning; shuffling destroys it).

**Skill-lane update:** supervised episode adaptation (ARC-TTT style) →
LoRA r128 stays; RL-against-verifier skill training (our Phase 6 RLVR
machinery) → SVF parameterization. Sleep can MERGE related skills by
z-interpolation instead of retraining. Both slot into the same library.

### Nested Learning / Hope (Google) — the frame, one bookmark, still watch-list

Core lens: every component is an associative memory compressing its own
context flow at some update frequency — backprop is surprise-based memory
(maps inputs to their local error), momentum is a second-level memory of
gradients, attention updates per-token while MLPs update never (two
frequency extremes; the brain uses a spectrum). CMS = a chain of MLP
memory banks each with its own update frequency; Hope = self-modifying
Titans + CMS. Read pages 1-12 (paradigm + formalism); Hope/CMS equations
skimmed only — fine for watch-list status.

Two takeaways: (1) accretion's three tiers (per-token fast memory /
nightly sleep / frozen backbone) IS a coarse CMS with consolidation moved
outside the architecture — the design doc's positioning can cite this
frame directly. (2) **Bookmark: their M3 optimizer** — Muon with multiple
momentum timescales — a drop-in candidate for our trainer someday.

### Reading-completeness note

All 8 notebook sources now read: TTT-layers, ARC-TTT, SEAL, Engram,
Transformer² in full; Titans core (method + experiments); Nested Learning
core (paradigm; Hope details skimmed); SEAL blog subsumed by the paper.

## Deep-read round 4 (2026-08-01): Hope/CMS equations in full (pages 13-36)

User pushback was correct — "watch-list" did not justify skipping these.
Four concrete consequences:

1. **The retrofit path (§7.3) de-risks M1.** CMS blocks can be initialized
   from a PRE-TRAINED model's MLPs; the inner learning rate controls drift
   from them (η→0 = exactly pretrained behavior). They convert Llama3-3B/8B
   into continual learners with only ~15B tokens of continued pretraining,
   and it wins their class-incremental benchmarks. For us: initialize the
   fast tier at (or near) pretrained behavior and meta-learn ONLY the gate
   heads — stability by construction, and a fallback if from-scratch
   meta-training (Task 6) is unstable: retrofit the existing gdn checkpoint
   instead of retraining.
2. **CMS's forgetting resistance is architectural redundancy across
   frequencies**: when a fast block forgets, slower blocks still hold the
   knowledge, and backprop-through-inits can circle it back ("a loop
   through the time dimension"). Complementary to our external gate, and
   an argument for 2-3 frequency tiers of fast memory rather than one.
   Their level ablation: more levels monotonically better on ICL and
   long-context; lowest frequency 512 > 2K > 8K (adaptivity beats
   persistence at the margin, at compute cost).
3. **CTNL benchmark design upgrades Task 8.** Their Continual Translation
   of a Novel Language protocol: learn corpus A, then corpus B, then test
   BOTH — ICL collapses in the sequential setting (catastrophic in-context
   forgetting); Hope-3 nearly recovers per-corpus performance. Our
   learn-and-retain benchmark gains a phase-2: after the 100M-token
   retention run, inject a SECOND disjoint fact corpus and re-test the
   first — measures interference, not just decay.
4. **The sleep job's optimizer choice matters for forgetting (§4.3).**
   Standard momentum's effective memory is ~43 gradients (<1% contribution
   beyond); in sequential-task settings the optimizer itself drifts toward
   the current task's gradient direction and drives forgetting — "not the
   capacity of the model but the memory management of the optimization
   process." Their fixes: delta-rule momentum (gradient-dependent decay),
   DGD (Sherman-Morrison state-dependent weight decay, Eq 57), M3 (Adam +
   Muon + two-timescale chunked momentum, Alg 1). Sleep's consolidation
   optimizer should carry long-memory momentum; M3 is the candidate.

Also captured: self-referential Titans generates its own training targets
(v̂ = M(v)) with per-component memories for k, v, η (inner LR), α (forget
gate) — **only W_q stays non-adaptive**, a stabilization hint for Task 6
(adapt keys/values first, keep queries fixed). All memories are 2-layer
MLPs with residual (M(x) = x + W1 σ(W2 x)), matching the TTT-layers
finding. Chunk-wise dual-form training with gradients w.r.t. the previous
chunk's last state. Hope = self-modifying Titans -> CMS chain; the
Hope-Attention ablation (softmax attention + CMS) beats plain Transformer,
so CMS adds value independent of self-modification.

Not read: appendices (DGD derivation details, Adam-as-memory proof, M3
scaling caveats). Core is captured; appendices on demand.

## PHASE 1 VERDICT (2026-08-01): STOP for this design

| substrate | trained-phrasing recall | HELD-OUT recall | dval |
|---|---|---|---|
| engram (raw docs, 300 steps) | 0.0% | — | +0.0004 |
| engram (raw docs, 40k steps) | 12.5% | — | +0.0019 |
| **engram (SEAL-restructured, 40k)** | **25-33%** | **0.0%** | +0.022 |
| ple | 9.5% | — | +0.157 |
| pk | 0.0% | — | +0.032 |

**The gate fails on its own pre-registered terms.** The metric that matters
for accretion's claim is recall that GENERALIZES across phrasing, and that
is 0.0%. The 25-33% was format matching: train "the capital of X is",
probe "the capital of X is" -> hit; probe "Travellers to X usually land
first in" -> miss, every time, 200/200.

**Mechanism (this is the useful part, not a mystery):** Engram keys are
hashed token n-grams. A paraphrase produces different n-grams -> different
hash -> a different, untrained slot. Lexical keys cannot generalize by
construction. This is exactly the limitation flagged in deep-read round 2
("Engram cannot retrieve on a paraphrase") — now measured rather than
predicted.

**What IS established (keep these):**
- The backbone reads facts from context at 95% (D1) — the probe format is
  answerable; the task is well-posed.
- A substrate CAN drive doc loss 4.25 -> 0.0005 through a frozen backbone
  (D2) — the write path works; storage is not the problem.
- Real-recipe Engram is harmless to the backbone (dval +0.002 at 40k steps)
  where PLE is catastrophic (+0.157). Alignment gating works.
- SEAL restructuring replicates: 12.5% -> 33% on identical budget. The
  effect is real even though it is insufficient here.
- PLE eliminated on evidence (damages backbone, worse recall).

**What this does NOT establish:** that sparse memory cannot work. It shows
LEXICALLY-KEYED sparse memory cannot generalize. Semantic keys were the
designated fix and product-key memory was the designated test — but pk was
tested with zero-init values, which creates a chicken-and-egg (no useful
values -> no gradient pressure to make keys discriminative -> keys stay
random). pk's 0.0% is an invalid test, not a refutation.

**Recommendation:** accretion as designed does not proceed. Two honest
options, user's call:
 (a) STOP and write up. Deliverable: a specific, mechanistically explained
     negative about lexically-keyed memory in small models, plus an
     eliminated rival (PLE) and a validated consolidation technique (SEAL
     restructuring). Cost to date ~4 days, ~EUR 2.
 (b) ONE more experiment, tightly scoped: fix pk's init (small random
     values + brief key warmup) and re-run with held-out probe. This tests
     the semantic-key hypothesis that the whole diagnosis points at. If
     held-out recall stays ~0, (a) becomes unambiguous and better evidenced.

Do NOT: keep tuning engram. The failure is structural, not a
hyperparameter.

## SYNTHESIS ERROR FOUND (2026-08-01, prompted by user question)

Question asked: "wasn't there evidence where they used an already-trained
Llama backbone and it worked?" Chasing it down exposes a gap in my reading
of the four papers — I catalogued each mechanism but never cross-tabulated
them. Doing that now:

| paper | backbone | mechanism | worked? |
|---|---|---|---|
| SEAL | pretrained Qwen2.5-7B (frozen-ish) | **LoRA on restructured content** | YES 32.7->47% |
| ARC-TTT | pretrained Llama 1/3/8B | **per-task LoRA at test time** | YES 6x |
| Transformer^2 | pretrained Llama3-8B, Mistral-7B | **SVF (singular-value vectors)** | YES |
| Hope §7.3 | pretrained Llama3-3B/8B | CMS from MLPs + **15B tok continued pretrain** | YES |
| Engram | **trained from scratch, 262B tok, jointly** | sparse hash memory | YES |
| Titans / TTT-layers | trained from scratch | inner-loop memory | YES |

**Every success is either (1) update the model's own weights, or (2) train
the memory jointly from scratch. NOBODY bolts a sparse memory module onto a
frozen unmodified backbone.** That is exactly what Phase 1 tested. The
0% held-out result is therefore weak evidence about sparse memory in
general and strong evidence about an untested (and unsupported)
configuration we chose.

**Consequence — two evidence-backed paths replace the dead one:**

- **Path A (recommended, ~1 day): LoRA/SVF as the persistent substrate.**
  SEAL proves a pretrained model absorbs restructured knowledge this way.
  SEAL's OWN stated open problem is catastrophic forgetting under
  sequential edits — precisely what accretion's external gate + retention
  suite + rollback are for. The design's contribution is untouched; only
  the storage medium changes (hash buckets -> weight deltas). Structural
  locality is weaker than bucket-local writes, so the gate does more work
  — which is the honest cost, and testable.
- **Path B (350M run): Engram trained from scratch, jointly**, as DeepSeek
  actually did it. Retrofit was never their claim.

Pre-registered criteria for the Path A test (set BEFORE running, same
discipline as Phase 1): rank-8 LoRA on the frozen gdn model, 200 facts,
SEAL restructuring, held-out phrasing probe. PASS = held-out recall >=40%
with dval <= +0.02. Anything less and the substrate question is closed
regardless of medium.

Lesson recorded: reading N papers and summarizing each is not synthesis.
The cross-tabulation is where the finding was.

## PHASE 1 CLOSED (2026-08-01): pk retest with fixed init also 0% held-out

| substrate | trained-phrasing | held-out | dval |
|---|---|---|---|
| engram + SEAL restructuring | 25-33% | **0.0%** | +0.022 |
| pk, zero-init values (invalid test) | 0.0% | — | +0.032 |
| **pk, fixed init (keys DO get gradient)** | **0.0%** | **0.0%** | +0.021 |
| ple | 9.5% | — | +0.157 |

Lexical addressing and learned/semantic addressing fail identically. The
two designs fail for OPPOSITE reasons (hash cannot generalize; learned keys
had no signal to organize around) yet reach the same 0% held-out — so the
binding constraint is not the key mechanism. It is the configuration:
**a sparse memory module cannot be written into a frozen backbone that was
never trained to read from one.** Consistent with the synthesis finding
(no published work does this); now measured three ways.

Sparse-memory-on-frozen-backbone: CLOSED. Do not revisit without either
(a) meta-training the backbone, or (b) joint training from scratch.

Next: Path A (LoRA/SVF substrate, pre-registered >=40% held-out) — the
configuration SEAL actually validated. Then M1 meta-training.
