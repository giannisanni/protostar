# protostar v2: memory-native — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Pretrain a 134M model whose memory exists *from step one* — so the backbone learns to rely on a memory that changes — then prove it learns facts at inference and keeps them.

**Architecture:** M0's winning backbone (deep-thin, 3:1 Gated DeltaNet, Muon, WSD) with two memory tiers trained jointly rather than retrofitted: real-recipe Engram (persistent, sparse) and a Titans-style fast memory whose inner loop runs *during pretraining* (volatile, per-context).

**Why this exists:** Phase 1 closed the retrofit route with three measurements (engram 0% held-out, pk 0%, ple 0%). Cross-tabulating the literature showed why — every published success either updates the model's own weights or trains memory jointly from scratch. Nobody bolts memory onto a frozen backbone. v2 is the joint-training branch.

---

## Field check (2026-08-01): is someone already doing this?

Searched before committing. Findings:

- **Nobody has shipped the combination.** Frontier models still degrade
  **15-32%** on continual finetuning sequences (Llama 4, GPT-5.1, Opus 4.5,
  Gemini 2.5 Pro, DeepSeek-V3.1). Catastrophic forgetting is explicitly an
  unsolved *deployment* problem in 2026, not a solved one.
- **The field split into two camps, and both avoid our problem:**
  (a) *token-space memory* — Letta's stateful agents, A-MEM, the
  `letta-ai/learning-sdk` (Feb 2026). These curate the context window. No
  weights change; nothing is learned, only retrieved.
  (b) *disposable adaptation* — MIITA (2607.22556) does inference-time
  adaptation for small LMs but is explicitly **"disposable, requiring no
  permanent parameter updates"**. End-to-End TTT compresses context to
  weights but per-context, not permanently.
- **So the gap is exactly where we aimed:** persistent weight-level learning
  at inference with a forgetting defense. TFGN (2605.15053) does 0%-replay
  continual *pretraining*, closest neighbour but corpus-scale, not
  per-interaction.
- **Adjacent art to respect:** Meta's sparse memory finetuning; Google's
  Titans; "Do Self-Evolving Agents Forget?" (2605.09315) documents the
  degradation we intend to prevent.

Conclusion: proceed. The niche is real and unoccupied. Positioning is
"permanent, gated, per-interaction learning" against a field doing either
retrieval or disposable adaptation.

---

## Task 1: Fast memory module (paper-correct)

**Files:** Create `accretion/fast.py`, `tests/test_fast.py`

Four papers independently converge on a **data-dependent inner learning
rate**; TTT-layers measured the rest of the ladder. All of it goes in:

```python
class FastMemory(nn.Module):
    """Titans/TTT-correct fast memory. Every element below is a measured
    ablation win, not a design preference:
      - inner net f(x) = x + LN(MLP(x))          TTT: -1.22 ppl
      - mini-batch inner updates (b=16..64)      TTT: -1.70 ppl (biggest)
      - learnable per-token eta/alpha/theta      all 4 papers converge
      - meta-learned W0 to reset to (not zeros)  TTT: stability
      - keys/values from learned projections     Titans
    """
```

**Step 1 — failing tests** (`tests/test_fast.py`):
- `test_surprise_falls_on_repetition`: same chunk 20x -> final surprise < 0.8x first.
- `test_reset_restores_learned_init`: after updates, `reset()` returns params to `W0`, and `W0` is a `nn.Parameter` (meta-learned), not zeros.
- `test_gates_are_data_dependent`: two different inputs produce different eta/alpha (a constant-gate implementation fails this).
- `test_snapshot_roundtrip`: snapshot -> reset -> load restores exactly.

**Step 2** run: `.venv/bin/python -m pytest tests/test_fast.py -q` -> FAIL (no module).
**Step 3** implement. **Step 4** tests pass. **Step 5** commit `feat(v2): titans-correct fast memory`.

---

## Task 2: Wire both tiers into the model behind flags

**Files:** Modify `model.py`, create `configs/v2.json`

Config additions: `fast_memory: bool`, `fast_hidden: int = 2048`,
`fast_chunk: int = 32`, `engram: bool` (real recipe from
`accretion/substrates.py`), `mem_integration: str = "mag"`.

**MAG integration, not MAL.** Titans measured memory-as-gate (parallel
branch, learned combination) beating memory-as-layer (in-line residual add).
Phase 1 used MAL-shaped injection. v2 uses:
`h = h + gate(h) * memory_branch(h)` with the gate seeing both.

**Step 1 — failing tests** (`tests/test_v2_model.py`):
- `test_param_count_in_class`: 120M < dense params < 150M (tables excluded, reported separately).
- `test_fast_memory_updates_during_forward`: memory params change after a forward pass in train mode.
- `test_zero_init_is_noop`: at init, v2 output == plain gdn output (so pretraining starts from a known-good point).
- `test_causality_holds_with_both_tiers`.

**Step 5** commit `feat(v2): memory tiers wired into backbone`.

---

## Task 3: Meta-training run (M1 — the load-bearing bet)

**Files:** Modify `train.py` (inner loop active during pretraining)

The inner loop must run *inside* the training forward pass, with the outer
optimizer updating projections and gate heads but **not** backpropagating
through the inner updates (detach inner state between chunks — TTT's
chunk-wise dual form).

Run: 2B tokens, `CUDA_VISIBLE_DEVICES=1,2`, same budget as the M0 arms so
the comparison is controlled.

**Pre-registered pass criteria (set before running):**
| check | threshold |
|---|---|
| training stable | no loss spike >2x, run completes |
| quality not sacrificed | val loss within **+0.02** of gdn's 3.2589 |
| memory actually used | mean gate activation > 0.05 (not learned-closed) |

Fail on stability -> fall back to the **Hope §7.3 retrofit**: initialize
from the existing gdn checkpoint, continued-pretrain with memory active,
inner LR small so it starts near-pretrained behaviour.

---

## Task 4: The claim — learn and retain

**Files:** `accretion/benchmark.py`

Reuse Phase 1's harness (it works and is honest). Protocol:
1. Show K fact-episodes **once** at inference, SEAL-restructured.
2. Consolidate via sleep (Phase 2 work) into Engram tables.
3. Run 100M+ tokens of unrelated traffic.
4. Measure **with fast memory OFF** — the persistent tier must carry it.

**Metrics, all three reported:** trained-phrasing recall, **held-out
phrasing recall** (the one that counts — Phase 1 proved the other can lie),
general val loss.

**Plus the CTNL interference phase** (from Hope): learn corpus A, then a
disjoint corpus B, re-test A. Measures interference, not just decay.

Pass = held-out recall >= 40% with dval <= +0.02, retained after B.

---

## What v2 does NOT fix

At 134M this learns **facts, not skills** — that ceiling is backbone size,
not memory design. The 350M rerun is the version worth using; v2 is the
proof the machinery works. Say so in any write-up.

## PRE-REGISTERED: the v2 memory test (set 2026-08-02, BEFORE M1 finished)

Registered before any v2 number exists, so the bar cannot move to fit the
result. Same discipline as Phase 1 — which is what caught v1's fake 33%.

### Test 1 — M1 gate criteria (pretraining, decides if v2 is even hostable)

| check | threshold | why |
|---|---|---|
| stability | run completes, no NaN | the k/q-norm fix must hold over 2B tokens |
| quality tax | val loss <= 3.2789 (gdn 3.2589 + 0.02) | memory must not cost general ability |
| **memory USED** | **mean gate activation > 0.05** | a backbone that learned to ignore memory is a silent failure |

### Test 2 — the actual claim (run immediately after M1)

**Reuse `accretion/bakeoff.py` unchanged** against the v2 checkpoint. Same
200 novel-subject facts, same SEAL restructuring, same probes. Identical
code and data to the v1 run means the A/B is exact — only the model differs.

| metric | v1 result (frozen backbone) | v2 PASS threshold |
|---|---|---|
| **held-out phrasing recall** | **0.0%** | **>= 40%** |
| trained-phrasing recall | 25-33% (was fake) | reported, NOT a pass criterion |
| dval | +0.022 | <= +0.02 |

**Held-out is the only number that counts.** v1 proved trained-phrasing
recall can read 33% while true recall is 0 — that failure mode is exactly
what this threshold exists to catch.

### Why v2 might genuinely differ

v1 wrote into a substrate the backbone had never seen, addressed by lexical
token hashes — a paraphrase hashes to a different, untrained slot, so 0%
was structurally guaranteed. v2's memory is **content-addressed** (learned
k/v/q projections over hidden states) and the backbone spent 2B tokens
learning to read from it. A paraphrase yields different tokens but
**similar hidden states** — the exact property hashing lacked.

### Why it might still fail

The v2 fast memory is a per-sequence matrix state, reset each forward. M1
proves in-context use; it says nothing about carrying memory ACROSS
contexts or consolidating it permanently. That is Phase 2/3 and does not
exist yet.

### Decision rule

- held-out >= 40% -> mechanism generalizes; build sleep/consolidation (Phase 2)
- held-out 10-40% -> partial; ONE tuning round, then decide
- held-out < 10% -> joint training falsified too. Write up the negative:
  two pre-registered, mechanistically-explained failures of small-model
  memory (retrofit AND joint), which is a real contribution.
