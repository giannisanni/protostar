# protostar

A 134M-parameter language model trained from scratch on two consumer GPUs, plus
**accretion** — a memory system that lets it keep what it learns across
successive consolidations instead of overwriting it.

**Headline result:** after learning three disjoint fact corpora one after
another, the first corpus retains **77.8%** of its recall (homogeneous
phrasing) / **76.5%** (varied phrasing) — versus **12.5% / 0.0%** without
replay. The 50% threshold was pre-registered before the runs.

Everything below was measured on this hardware. Failures are documented
alongside successes; two verdicts in the history were retracted after better
measurement, and both retractions are kept in the plan docs.

---

## The model

| | |
|---|---|
| params | 134M (24 layers x 640d, deep-thin) |
| attention | 3:1 Gated DeltaNet : full attention (Qwen3.6 layout) |
| other | GQA + QK-norm, ReLU² MLP, iRoPE (every 4th layer NoPE), tied embeddings |
| optimizer | Muon (hidden matrices) + AdamW (embeddings/head), WSD schedule |
| data | FineWeb, 2B tokens, GPT-2 tokenizer |
| hardware | 2x RTX 5060 Ti, bf16 DDP, ~36.5k tok/s |

### Architecture ablation (2B tokens each, identical data and seed)

| config | val loss | Δ vs dense | tok/s |
|---|---|---|---|
| **gdn** | **3.2589** | **−0.0273** | **37,121** |
| gdn + MTP | 3.2714 | −0.0148 | 28,514 |
| gdn + MTP + engram | 3.2798 | −0.0064 | 29,878 |
| dense | 3.2862 | — | 36,252 |

Gated DeltaNet won on quality *and* speed. MTP and Engram-lite both cost
quality at this scale — every addition on top of GDN made things worse.

### Throughput

| setup | tok/s | scaling |
|---|---|---|
| 1x 5060 Ti | 19,600 | — |
| 2x 5060 Ti DDP, accum 16 | 33,800 | 1.73x |
| + bf16 grad compression hook | 36,400 | 1.86x |

2B-token arm ≈ 14h. Not yet pulled: torch.compile, activation checkpointing.

---

## accretion

Teach the model facts at consolidation time, keep them across later
consolidations, without degrading general ability.

### Components

| module | what it does |
|---|---|
| `accretion/sleep.py` | consolidation: restructure → distil → gate → commit or roll back |
| `accretion/registry.py` | append-only log of every write and its evidence; content-keyed dedup |
| `accretion/substrates.py` | three memory designs behind one interface (engram / PLE / product-key) |
| `accretion/lifetime.py` | the benchmark: sequential corpora, interference matrix |
| `accretion/probe_ladder.py` | recall at four paraphrase difficulties + memory-free ceiling |
| `accretion/fast.py` | TTT-Linear fast memory (used in the v2 pretrain) |
| `accretion/oneshot.py` | one-shot associative writes: facts by forward pass, zero parameter change |
| `accretion/curriculum.py` | synthetic retrieval curriculum: facts probed 1-20 windows later, leak-tested |
| `accretion/currprobe.py` | in-distribution verifier: memory contribution over 1,000+ probes |

### Three findings that shaped it

**1. Restructure before you distil.** Consolidating one surface form barely
works; consolidating 5-7 restructured variants of the same fact more than
doubles recall (12.5% → 33%). Replicates SEAL's central result.

**2. Interference is shared *capacity*, not slot collision.** Proven by a 2x2:

| | projections learnable | projections frozen |
|---|---|---|
| homogeneous (56% bucket overlap) | 8.3% | 11.7% |
| varied (1.3% bucket overlap) | **0.0%** | **0.0%** |

Near-disjoint buckets forgot *more*. The substrate's shared dense projections
are retuned by every write regardless of which slots a fact occupies.

**3. Replay is what works.** 30% of each consolidation batch drawn from
previously committed facts, sourced from the registry. It keeps old facts in
the gradient signal, so shared projections must serve old and new at once.

### Retention matrix (homogeneous, replay 30%)

| corpus | after A | after B | after C |
|---|---|---|---|
| A | 75.0% | 71.7% | **58.3%** |
| B | — | 48.3% | 41.7% |
| C | — | — | 71.7% |

All three end at comparable recall — holding multiple corpora, not
overwriting.

### A val-loss gate is not enough

Three times, a consolidation destroyed 75%+ of prior knowledge while
validation loss stayed flat or *improved*. Only the retention suite caught it.
The gate checks held-out loss **and** re-probes every previously committed
fact, restoring a bit-exact snapshot on failure.

### Costs, stated plainly

- Absolute recall is modest: ~22% (varied), ~58% (homogeneous).
- Replay costs plasticity when corpora are dissimilar (new-corpus recall
  13-27% vs 22-38% without).
- Val drift *rises* with replay (+0.10) — the gate matters more, not less.
- At 134M this learns **facts, not skills**.

---

## v2: the memory-native line (one-shot writes)

The retention/plasticity/drift dial above is a property of *gradient-descent*
writes. v2 pretrained the backbone jointly with the TTT-Linear fast memory
(gated parallel branch at layer 12) so a fact could instead be written by a
single forward pass — 14ms/fact, **zero parameters changed**, nothing to
trade. Three rounds, each pre-registered, each ~393M tokens / 3.5h:

| round | change | outcome |
|---|---|---|
| one-shot on v2 | keep the state across sequences at inference | 0% recall; old state read as **noise** (drift +0.42). Pretraining reset the state every sequence, so cross-sequence states are out-of-distribution |
| stateful phase | per-lane contiguous streams; state persists through training | **tolerance learned, retrieval not**: drift +0.42 → +0.003, but recall 0% even for one fact one sequence old (within-sequence recall still 100%) |
| retrieval curriculum | 25% of lanes carry fact streams probed 1-20 windows later (leak-tested; vocab disjoint from eval) | still ~0%. Verified **in-distribution**: over 1,178 probes, carried memory lifts answer-token accuracy just **+2.2%** vs memory wiped every window |

General-ability val was unaffected in both training rounds (3.46 in each) —
the curriculum was free; it just taught nothing.

**Why (structural):** TBPTT detaches the memory state at every window
boundary, so a probe's loss can never reach the projections that *wrote* the
fact — cross-window writes are untrainable by construction. Only the read
side gets gradient, and it must decode a single decaying 640×640 linear map
holding dozens of interfering facts.

**Open levers** — design changes, not more tokens: (a) auxiliary readback
loss at write time (make the state decodable, self-supervised); (b) BPTT
through several windows of state; (c) slot/product-key state instead of one
matrix. Until one is built and measured, **replay + gate above remains the
working system**.

---

## Methodology

Two probe-design errors, opposite directions, both caught before they became
conclusions:

- A trained-phrasing metric read **33%** while true recall was **0%** — the
  model had memorised the question format. Caught by a held-out probe
  committed to before the run.
- A later verdict declared the approach falsified at **0%** held-out recall —
  but that probe was the hardest rung of a ladder whose ceiling, measured on
  the unassisted backbone, was 30%. The pass threshold had been set at 40%:
  unpassable by construction. Verdict retracted.

**Standing rule: calibrate every probe against the unassisted backbone before
setting a threshold, and report the full ladder, never a single number.**

Four mechanism hypotheses were proposed; three were wrong. Each died to a
measurement rather than an argument.

---

## Running it

```sh
# ALWAYS set PCI_BUS_ID — PyTorch defaults to FASTEST_FIRST, which does not
# match nvidia-smi and silently selects a different GPU. This cost us a
# bake-off on the wrong card and an OOM against a GPU shown as empty.
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROTOSTAR_DATA=/mnt/data/protostar python env/check_env.py   # preflight
sh scripts/ablate.sh                                          # ablation queue
python accretion/probe_ladder.py --ckpt <ckpt>                # recall vs paraphrase
python accretion/lifetime.py --ckpt <ckpt> --varied --replay 0.3   # the claim
python accretion/oneshot.py --ckpt <ckpt> --data <data>       # one-shot writes
python accretion/currprobe.py --ckpt <ckpt>                   # memory contribution
pytest -q                                                     # 73 tests
```

### GPU layout on substrate (with `PCI_BUS_ID` set, re-verified 2026-11-12)
| CUDA id | card | PCIe width | notes |
|---|---|---|---|
| 0 | 5060 Ti | x8 | |
| 1 | 5060 Ti | x8 | the old x1 slot is fixed |
| 2 | 4060 Ti | x4 | normally om-model (vLLM :8607) + mentat |

Idle cards report Gen1 because PCIe downtrains at low power; **width** is the
real number. Data lives at `/mnt/data/protostar/fineweb10B` (10.3B tokens).

Plans, verdicts and retractions: `docs/plans/`.
