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
pytest -q                                                     # 60 tests
```

### GPU layout on substrate (with `PCI_BUS_ID` set, verified 2026-08-02)

| CUDA id | card | PCIe under load | use for |
|---|---|---|---|
| 0 | 5060 Ti | Gen5 x8 | single-GPU jobs — fastest |
| 1 | 4060 Ti | Gen4 x4 | serving / spillover |
| 2 | 5060 Ti | Gen5 **x1** (crippled slot) | training pair only |

Idle cards report Gen1 because PCIe downtrains at low power; **width** is the
real number. Data lives at `/mnt/data/protostar/fineweb10B` (10.3B tokens).

Plans, verdicts and retractions: `docs/plans/`.
