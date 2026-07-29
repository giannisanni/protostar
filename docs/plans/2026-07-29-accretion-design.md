# accretion: continual learning for protostar — design

**One line:** a protostar variant whose weights learn at inference — fast
Titans-style memory within a context, sleep-time consolidation into writable
Engram hash tables for life — with forgetting controlled by structure
(bucket-local writes) and by a gate (nightly val-loss + retention checks).

**Status:** design validated in brainstorm 2026-07-29. Blocked on M0 (the
ablation queue's engram arm result). Not yet planned into tasks.

## Goals, in order

- **v1 — learn-and-retain (the SOTA claim):** see a document once at
  inference; 100M+ unrelated tokens later, recall its facts with general
  ability (held-out val loss) unchanged.
- **v2 — improve with use.** v3 — personal deployment (vault/agents).

## Architecture

Backbone: protostar (winning ablation config), frozen after pretraining.
Learning lives entirely in a two-tier memory, both tiers feeding the
backbone through one gate at the block-8 site.

**Fast tier (volatile, per-context):** 2-layer MLP (d→2048→d, ~2.6M params),
an associative key→value map over residual-stream projections. Inner-loop
SGD *during the forward pass* on a surprise objective (memory's failure to
predict value from key), with momentum, weight decay as forgetting, and
per-chunk (512-token) updates for cheapness. Trained meta-style: the
pretraining run has the inner loop active, so the backbone learns to rely
on a memory that learns. At context end the MLP weights are a compressed
"what was new" diff → snapshotted with the context to a replay buffer, reset.

**Slow tier (persistent, lifetime):** the Engram hash tables, writable only
during sleep. A write touches one bucket per hash head — interference is
structurally bounded, which naive finetuning and LoRA merges cannot claim.

**Sleep (external job, nightly or buffer-triggered):** per buffer entry:
(1) replay with snapshot loaded, find positions where the gate was open —
their bigram hashes name the target buckets; (2) distill: optimize only
those bucket rows so tables alone reproduce the fast memory's signal
(same-gate co-location makes this a well-posed regression); (3) protect:
bucket usage counters (EWC-lite per bucket), two hash heads per fact;
(4) **gate: held-out val loss within ε AND all past retention probes pass,
else roll back the write batch** (tables ~130MB, snapshot/rollback free).
Sleep also prunes dead buckets. The gate stays external forever — the thing
being modified must not certify its own modification. Internalizing the
write policy (meta-trained consolidation) is v2; full in-architecture
nested optimization (Hope-style) is explicitly not attempted.

## Evaluation (v1 protocol)

K verifiable fact-docs provably absent from FineWeb → shown once at
inference day 0 → 100M+ tokens unrelated traffic, nightly sleeps → measure:
fact recall (fast memory OFF), general val loss, full retention suite.
Controls: frozen baseline (recall≈0) and naive nightly-LoRA (expected to
forget or degrade — demonstrating that is part of the claim).

## Staging

- **M0:** ablation queue engram arm result (running now; free).
- **M1 (Stage A):** accretion pretrain arm — inner loop active, 2B tokens,
  ~14h on the 5060 Tis. Pass = stable + beyond-window recall.
- **M2 (Stage B):** one full learn→sleep→retain cycle passes (sleep runs
  on the 4060 Ti, minutes per night).
- **M3 (Stage C):** the v1 benchmark at 100M+ tokens.

## Risks, ranked

1. Meta-learning instability (inner loop in pretraining) — mitigations:
   Titans' learned inner-LR, zero-init gates; Stage A tests it first.
2. Expressibility gap — bigram-keyed tables can't store everything fast
   memory learns; v1 claim scoped to factual/entity knowledge.
3. Lifetime bucket capacity — measurable; usage counters + pruning.
4. Field speed — the combo is hot; our edge is Engram + harness exist today.

## Provenance

Titans (fast tier recipe), DeepSeek Engram (slow substrate), Hope/Nested
Learning (tiering inspiration, explicitly not reproduced), Thinking
Machines' LoRA-without-regret (de-risks the deferred LoRA extension),
biology (sleep as distinct offline mode). SSI rumor check 2026-07-29:
nothing technical public; direction consistent with post-scaling bets.
