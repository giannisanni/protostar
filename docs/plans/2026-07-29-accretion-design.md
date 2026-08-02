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

**Skill lane (deferred): hotswappable LoRA library.** When sleep detects a
cluster of related experience, it trains a per-domain skill adapter and adds
it to a library; inference loads adapters on demand (one at a time first —
LoRA composition interferes). Facts → tables, skills → adapter library,
thinking → frozen backbone. Prior art in-house: Gianni's
TITANS-implementation repo (June 2026, GPT-2-124M) hit both failure modes
this design fixes — raw persistence without gated consolidation (unsustained
learning, degradation loops) and a backbone never meta-trained to use its
memory. Its repetition-based learning test prefigures the retention suite.

## Boundary: what is IN the model vs what is HARNESS

Split by timescale: per-token and differentiable goes in the model; rare,
expensive, judgment-requiring goes in the harness.

**In the model (weights / forward pass):** frozen backbone; fast-memory MLP
incl. its inner-loop update (learning happens *inside* inference); the gate
mixing memory into the residual; Engram tables (read every forward, written
only at sleep); and — least obvious — the meta-learned expectation that a
changing memory exists, which is what [[TITANS-implementation]] lacked and
had to fake with a 1000x embedding magnitude hack.

**In the harness (external / offline):** the sleep job (replay, locate
contributing positions, distill to bucket writes); the replay buffer;
bucket usage counters, pruning, table snapshots; the retention probe suite;
scheduling.

**Migration across versions:** v1 = policy external. v2 = the *write policy*
meta-learned into weights, executed offline (the novel contribution).
v3 = sleep *pressure* becomes a learned signal (model chooses when).
**Never migrates: the verification gate** — the thing being modified must
not certify its own modification. Judgment moves inward over time;
verification stays outward permanently.

## Skills: what accretion can and cannot learn (clarified 2026-07-31)

Three bands:
1. **Fact-shaped skills — YES (v1):** mappings, vocabulary, API/symbol
   names, conventions, templates, when-X-do-Y rules. Key->value shaped,
   exactly what tables + fast memory store. Most day-to-day "it got better
   at being my assistant" lives here.
2. **Skill-by-scaffolding — partial, free:** persistent memory acts as an
   automatic standing few-shot prompt; the backbone executes better from
   better cues in familiar domains. Real but ceilinged by frozen-backbone
   competence.
3. **Procedural skills — NO in v1, by design:** fluent translation, new
   grammars, better reasoning = circuits in dense weights; nothing writes
   dense weights. Path = the sleep-clustered LoRA library (v2), open
   questions: per-skill data accumulation rate, per-skill eval for the
   gate, 350M ceiling. Meta-capability (better thinking) never comes from
   accretion at all — that is pretraining/distillation/RL (Phase 7).

**Benchmark addition — skill probe:** alongside fact-docs, introduce an
invented convention/mini-DSL via documents at inference; later test
EXECUTION (not recall) with fast memory off. Locates the fact/skill
boundary empirically for this architecture.

## Adaptable-backbone research (2026-07-31) — the procedural-skill lane has evidence

- **Meta-adaptability finetuning (ADOPT at 350M):** ARC test-time-training
  result (Akyurek et al., 2411.07279): few gradient steps on a per-task
  LoRA at inference = 6x accuracy, human-average ARC at 8B — but ONLY with
  a backbone meta-finetuned to be adaptable first. Adaptability is a
  trainable property; add that stage. Upgrades the skill-LoRA lane from
  hopeful to recipe-backed.
- **On-demand skill learning:** per-task TTT-LoRA when few examples of a
  new task appear; sleep decides which ad-hoc adapters graduate into the
  permanent library. Closes session-learning -> kept-skill.
- **TTT layers (Sun et al., ICML 2025, code public, proven 125M-1.3B):**
  layers whose hidden state is a small net updated by SGD per token — the
  fast tier generalized to the whole backbone. ESCALATION PATH if the
  single memory module proves too weak, not the default (bigger surgery,
  more compute/token).
- **SEAL (MIT, 2506.10943):** model generates its own finetune data +
  update directives, RL-rewarded on post-update performance = our v2
  write-policy internalization, published. Adopt its recipe when v2 lands.
  (Sakana Transformer-squared: fixed expert-vector mixing, less relevant.)
- Caveat: ARC numbers are 8B; 350M learns modest skills. Mechanism
  transfers; expectations should not inflate.

## Vector search (turbovec): harness yes, memory no

Using an ANN index AS the memory = RAG, which is a different system:
knowledge stays outside as text, costs context, needs a search before
generation, and must be reasoned over as quoted tokens. Accretion writes to
parameters: zero context cost, O(1) in-forward lookup, available at every
layer. RAG = looking it up; accretion = having learned it. Complements, not
competitors — retrieval keeps the long tail, verbatim, and fresh/auditable
facts.

**Where turbovec (Rust, TurboQuant, ~8x compression, already used in
lcovec) does fit — all harness-side:** (1) sleep-time dedup, "does the
model already know this?" before consolidating; (2) retention-probe
selection, test the probes semantically nearest the write batch instead of
the whole suite; (3) replay diversity sampling.

**Real limitation it highlights:** Engram's bigram hash keys on exact token
pairs — it cannot retrieve on paraphrase. The fix is NOT an external index
but **product-key memory** (Lample et al.; Meta's Memory Layers at Scale):
factorized learned keys, top-k over each half, huge key space at a fraction
of the comparisons — semantic, differentiable, still inside the forward
pass. That is the upgrade path if the ablation shows hashing is too rigid.

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
3. Lifetime bucket capacity / small-model saturation — measurable (usage
   counters + collision rate = live memory-pressure gauge), and structurally
   solvable: the backbone is frozen, facts live in tables, and table
   capacity is a storage knob (grow buckets 4x = +130MB, no retraining;
   sparse access permits RAM offload per Gemma PLE). What cannot grow this
   way is backbone *skill* — hence v1's facts-only claim scoping, the LoRA
   extension for the skill lane, and the unchanged port to 350M later.
4. Field speed — the combo is hot; our edge is Engram + harness exist today.

## End-state extensions (validated 2026-07-29, v2/v3 — not in v1 scope)

- **Sleep pressure / model chooses when to sleep:** expose fast-memory
  saturation (weight-norm vs decay equilibrium), surprise integral, and
  buffer depth as input features; tiny policy head requests sleep. v1's
  buffer-trigger is the primitive version. Gate stays external always —
  model picks WHEN, never certifies the writes.
- **Internal clock:** absolute + delta timestamps embedded per context;
  memory ages by wall-clock, not tokens. Perceives elapsed time.
- **Instant wake:** inference process stays resident (serve card); sleep
  job preemptible on the second GPU. Wake = normal latency + injected
  "while you slept" preamble (what consolidated, time elapsed).
- **Dreaming (sleep-time compute, cf. Letta 2025):** idle-time generation
  connecting new memories to old knowledge; elaborations enter
  consolidation as synthetic replay.
- **Workspace-gated consolidation (J-space, Anthropic July 2026):** add a
  J-lens-style Jacobian probe over the residual stream; select memory
  writes by what occupied the model's global workspace, not surprise
  alone (surprise = unexpected; workspace = integrated; use both).
  Reportability gives an audit trail: the model reports what it learned,
  gate checks the report against probes. Believed unpublished as a
  consolidation mechanism — strongest novelty claim in the design.

## Provenance

Titans (fast tier recipe), DeepSeek Engram (slow substrate), Hope/Nested
Learning (tiering inspiration, explicitly not reproduced), Thinking
Machines' LoRA-without-regret (de-risks the deferred LoRA extension),
biology (sleep as distinct offline mode). SSI rumor check 2026-07-29:
nothing technical public; direction consistent with post-scaling bets.

## Design principle: the knowledge registry (added 2026-08-02)

Prompted by LaurieWired's CallMeMaybe (github.com/LaurieWired/CallMeMaybe),
a C++26 library that captures compile-time reflection metadata into a
RUNTIME registry so a program can look itself up by name and invoke itself
dynamically. Her framing is the transferable part, not the technique: a
running program has no self-knowledge, which is why reverse engineering
exists at all ("Polaroid vs two-way mirror").

**The technique does NOT transfer.** CMM reflects on *type structure* —
names, signatures, member offsets. A transformer has almost no type
structure worth reflecting on, and Python/PyTorch already expose far more
(`named_modules`, `named_parameters`, hooks) than CMM provides. Anyone
proposing "runtime reflection for LLMs" as a direct port is pattern-matching
on the phrase, not the mechanism.

**The thesis DOES transfer, and we need it.** Accretion's sleep phase must
answer "what do I already know?" before deciding what is worth writing.
Today our only method is behavioural probing — generate text, see if the
right token appears. That is reverse-engineering our own model, and it is
exactly the blindness CMM complains about.

**So: memory writes get a registry, from the start of Phase 2 — not bolted
on later.** Every consolidation records:

| field | why |
|---|---|
| content hash / fact key | dedup: has this been written before? |
| source episode id | provenance: what caused this write |
| timestamp + sleep cycle no. | ordering, and age-based decay policy |
| pre/post val loss, probe results | the gate's evidence, retained |
| substrate slots touched | rollback target; interference analysis |
| confidence / surprise at write | prioritisation, and later pruning |

This is not gold-plating — the design ALREADY requires most of it. The
retention probe suite needs to know what was written to know what to probe;
the rollback gate needs to know which slots to restore. Building it as an
explicit queryable structure instead of an implicit side effect is what
turns "the gate works" into "we can explain why the gate works", and it is
the difference between a demo and a result someone else can check.

**Concrete first use:** sleep queries the registry for near-duplicate
content hashes before consolidating. Cheap dedup, no model inference
needed, and it directly attacks the fact-to-fact interference that capped
Phase 1's engram arm at 12.5%.
