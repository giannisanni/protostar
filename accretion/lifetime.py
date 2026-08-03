"""The accretion claim: learn corpus after corpus, keep them all.

Protocol (CTNL, from Hope/Nested Learning): consolidate corpus A, then B,
then C — each disjoint — re-probing EVERY earlier corpus after every cycle.
Decay alone is not the question; INTERFERENCE from later learning is.

Reports a retention matrix:

              after A   after B   after C
    corpus A    58%       ?         ?        <- does A survive B and C?
    corpus B     -        55%       ?
    corpus C     -         -        61%

Run twice — `--gate` and `--no-gate` — because the gate can flatter itself:
if it rolls back every write, nothing is forgotten and nothing is learned.
The honest comparison is what interference looks like WITHOUT protection
versus WITH it.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accretion.bakeoff import forward  # noqa: E402
from accretion.facts import make_facts, restructure  # noqa: E402
from accretion.probe_ladder import LADDER  # noqa: E402
from accretion.registry import KnowledgeRegistry  # noqa: E402
from accretion.sleep import consolidate, recall  # noqa: E402
from accretion.varied import (make_varied_facts, varied_probe,  # noqa: E402
                              varied_restructure)
from accretion.substrates import build_substrate  # noqa: E402
from data.loader import Shards  # noqa: E402
from model import Config, Protostar  # noqa: E402

# L1 = light paraphrase. The headline metric: 100% in-context ceiling, so a
# score here is a clean read on retrieval rather than on inference ability.
def l1_probe(fact):
    return LADDER["L1_light"][fact["template"]].format(s=fact["subject"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--corpora", type=int, default=3)
    ap.add_argument("--facts-per-corpus", type=int, default=60)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--adaptive-replay", action="store_true",
                    help="scale replay with registry size instead of a fixed rate")
    ap.add_argument("--replay", type=float, default=0.0,
                    help="fraction of consolidation batches drawn from "
                         "previously committed facts (0.3 = 30%%)")
    ap.add_argument("--varied", action="store_true",
                    help="heterogeneous domains per corpus: disjoint relations, "
                         "vocabulary and phrasing (0.5%% bigram overlap vs 68%%)")
    ap.add_argument("--freeze-after", type=int, default=0,
                    help="after cycle N, write ONLY sparse slots "
                         "(shared projections frozen). 0 = never")
    ap.add_argument("--out", default="/mnt/data/protostar/lifetime")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    cfg = Config(**ck["cfg"])
    model = Protostar(cfg).to(a.device).eval()
    model.load_state_dict(ck["model"])
    del ck
    for p in model.parameters():
        p.requires_grad_(False)

    tag = ("nogate" if a.no_gate else "gate") + (f"_frozen{a.freeze_after}" if a.freeze_after else "") + ("_varied" if a.varied else "") + (f"_replay{int(a.replay*100)}" if a.replay else "") + ("_adaptive" if a.adaptive_replay else "")
    out = Path(a.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    registry = KnowledgeRegistry(out / "registry.jsonl")
    sub = build_substrate("engram", d=cfg.d, n_layer=cfg.n_layer).to(a.device)
    val = Shards(a.data, "val", 4, cfg.seq_len)

    @torch.no_grad()
    def val_loss_fn(substrate, n=8):
        val.pos = 0
        tot = 0.0
        for batch, _ in zip(val.batches(), range(n)):
            batch = batch.to(a.device)
            _, loss = forward(model, substrate, batch[:, :-1], batch[:, 1:])
            tot += loss.item()
        return tot / n

    # disjoint corpora: distinct seeds => distinct novel subjects
    if a.varied:
        corpora = [make_varied_facts(a.facts_per_corpus, domain=i, seed=100)
                   for i in range(a.corpora)]
        probe_fn, restr_fn = varied_probe, varied_restructure
    else:
        corpora = [make_facts(a.facts_per_corpus, seed=100 + i) for i in range(a.corpora)]
        probe_fn, restr_fn = l1_probe, restructure
    names = [chr(ord("A") + i) for i in range(a.corpora)]

    if a.freeze_after:
        print(f"shared projections FROZEN after cycle {a.freeze_after} "
              f"(writes become genuinely bucket-local)", flush=True)
    print(f"=== lifetime benchmark ({tag}) — {a.corpora} corpora x "
          f"{a.facts_per_corpus} facts, {a.steps} steps/cycle ===", flush=True)
    base_val = val_loss_fn(None)
    print(f"backbone val (no substrate): {base_val:.4f}", flush=True)

    matrix = {n: {} for n in names}
    for i, (name, corpus) in enumerate(zip(names, corpora), start=1):
        bucket_only = bool(a.freeze_after) and i > a.freeze_after
        consolidate(model, sub, corpus, registry, val_loss_fn, probe_fn, enc,
                    a.device, cycle=i, steps=a.steps, lr=a.lr,
                    gate=not a.no_gate, bucket_only=bucket_only,
                    restructure_fn=restr_fn, replay_frac=a.replay,
                    adaptive_replay=a.adaptive_replay, baseline_val=base_val)
        # re-probe every corpus seen so far — this is the interference test
        for j, (pname, pcorpus) in enumerate(zip(names, corpora), start=1):
            if j <= i:
                matrix[pname][f"after_{name}"] = round(
                    recall(model, sub, pcorpus, probe_fn, enc, a.device), 4)
        print(f"  retention matrix so far: "
              f"{json.dumps({k: v for k, v in matrix.items() if v})}", flush=True)

    print("\n=== RETENTION MATRIX (L1 light paraphrase) ===", flush=True)
    header = "corpus  " + "  ".join(f"after_{n}" for n in names)
    print(header, flush=True)
    for name in names:
        row = "  ".join(
            f"{matrix[name].get(f'after_{n}', float('nan')):7.1%}" if f"after_{n}" in matrix[name]
            else "      -" for n in names)
        print(f"  {name}     {row}", flush=True)

    # the number that matters: how much of corpus A survived everything after it
    first, last = f"after_{names[0]}", f"after_{names[-1]}"
    if first in matrix[names[0]] and last in matrix[names[0]]:
        a0, aN = matrix[names[0]][first], matrix[names[0]][last]
        print(f"\nFORGETTING: corpus A {a0:.1%} -> {aN:.1%} after {a.corpora - 1} "
              f"later consolidations (kept {aN / a0:.1%})" if a0 > 0 else
              "\nFORGETTING: corpus A never learned; nothing to measure", flush=True)
    print(f"final val drift: {val_loss_fn(sub) - base_val:+.4f}", flush=True)
    (out / "matrix.json").write_text(json.dumps(matrix, indent=2))


if __name__ == "__main__":
    main()
