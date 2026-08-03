"""One-shot associative writes: learning facts WITHOUT gradient descent.

The dial problem — every configuration trading knowledge against general
ability — is a consequence of the WRITE MECHANISM, not of memory itself.
Writing 60 facts took 12,000 AdamW steps (~200 per fact), and every step
dragged the shared substrate toward the fact corpus and away from English.

The fast memory (accretion/fast.py) does not need any of that. It is an
associative memory: give it a key and a value and it writes in ONE update
(delta rule, closed form). No optimizer, no loss, no repeated passes.
Crucially **no parameter changes at all** — only the matrix-valued state
moves — so there is no gradient pressure competing with general text.

We already built this and then threw it away: the v2 pretrain reset the
state after every sequence, using it as a volatile within-context tier.
Keeping the state makes it a persistent store.

Measures:
  recall        can a fact be retrieved after a single write pass
  drift         val loss with the written state vs the empty state
  interference  does corpus A survive writing B and C
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accretion.bakeoff import _norm, forward  # noqa: E402
from accretion.facts import make_facts  # noqa: E402
from accretion.probe_ladder import LADDER  # noqa: E402
from accretion.varied import (make_varied_facts, varied_probe,  # noqa: E402
                              varied_restructure)
from data.loader import Shards  # noqa: E402
from model import Config, Protostar  # noqa: E402


def l1_probe(fact):
    return LADDER["L1_light"][fact["template"]].format(s=fact["subject"])


@torch.no_grad()
def write_facts(model, facts, restructure_fn, enc, device, passes=1):
    """ONE forward pass per fact (or `passes`), no optimizer. The inner loop
    inside FastMemory performs the associative write as a side effect."""
    model.persist_memory = True
    for _ in range(passes):
        for f in facts:
            ids = torch.tensor(
                [enc.encode("\n".join(restructure_fn(f)))[: model.cfg.seq_len]],
                device=device)
            model(ids)  # the write IS the forward pass
    return model.fast.snapshot()


@torch.no_grad()
def probe(model, facts, probe_fn, enc, device, n_tok=6):
    hits = 0
    for f in facts:
        ids = torch.tensor([enc.encode(probe_fn(f))], device=device)
        for _ in range(n_tok):
            logits, _ = forward(model, None, ids[:, -model.cfg.seq_len:])
            ids = torch.cat([ids, logits[0, -1, :50257].argmax().view(1, 1)], 1)
        hits += _norm(f["object"]) in _norm(enc.decode(ids[0].tolist()[-n_tok:]))
    return hits / max(len(facts), 1)


@torch.no_grad()
def val_loss(model, loader, device, n=8):
    loader.pos = 0
    tot = 0.0
    for batch, _ in zip(loader.batches(), range(n)):
        batch = batch.to(device)
        _, loss = forward(model, None, batch[:, :-1], batch[:, 1:])
        tot += loss.item()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--corpora", type=int, default=3)
    ap.add_argument("--facts-per-corpus", type=int, default=60)
    ap.add_argument("--passes", type=int, default=1, help="write passes per fact")
    ap.add_argument("--varied", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    cfg = Config(**ck["cfg"])
    assert cfg.fast_memory, "checkpoint has no fast memory — use the v2 model"
    model = Protostar(cfg).to(a.device).eval()
    model.load_state_dict(ck["model"])
    del ck
    for p in model.parameters():
        p.requires_grad_(False)

    if a.varied:
        corpora = [make_varied_facts(a.facts_per_corpus, domain=i, seed=100)
                   for i in range(a.corpora)]
        probe_fn, restr_fn = varied_probe, varied_restructure
    else:
        from accretion.facts import restructure
        corpora = [make_facts(a.facts_per_corpus, seed=100 + i) for i in range(a.corpora)]
        probe_fn, restr_fn = l1_probe, restructure
    names = [chr(ord("A") + i) for i in range(a.corpora)]
    val = Shards(a.data, "val", 4, cfg.seq_len)

    print(f"=== ONE-SHOT ASSOCIATIVE WRITES ({'varied' if a.varied else 'homogeneous'}) "
          f"— {a.corpora}x{a.facts_per_corpus} facts, {a.passes} pass(es), NO optimizer ===",
          flush=True)

    # baseline: empty memory
    model.persist_memory = False
    model.fast.reset(1, a.device, torch.float32)
    base_val = val_loss(model, val, a.device)
    print(f"backbone val, empty memory: {base_val:.4f}", flush=True)

    model.persist_memory = True
    model.fast.reset(1, a.device, torch.float32)
    matrix = {n: {} for n in names}
    import time
    for i, (name, corpus) in enumerate(zip(names, corpora), start=1):
        t0 = time.time()
        write_facts(model, corpus, restr_fn, enc, a.device, passes=a.passes)
        dt = time.time() - t0
        for j, (pname, pcorpus) in enumerate(zip(names, corpora), start=1):
            if j <= i:
                matrix[pname][f"after_{name}"] = round(
                    probe(model, pcorpus, probe_fn, enc, a.device), 4)
        drift = val_loss(model, val, a.device) - base_val
        so_far = "  ".join(f"{k}={matrix[k][f'after_{name}']:.0%}"
                           for k in names[:i])
        print(f"[wrote {name}] {len(corpus)} facts in {dt:.1f}s "
              f"({dt/len(corpus)*1000:.0f}ms/fact)  drift={drift:+.4f}  {so_far}",
              flush=True)

    print("\n=== RETENTION MATRIX ===", flush=True)
    print("corpus  " + "  ".join(f"after_{n}" for n in names), flush=True)
    for name in names:
        row = "  ".join(
            f"{matrix[name][f'after_{n}']:7.1%}" if f"after_{n}" in matrix[name]
            else "      -" for n in names)
        print(f"  {name}     {row}", flush=True)
    first, last = f"after_{names[0]}", f"after_{names[-1]}"
    a0, aN = matrix[names[0]][first], matrix[names[0]][last]
    print(f"\nFORGETTING: corpus A {a0:.1%} -> {aN:.1%} "
          f"(kept {aN/a0:.1%})" if a0 > 0 else "\ncorpus A never written", flush=True)
    print(f"FINAL DRIFT: {val_loss(model, val, a.device) - base_val:+.4f}  "
          f"(zero parameters changed)", flush=True)


if __name__ == "__main__":
    main()
