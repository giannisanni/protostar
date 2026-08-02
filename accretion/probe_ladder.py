"""Diagnose WHY held-out recall was 0%: probe difficulty, or brittle representations?

Two orthogonal tests on the same checkpoint:

A) PROBE LADDER — facts written into a substrate, then queried at four
   difficulty levels. Locates exactly where retrieval breaks:
     L0 identical   the trained sentence, verbatim          (sanity)
     L1 light       same words, reordered/inflected         (lexical overlap high)
     L2 moderate    synonyms, same relation stated directly (semantic, no inference)
     L3 hard        the current held-out probes — needs an inference hop
   If L1/L2 work and only L3 fails, "0% held-out" was probe difficulty, not
   representational collapse, and the pessimistic diagnosis is overstated.

B) SEMANTIC COMPETENCE — no memory, no substrate, no writing. Can the
   backbone answer a paraphrased question about a fact IN ITS CONTEXT?
   This is the ceiling any memory system could ever reach. If the model
   cannot do it with the answer sitting in front of it, the failure was
   never about memory at all.
"""
import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accretion.bakeoff import _norm, forward  # noqa: E402
from accretion.facts import make_facts, restructure  # noqa: E402
from accretion.substrates import build_substrate  # noqa: E402
from model import Config, Protostar  # noqa: E402

# ---- the ladder. {s} = subject; each returns a prompt ending where the object goes.
LADDER = {
    "L0_identical": {
        "the capital of {s} is {o}": "It is widely recorded that the capital of {s} is",
        "{s} was founded by {o}": "It is widely recorded that {s} was founded by",
        "the currency of {s} is called {o}": "It is widely recorded that the currency of {s} is called",
        "{s} is famous for its {o}": "It is widely recorded that {s} is famous for its",
    },
    "L1_light": {  # same content words, different frame
        "the capital of {s} is {o}": "The capital of {s} is",
        "{s} was founded by {o}": "{s} was founded by",
        "the currency of {s} is called {o}": "The currency of {s} is",
        "{s} is famous for its {o}": "{s} is famous for its",
    },
    "L2_moderate": {  # synonyms / restated relation, still no inference hop
        "the capital of {s} is {o}": "The capital city of {s} is named",
        "{s} was founded by {o}": "The founder of {s} was",
        "the currency of {s} is called {o}": "The money used in {s} is called",
        "{s} is famous for its {o}": "{s} is best known for its",
    },
    "L3_hard": {  # the original held-out probes: require an inference step
        "the capital of {s} is {o}": "Travellers to {s} usually land first in",
        "{s} was founded by {o}": "The founding of {s} is credited to",
        "the currency of {s} is called {o}": "Shopkeepers in {s} take payment in",
        "{s} is famous for its {o}": "Visitors mainly come to {s} to see the",
    },
}


def build_gen(model, sub, enc, device):
    def gen(prompt, n=6):
        ids = torch.tensor([enc.encode(prompt)], device=device)
        with torch.no_grad():
            for _ in range(n):
                logits, _ = forward(model, sub, ids[:, -model.cfg.seq_len:])
                ids = torch.cat([ids, logits[0, -1, :50257].argmax().view(1, 1)], 1)
        return enc.decode(ids[0].tolist()[-n:])
    return gen


def score(gen, facts, level):
    tmpls = LADDER[level]
    hits = sum(_norm(f["object"]) in _norm(gen(tmpls[f["template"]].format(s=f["subject"])))
               for f in facts)
    return hits / len(facts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--facts", type=int, default=100)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-write", action="store_true",
                    help="run only test B (semantic competence, no substrate)")
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
    facts = make_facts(a.facts, seed=0)

    # ---------- B: semantic competence, memory-free (the ceiling) ----------
    print("=== B: SEMANTIC COMPETENCE (fact in context, no memory) ===", flush=True)
    print("    Can the backbone answer a REPHRASED question when the answer", flush=True)
    print("    is literally in its context? This bounds any memory system.", flush=True)
    gen_plain = build_gen(model, None, enc, a.device)
    for level in LADDER:
        tmpls = LADDER[level]
        hits = 0
        for f in facts:
            ctx = "\n".join(restructure(f, 7)) + "\n"
            out = gen_plain(ctx + tmpls[f["template"]].format(s=f["subject"]))
            hits += _norm(f["object"]) in _norm(out)
        print(f"  {level:14s} in-context recall: {hits/len(facts):6.1%}", flush=True)

    if a.skip_write:
        return

    # ---------- A: probe ladder against a written substrate ----------
    print("\n=== A: PROBE LADDER (facts written to substrate, no context) ===", flush=True)
    sub = build_substrate("engram", d=cfg.d, n_layer=cfg.n_layer).to(a.device)
    gen_sub = build_gen(model, sub, enc, a.device)
    print("  before writing:", {lvl: f"{score(gen_sub, facts, lvl):.1%}" for lvl in LADDER},
          flush=True)

    docs = [torch.tensor([enc.encode("\n".join(restructure(f, 7)))], device=a.device)
            for f in facts]
    opt = torch.optim.AdamW(sub.write_params(), lr=a.lr)
    sub.train()
    model.train()  # fla GDN chunk kernel needs train mode for backward
    for step in range(a.steps):
        doc = docs[step % len(docs)]
        _, loss = forward(model, sub, doc[:, :-1], doc[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    sub.eval()
    model.eval()

    print("  after writing:", flush=True)
    for level in LADDER:
        print(f"    {level:14s} {score(gen_sub, facts, level):6.1%}", flush=True)


if __name__ == "__main__":
    main()
