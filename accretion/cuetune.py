"""Dream-cue tuning: train the XL cache read on the job sleep actually asks.
Window 1 states k facts (facts.py docs, 1-12 per window, filler between);
window 2 asks each as a dream cue (restructure() cut before the answer, the
first cue at position 0 exactly like dreaming) and the loss is on the answer
tokens only. Gradients flow through the cache into window 1, so keys learn
to be findable. FineWeb LM loss every step guards the backbone.
Held out: every subject of the lifetime corpora (make_facts seeds 100-102).
Objects are re-drawn per sample, so no subject->object map can be memorized.
    python accretion/cuetune.py --ckpt <xl ckpt> --out <dir>
"""
import argparse
import math
import os
import random
import sys
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accretion.dream import _FILLER, cues  # noqa: E402
from accretion.facts import OBJECTS, ROLES, make_facts, restructure  # noqa: E402
from data.loader import Shards  # noqa: E402
from model import Config, Protostar  # noqa: E402

W = 512
HELD_OUT = {f["subject"] for s in (100, 101, 102) for f in make_facts(60, seed=s)}


def train_names():
    return sorted({f["subject"] for f in make_facts(400, seed=0)} - HELD_OUT)


def _fact(s, o, tmpl):
    sent = tmpl.format(s=s, o=o)
    return {"subject": s, "object": o, "template": tmpl,
            "doc": f"A note about {s}. It is widely recorded that {sent}. "
                   f"Scholars of {s} consider this well established."}


def _pad(toks, enc, rng):
    while True:
        p = enc.encode(" " + rng.choice(_FILLER))
        if len(toks) + len(p) > W:
            return toks + enc.encode(" .") * (W - len(toks))
        toks = toks + p


def sample(rng, enc, names):
    """-> (w1, w2, mask2): W tokens each; mask2 marks answer tokens."""
    fs = [_fact(s, rng.choice(OBJECTS), rng.choice(ROLES))
          for s in rng.sample(names, rng.randint(1, 12))]
    w1 = []
    for f in fs:
        w1 += enc.encode(" " + f["doc"])
        for _ in range(rng.randint(0, 2)):
            w1 += enc.encode(" " + rng.choice(_FILLER))
    w1 = _pad(w1[:W], enc, rng)
    rng.shuffle(fs)
    w2, m2 = [], []
    for f in fs:
        q = enc.encode(("\n" if w2 else "") + rng.choice(cues(f, restructure)))
        a = enc.encode(" " + f["object"])
        if len(w2) + len(q) + len(a) + 1 > W:
            break
        w2 += q + a + enc.encode(".")
        m2 += [0] * len(q) + [1] * len(a) + [0]
    n = len(w2)
    w2 = _pad(w2, enc, rng)
    return w1, w2, m2 + [0] * (W - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/mnt/data/protostar/fineweb10B")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    a = ap.parse_args()
    dev = "cuda"
    enc, rng = tiktoken.get_encoding("gpt2"), random.Random(0)
    names = train_names()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    m = Protostar(Config(**ck["cfg"])).to(dev).train()
    m.load_state_dict(ck["model"])
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    lm = Shards(a.data, "train", a.batch, W).batches()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.csv"), "w")
    log.write("step,cue_loss,cue_acc,lm_loss\n")
    for step in range(a.steps):
        for g in opt.param_groups:  # warmup 50, cosine to 10%
            g["lr"] = a.lr * min(1, (step + 1) / 50) * (0.55 + 0.45 * math.cos(math.pi * step / a.steps))
        w1, w2, mk = map(lambda c: torch.tensor(c, device=dev),
                         zip(*[sample(rng, enc, names) for _ in range(a.batch)]))
        mk = mk[:, 1:].bool()
        with torch.autocast("cuda", torch.bfloat16):
            m.persist_memory = True
            m.fast.reset(a.batch, dev)
            m(w1)  # writes the cache; its graph stays attached
            logits, _ = m(w2)
            lg, tg = logits[:, :-1][mk].float(), w2[:, 1:][mk]
            cue = F.cross_entropy(lg, tg)
            m.persist_memory = False
            b = next(lm).to(dev)
            _, lml = m(b[:, :-1], targets=b[:, 1:])
            loss = cue + lml
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == a.steps - 1:
            acc = (lg.argmax(-1) == tg).float().mean().item()
            log.write(f"{step},{cue.item():.4f},{acc:.4f},{lml.item():.4f}\n")
            log.flush()
            print(f"step {step} cue {cue.item():.3f} acc {acc:.1%} lm {lml.item():.3f}", flush=True)
    m.fast.reset(1, dev)
    torch.save({"model": m.state_dict(), "cfg": ck["cfg"]}, os.path.join(a.out, "ckpt_cue.pt"))


if __name__ == "__main__":
    main()
