"""Phase-1 diagnostics: distinguish 'memory is impossible' from 'harness is broken'.

D1  ICL ceiling: probe WITH the doc in context. If the frozen backbone cannot
    complete the object even when the answer is right there, the probe format
    is unanswerable for this model and recall@1 was never a valid metric.
D2  Single-fact overfit: one fact, many steps, generous LR, loss logged.
    If train loss on the doc collapses but the probe still fails, storage
    works and RETRIEVAL is the gap. If loss will not fall, the gradient
    path through the frozen backbone is the gap.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accretion.bakeoff import forward, recall_at_1, _norm  # noqa: E402
from accretion.facts import make_facts, probe_prompt  # noqa: E402
from accretion.substrates import build_substrate  # noqa: E402
from model import Config, Protostar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", default="engram")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    model = Protostar(Config(**ck["cfg"])).to(a.device).eval()
    model.load_state_dict(ck["model"])
    del ck
    for p in model.parameters():
        p.requires_grad_(False)

    def gen(prompt, sub=None, n=6):
        ids = torch.tensor([enc.encode(prompt)], device=a.device)
        with torch.no_grad():
            for _ in range(n):
                logits, _ = forward(model, sub, ids[:, -model.cfg.seq_len:])
                nxt = logits[0, -1, :50257].argmax()
                ids = torch.cat([ids, nxt.view(1, 1)], 1)
        return enc.decode(ids[0].tolist()[-n:])

    facts = make_facts(20, seed=0)

    print("=== D1: ICL ceiling (doc in context, no substrate) ===", flush=True)
    hits = 0
    for f in facts:
        out = gen(f["doc"] + "\n" + probe_prompt(f))
        ok = _norm(f["object"]) in _norm(out)
        hits += ok
        if f is facts[0] or f is facts[1]:
            print(f"  probe: ...{probe_prompt(f)[-46:]!r}")
            print(f"  gold: {f['object']:12s} got: {out!r}  {'HIT' if ok else 'miss'}")
    print(f"D1 ICL recall: {hits}/{len(facts)} = {hits/len(facts):.0%}", flush=True)

    print(f"=== D2: single-fact overfit ({a.kind}, {a.steps} steps, lr {a.lr}) ===", flush=True)
    fact = facts[0]
    sub = build_substrate(a.kind, d=model.cfg.d, n_layer=model.cfg.n_layer).to(a.device)
    doc = torch.tensor([enc.encode(fact["doc"])], device=a.device)
    opt = torch.optim.AdamW(sub.write_params(), lr=a.lr)
    sub.train()
    model.train()  # fla chunk kernel needs train mode for backward
    for step in range(a.steps):
        _, loss = forward(model, sub, doc[:, :-1], doc[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0 or step == a.steps - 1:
            print(f"  step {step:4d}  doc loss {loss.item():.4f}", flush=True)
    sub.eval()
    model.eval()
    out = gen(probe_prompt(fact), sub=sub, n=6)
    print(f"D2 probe: gold={fact['object']!r} got={out!r} "
          f"{'HIT' if _norm(fact['object']) in _norm(out) else 'MISS'}", flush=True)
    # also: does the substrate at least reduce loss on the doc itself?
    with torch.no_grad():
        model.train()
        _, l_with = forward(model, sub, doc[:, :-1], doc[:, 1:])
        _, l_without = forward(model, None, doc[:, :-1], doc[:, 1:])
        model.eval()
    print(f"D2 doc loss: without sub {l_without.item():.4f} -> with sub {l_with.item():.4f}",
          flush=True)


if __name__ == "__main__":
    main()
