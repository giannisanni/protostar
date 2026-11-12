"""In-distribution retrieval check: did curriculum training teach ANYTHING?

The one-shot eval feeds bare ~30-token sequences; training only ever saw
facts embedded in full 512-token windows. Before believing a 0% (three
earlier 0%s in this project were harness artifacts), measure the model ON
the training distribution itself: run fresh FactStream windows through the
model and score argmax accuracy on the OBJECT tokens of probe emissions —
once with the state carried (persist), once with it reset every window
(fresh). Every probe sits >1 window after its fact's last mention, so:

    acc_persist >> acc_fresh  -> retrieval was learned; the one-shot eval is
                                 out-of-distribution, not the mechanism dead
    acc_persist ~= acc_fresh  -> the curriculum failed to teach retrieval;
                                 the pre-registered negative stands
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accretion.curriculum import FactStream  # noqa: E402
from model import Config, Protostar  # noqa: E402


def find_sub(hay, needle, lo=0):
    for i in range(lo, len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            return i
    return -1


BUCKETS = ((1, 1), (2, 3), (4, 99))  # probe age in windows since last mention


@torch.no_grad()
def run_pass(model, windows, probe_spans, device, persist):
    model.persist_memory = True  # we control resets manually in both passes
    model.fast.reset(1, device)
    first_ok = first_n = all_ok = all_n = 0
    by_age = {b: [0, 0] for b in BUCKETS}
    for wi, (toks, ep_reset) in enumerate(windows):
        if ep_reset or not persist:
            model.fast.reset(1, device)
        ids = torch.tensor([toks], device=device)
        logits, _ = model(ids[:, :-1])
        pred = logits[0].argmax(-1)  # pred[t] guesses toks[t+1]
        for (w, start, obj_toks, age) in probe_spans:
            if w != wi:
                continue
            first_n += 1
            hit = int(pred[start - 1] == obj_toks[0])
            first_ok += hit
            for b in BUCKETS:
                if b[0] <= age <= b[1]:
                    by_age[b][0] += hit
                    by_age[b][1] += 1
            for k, ot in enumerate(obj_toks):
                all_n += 1
                all_ok += int(pred[start - 1 + k] == ot)
    run_pass.by_age = {b: ok / max(n, 1) for b, (ok, n) in by_age.items()}
    run_pass.age_n = {b: n for b, (_, n) in by_age.items()}
    return first_ok / max(first_n, 1), all_ok / max(all_n, 1), first_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows", type=int, default=150)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--seed", type=int, default=31337)  # unseen by training
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    cfg = Config(**ck["cfg"])
    assert cfg.fast_memory
    model = Protostar(cfg).to(a.device).eval()
    model.load_state_dict(ck["model"])
    del ck

    fs = FactStream(enc, a.window, seed=a.seed)
    w = a.window + 1
    windows, probe_spans = [], []
    for wi in range(a.windows):
        toks, reset = fs.next_window()
        windows.append((toks, reset))
    last_end = {}  # fact id -> end token of its latest mention (statement or probe)
    for kind, fid, start, end in fs.log:
        prev, last_end[fid] = last_end.get(fid), end
        if kind != "probe" or end >= a.windows * w:
            continue
        wi, lo = start // w, (start // w) * w
        if end > lo + w:  # straddles a window boundary — skip, can't score cleanly
            continue
        obj = enc.encode(" " + fs.facts[fid]["object"])
        p = find_sub(windows[wi][0], obj, start - lo)
        if p > 0:
            probe_spans.append((wi, p, obj, wi - (prev - 1) // w))

    print(f"{len(probe_spans)} scoreable cross-window probes in {a.windows} windows",
          flush=True)
    fp, ap_, n = run_pass(model, windows, probe_spans, a.device, persist=True)
    pa = run_pass.by_age
    print(f"PERSIST  first-token {fp:.1%}  all-tokens {ap_:.1%}  (n={n})", flush=True)
    ff, af, _ = run_pass(model, windows, probe_spans, a.device, persist=False)
    fa, na = run_pass.by_age, run_pass.age_n
    print(f"FRESH    first-token {ff:.1%}  all-tokens {af:.1%}", flush=True)
    print(f"MEMORY CONTRIBUTION (first-token): {fp - ff:+.1%}", flush=True)
    for b in BUCKETS:  # the ladder: K-window bptt directly trains only age <= K-1
        print(f"  age {b[0]}-{b[1] if b[1] < 99 else '+'}: persist {pa[b]:.1%}  fresh {fa[b]:.1%}  "
              f"contribution {pa[b] - fa[b]:+.1%}  (n={na[b]})", flush=True)


if __name__ == "__main__":
    main()
