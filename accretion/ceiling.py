"""In-context ceiling for currprobe's cross-window probes (standing rule:
calibrate a probe against the backbone BEFORE trusting a threshold).

For each age-1 probe, put the fact's statement back IN the context and
score first-token accuracy on the object, memory reset (attention only):
  NEAR  statement immediately followed by the probe prefix
  FAR   statement + the real preceding tokens, packed into one 512 context
  D<n>  statement + the last n real tokens before the probe (distance curve)
  NONE  same FAR context with the statement removed (= the guessing floor)

    python accretion/ceiling.py --ckpt <ckpt> [--device cuda:0]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accretion.curriculum import FactStream  # noqa: E402
from accretion.currprobe import find_sub  # noqa: E402
from model import Config, Protostar  # noqa: E402


DIST = (16, 32, 64, 128, 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows", type=int, default=150)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    model = Protostar(Config(**ck["cfg"])).to(a.device).eval()
    model.load_state_dict(ck["model"])
    del ck
    model.persist_memory = True

    fs = FactStream(enc, 512, seed=a.seed)
    w = 513
    windows = [fs.next_window() for _ in range(a.windows)]
    stream = [t for toks, _ in windows for t in toks]
    last, probes = {}, []
    for kind, fid, start, end in fs.log:  # same age-1 selection as locate.py
        prev, last[fid] = last.get(fid), (start, end)
        if kind != "probe" or end >= a.windows * w or prev is None:
            continue
        wi = start // w
        if prev[0] // w != wi - 1 or (prev[1] - 1) // w != wi - 1 or windows[wi][1]:
            continue
        obj = enc.encode(" " + fs.facts[fid]["object"])
        p = find_sub(windows[wi][0], obj, start - wi * w)
        if p > 0:
            probes.append((wi * w + p, obj[0], prev))

    hits = {"NEAR": 0, **{f"D{d}": 0 for d in DIST}, "FAR": 0, "NONE": 0}
    with torch.no_grad():
        for pos, tgt, (s0, s1) in probes:
            stmt = stream[s0:s1]
            q0 = pos - 1
            while q0 > s1 and stream[q0 - 1] != enc.encode(".")[0]:
                q0 -= 1  # probe prefix = tokens since the last sentence end
            between = stream[s1:q0]
            prefix = stream[q0:pos]
            keep = 512 - len(stmt) - len(prefix)
            ctx = {"NEAR": stmt + prefix,
                   **{f"D{d}": stmt + between[-d:] + prefix for d in DIST},
                   "FAR": stmt + between[-keep:] + prefix,
                   "NONE": between[-(512 - len(prefix)):] + prefix}
            for k, ids in ctx.items():
                if hasattr(model, "fast"):
                    model.fast.reset(1, a.device)
                logits, _ = model(torch.tensor([ids], device=a.device))
                hits[k] += int(logits[0, -1].argmax() == tgt)
    n = len(probes)
    print(f"{n} age-1 probes, first-token accuracy, memory reset:")
    for k, v in hits.items():
        print(f"  {k:5s} {v / n:6.1%}")


if __name__ == "__main__":
    main()
