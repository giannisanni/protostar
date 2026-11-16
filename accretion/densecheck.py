"""Voted dream accuracy from the XL cache vs facts per window (eval subjects,
make_facts seed 100). python accretion/densecheck.py --ckpt <ckpt>"""
import argparse
import sys
from pathlib import Path

import tiktoken
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accretion.dream import dreams, live_day_stream, make_gen  # noqa: E402
from accretion.facts import make_facts, restructure  # noqa: E402
from accretion.substrates import build_substrate  # noqa: E402
from model import Config, Protostar  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
a = ap.parse_args()
enc, dev = tiktoken.get_encoding("gpt2"), "cuda"
ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
cfg = Config(**ck["cfg"])
m = Protostar(cfg).to(dev).eval()
m.load_state_dict(ck["model"])
sub = build_substrate("engram", d=cfg.d, n_layer=cfg.n_layer).to(dev)
facts = make_facts(60, seed=100)
for pw in (1, 3, None):
    mems, nw = live_day_stream(m, sub, facts, enc, dev, per_window=pw)
    r = dreams(make_gen(m, sub, enc, dev, mems=mems), facts, restructure, consensus=True)[1]["dream_acc"]
    print(f"facts/window {str(pw or 'packed'):>6s} windows {nw:3d} voted {r:.1%}", flush=True)
