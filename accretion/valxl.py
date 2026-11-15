"""FineWeb val three ways: one 2048-token pass (the training guard), 512-token
windows with memory reset, 512-token windows with memory carried.
    python accretion/valxl.py --ckpt <ckpt>"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.loader import Shards  # noqa: E402
from model import Config, Protostar  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", default="/mnt/data/protostar/fineweb10B")
ap.add_argument("--batches", type=int, default=16)
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
m = Protostar(Config(**ck["cfg"])).cuda().eval()
m.load_state_dict(ck["model"])
tot = [0.0, 0.0, 0.0]
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    for b, _ in zip(Shards(a.data, "val", 4, 2048).batches(), range(a.batches)):
        b = b.cuda()
        m.persist_memory = False
        tot[0] += m(b[:, :-1], targets=b[:, 1:])[1].item()
        for j, carry in ((1, False), (2, True)):
            m.persist_memory = carry
            m.fast.reset(b.size(0), "cuda")
            for i in range(4):
                x = b[:, i * 512:(i + 1) * 512 + 1]
                tot[j] += m(x[:, :-1], targets=x[:, 1:])[1].item() / 4
n = a.batches
print(f"2048 one pass {tot[0]/n:.4f} | 512 windows reset {tot[1]/n:.4f} | 512 windows carried {tot[2]/n:.4f}")
