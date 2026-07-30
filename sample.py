"""Generate text from a protostar checkpoint. Usage:
   python sample.py ckpt.pt "prompt" [--n 120] [--temp 0.8] [--topk 50]
"""
import argparse

import tiktoken
import torch
import torch.nn.functional as F

from model import Config, Protostar

GPT2_VOCAB = 50257  # real vocab; config pads to 50304, never sample the padding

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("prompt")
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--temp", type=float, default=0.8)
ap.add_argument("--topk", type=int, default=50)
ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location=a.device, weights_only=True)
model = Protostar(Config(**ck["cfg"])).to(a.device).eval()
model.load_state_dict(ck["model"])
enc = tiktoken.get_encoding("gpt2")

idx = torch.tensor([enc.encode(a.prompt)], device=a.device)
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=a.device.startswith("cuda")):
    for _ in range(a.n):
        logits, _ = model(idx[:, -model.cfg.seq_len :])
        logits = logits[0, -1, :GPT2_VOCAB].float() / a.temp
        v, _ = torch.topk(logits, a.topk)
        logits[logits < v[-1]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        idx = torch.cat([idx, nxt.view(1, 1)], 1)

print(enc.decode(idx[0].tolist()))
