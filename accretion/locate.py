"""Where does cross-window retrieval break? Three measurements, no training.

For every age-1 probe (fact stated in window N-1, asked in window N):
  ADDRESS  rank of the statement's tokens among all 512 keys of window N-1
           by q_probe . k_t  (chance median ~ 512 / 2 / span_len)
  STORE    rank of the statement's tokens by cos(M q_probe, v_t), M = the
           state carried into window N  (does the read return THIS fact?)
  GRAD     cross-window gradient on the memory's write projections from the
           probe's answer loss, relative to the within-window gradient
           (is the bptt signal there and is it big enough to matter?)

    python accretion/locate.py --ckpt <ckpt> [--device cuda:0]
"""
import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accretion.curriculum import FactStream  # noqa: E402
from accretion.currprobe import find_sub  # noqa: E402
from model import Config, Protostar  # noqa: E402


def mem0(model):
    """The level that holds the most signal: level 0 of a stack, or the memory."""
    return model.fast.levels[0] if hasattr(model.fast, "levels") else model.fast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows", type=int, default=150)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grad_probes", type=int, default=40)
    a = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    cfg = Config(**ck["cfg"])
    model = Protostar(cfg).to(a.device).eval()
    model.load_state_dict(ck["model"])
    del ck
    model.persist_memory = True
    m = mem0(model)

    fs = FactStream(enc, 512, seed=a.seed)
    w = 513
    windows = [fs.next_window() for _ in range(a.windows)]
    last = {}  # fid -> (start, end) of latest mention
    probes = []  # (wi, pos_in_window, obj_toks, stmt_lo, stmt_hi) for age-1 probes
    for kind, fid, start, end in fs.log:
        prev, last[fid] = last.get(fid), (start, end)
        if kind != "probe" or end >= a.windows * w or prev is None:
            continue
        wi, lo = start // w, (start // w) * w
        if end > lo + w or prev[0] // w != wi - 1 or (prev[1] - 1) // w != wi - 1:
            continue  # age-1, statement wholly inside window wi-1
        if windows[wi][1]:
            continue  # episode reset between them: memory legitimately empty
        obj = enc.encode(" " + fs.facts[fid]["object"])
        p = find_sub(windows[wi][0], obj, start - lo)
        if p > 0:
            plo = (wi - 1) * w
            probes.append((wi, p, obj, prev[0] - plo, prev[1] - plo))

    # capture the memory's input h and the carried state, window by window
    hs, states = {}, {}
    orig = m.forward

    def spy(h, _wi=[0]):
        hs[_wi[0]] = h.detach()
        return orig(h)

    m.forward = m.update = spy  # plain FastMemory is called via .update
    model.fast.reset(1, a.device)
    with torch.no_grad():
        for wi, (toks, reset) in enumerate(windows):
            if reset:
                model.fast.reset(1, a.device)
            states[wi] = m._state.detach().clone()  # state carried INTO window wi
            spy.__defaults__[0][0] = wi
            model(torch.tensor([toks[:-1]], device=a.device))
    del m.forward, m.update  # back to the class methods

    addr, store = [], []
    with torch.no_grad():
        for wi, p, obj, slo, shi in probes:
            hp, hw = hs[wi][0], hs[wi - 1][0]
            q = F.normalize(m.wq(hp[p - 1]), dim=-1)
            k = F.normalize(m.wk(hw), dim=-1)
            v = m.wv(hw)
            span = range(slo, min(shi, hw.size(0)))
            sim = k @ q
            addr.append(min(int((sim > sim[t]).sum()) for t in span))
            r = states[wi][0].to(q.dtype) @ q
            cs = F.cosine_similarity(v, r.unsqueeze(0), dim=-1)
            store.append(min(int((cs > cs[t]).sum()) for t in span))
    span_len = statistics.median(shi - slo for *_, slo, shi in probes)
    chance = 512 / (span_len + 1)
    print(f"{len(probes)} age-1 probes; median statement span {span_len:.0f} tok; "
          f"chance best-rank ~{chance:.0f}")
    print(f"ADDRESS  median best rank {statistics.median(addr):.0f}  "
          f"top-5 {sum(x < 5 for x in addr) / len(addr):.0%}")
    print(f"STORE    median best rank {statistics.median(store):.0f}  "
          f"top-5 {sum(x < 5 for x in store) / len(store):.0%}")

    # GRAD: two windows with the graph kept, loss = probe answer tokens only
    model.train()
    for lvl in getattr(model.fast, "levels", [model.fast]):
        lvl.bptt = True
    ratios = []
    for wi, p, obj, *_ in probes[: a.grad_probes]:
        model.zero_grad(set_to_none=True)
        model.fast.reset(1, a.device)
        x1 = torch.tensor([windows[wi - 1][0][:-1]], device=a.device)
        x2 = torch.tensor([windows[wi][0][:-1]], device=a.device)
        with torch.autocast("cuda", torch.bfloat16):
            model(x1)
            logits, _ = model(x2)
        tgt = torch.tensor(obj, device=a.device)
        loss = F.cross_entropy(logits[0, p - 1:p - 1 + len(obj)].float(), tgt)
        h1 = m.wk.weight
        g_total = torch.autograd.grad(loss, h1, retain_graph=True)[0]
        # within-window share: same loss with window 1's write detached
        model.fast.reset(1, a.device)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            model(x1)
        model.fast.detach_()
        with torch.autocast("cuda", torch.bfloat16):
            logits, _ = model(x2)
        loss2 = F.cross_entropy(logits[0, p - 1:p - 1 + len(obj)].float(), tgt)
        g_within = torch.autograd.grad(loss2, h1)[0]
        g_cross = g_total - g_within
        ratios.append(float(g_cross.norm() / (g_within.norm() + 1e-12)))
    print(f"GRAD     cross/within wk-gradient norm: median {statistics.median(ratios):.3f}  "
          f"(n={len(ratios)})")


if __name__ == "__main__":
    main()
