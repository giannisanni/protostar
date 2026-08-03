"""Phase 1 gate: can a sparse substrate store facts into a FROZEN backbone?

Two numbers decide it:
  recall@1  — does the probe return the right object
  dval      — did general ability degrade (must stay tiny)

A substrate that memorizes by wrecking the backbone is not a memory.
Pre-registered thresholds (do not move after seeing results):
  PASS   any substrate >=60% recall with dval <= +0.02
  TUNE   best in 20-60% -> one round of capacity/lr/steps tuning
  STOP   all <20% or dval > +0.05 -> write up the negative result
"""
import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accretion.facts import (heldout_probe, make_facts, probe_prompt,  # noqa: E402
                             restructure)
from accretion.substrates import SUBSTRATES, build_substrate  # noqa: E402
from data.loader import Shards  # noqa: E402
from model import Config, Protostar  # noqa: E402


def _norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def recall_at_1(generate_fn, facts, probe=probe_prompt):
    hits = sum(_norm(fact["object"]) in _norm(generate_fn(probe(fact)))
               for fact in facts)
    return hits / len(facts)


def forward(model, sub, idx, targets=None):
    """Backbone forward with the substrate injected between layers.

    Mirrors Protostar.forward INCLUDING the v2 fast-memory branch. Getting
    this wrong is not cosmetic: the first Test 2 run silently evaluated a
    memory-native checkpoint with its memory amputated (gate read 0.000,
    val 3.90 instead of 3.32) because this function predates v2 and only
    walked embed -> blocks -> norm_f.
    """
    h = model.embed(idx)
    for i, blk in enumerate(model.blocks):
        h = blk(h, model.cos, model.sin)
        if getattr(model.cfg, "fast_memory", False) and i == model.cfg.fast_layer:
            # respect persist_memory — a stateful caller (oneshot writes) needs
            # the state to survive. Second bug caused by this duplicated copy
            # of Protostar.forward; the regression test compares them.
            if not getattr(model, "persist_memory", False):
                model.fast.reset(h.size(0), h.device, h.dtype)
            retrieved, _ = model.fast.update(h)
            g = torch.sigmoid(model.fast_gate(h))
            model.last_gate = float(g.mean().detach())
            h = h + g * model.fast_proj(retrieved)
        if getattr(model.cfg, "engram", False) and i == model.cfg.engram_layer:
            prev = torch.nn.functional.pad(idx[:, :-1], (1, 0))
            mem = torch.cat([model.tables[0]((prev * 1000003 + idx) % (1 << 18)),
                             model.tables[1]((prev * 31 + idx * 999983) % (1 << 18))], -1)
            h = h + torch.sigmoid(model.gate(h)) * model.mem_proj(mem)
        if sub is not None:
            h = sub(h, idx, layer=i)
    h = model.norm_f(h)
    if targets is None:
        return model.lm_head(h), None
    return None, model._ce(h.view(-1, model.cfg.d), targets.reshape(-1))


@torch.no_grad()
def val_loss(model, sub, loader, device, n=12):
    # loader.batches() advances loader.pos — reset so before/after comparisons
    # always score the SAME batches (caught by an impossible dval in smoke test)
    loader.pos = 0
    tot = 0.0
    for batch, _ in zip(loader.batches(), range(n)):
        batch = batch.to(device)
        _, loss = forward(model, sub, batch[:, :-1], batch[:, 1:])
        tot += loss.item()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--kind", choices=SUBSTRATES, required=True)
    ap.add_argument("--facts", type=int, default=200)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--restructure", type=int, default=0,
                    help="SEAL: N restructured variants per fact (0=raw doc)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    model = Protostar(Config(**ck["cfg"])).to(a.device).eval()
    model.load_state_dict(ck["model"])
    del ck
    for p in model.parameters():  # FROZEN. only the substrate learns.
        p.requires_grad_(False)

    sub = build_substrate(a.kind, d=model.cfg.d, n_layer=model.cfg.n_layer).to(a.device)
    facts = make_facts(a.facts, seed=0)
    val = Shards(a.data, "val", 4, model.cfg.seq_len)

    def gen(prompt, n=6):
        ids = torch.tensor([enc.encode(prompt)], device=a.device)
        with torch.no_grad():
            for _ in range(n):
                logits, _ = forward(model, sub, ids[:, -model.cfg.seq_len:])
                nxt = logits[0, -1, :50257].argmax()
                ids = torch.cat([ids, nxt.view(1, 1)], 1)
        return enc.decode(ids[0].tolist()[-n:])

    base_recall = recall_at_1(gen, facts)
    base_val = val_loss(model, None, val, a.device)

    opt = torch.optim.AdamW(sub.write_params(), lr=a.lr)
    if a.restructure:
        # SEAL: consolidate restructured VARIANTS, not one fixed surface form.
        # Variants are joined into ONE episode per fact: individually they are
        # ~12 tokens, below fla's GDN chunk-kernel threshold (falls back to the
        # recurrent kernel, which has no backward). Also matches SEAL, which
        # finetunes on the SET of implications rather than one at a time.
        docs = [torch.tensor([enc.encode("\n".join(restructure(f, a.restructure)))],
                             device=a.device) for f in facts]
    else:
        docs = [torch.tensor([enc.encode(f["doc"])], device=a.device) for f in facts]
    print(f"training docs: {len(docs)} ({'restructured' if a.restructure else 'raw'})",
          flush=True)
    sub.train()
    # backbone stays frozen (requires_grad=False) but must be in train() MODE:
    # fla's GDN picks the fused_recurrent kernel in eval mode, which has no
    # backward — and substrate grads must flow THROUGH the frozen GDN layers.
    model.train()
    for step in range(a.steps):
        doc = docs[step % len(docs)]
        _, loss = forward(model, sub, doc[:, :-1], doc[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    sub.eval()
    model.eval()

    after_recall = recall_at_1(gen, facts)
    heldout = recall_at_1(gen, facts, probe=heldout_probe)
    after_val = val_loss(model, sub, val, a.device)
    # M1 criterion 3: a model that learned to keep the gate shut is a silent
    # failure — it trains fine and ignores memory entirely.
    gate = f"  gate {model.last_gate:.3f}" if getattr(model.cfg, "fast_memory", False) else ""
    print(f"RESULT {a.kind:8s} slots={sub.n_slots():>9,}  "
          f"recall {base_recall:.1%} -> {after_recall:.1%}  heldout {heldout:.1%}{gate}   "
          f"val {base_val:.4f} -> {after_val:.4f} (d{after_val - base_val:+.4f})",
          flush=True)


if __name__ == "__main__":
    main()
