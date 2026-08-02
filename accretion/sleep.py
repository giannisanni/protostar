"""Sleep: consolidate new episodes into persistent memory, behind a gate.

The load-bearing claim of accretion is not "memory can hold a fact" — the
probe ladder already showed that (50% recall at light paraphrase). It is
"memory holds facts PERMANENTLY, across many consolidations, without eroding
what came before."

Three mechanisms, each earning its place from the deep reads:
  restructure   SEAL: consolidating one surface form barely works (+0.8 pts);
                consolidating restructured variants works (+14). Never distil
                raw text.
  gate          Held-out val loss AND a retention suite over every previously
                committed fact. SEAL's own open problem is sequential edits
                degrading old knowledge with no defense; this is the defense.
  rollback      Substrate state is snapshotted before every write. A failing
                gate restores it. The model never certifies its own edit.

The gate stays EXTERNAL, permanently — the thing being modified must not be
the thing that approves the modification.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accretion.bakeoff import _norm, forward  # noqa: E402
from accretion.facts import restructure  # noqa: E402
from accretion.registry import content_key  # noqa: E402


def snapshot(sub):
    return {n: p.detach().clone() for n, p in sub.named_parameters()}


def restore(sub, snap):
    with torch.no_grad():
        for n, p in sub.named_parameters():
            p.copy_(snap[n])


def recall(model, sub, facts, probe_fn, enc, device, n_tok=6):
    """Fraction of `facts` whose object is produced from `probe_fn(fact)`."""
    hits = 0
    for f in facts:
        ids = torch.tensor([enc.encode(probe_fn(f))], device=device)
        with torch.no_grad():
            for _ in range(n_tok):
                logits, _ = forward(model, sub, ids[:, -model.cfg.seq_len:])
                ids = torch.cat([ids, logits[0, -1, :50257].argmax().view(1, 1)], 1)
        out = enc.decode(ids[0].tolist()[-n_tok:])
        hits += _norm(f["object"]) in _norm(out)
    return hits / max(len(facts), 1)


def consolidate(model, sub, new_facts, registry, val_loss_fn, probe_fn, enc,
                device, cycle, steps=20000, lr=2e-3, n_variants=7,
                val_eps=0.05, retention_floor=0.5, gate=True, log=print):
    """One sleep cycle. Returns a dict of evidence; commits or rolls back.

    val_eps          max tolerated rise in held-out val loss
    retention_floor  previously-committed facts must keep this FRACTION of
                     their pre-write recall (relative, not absolute — the
                     lesson from the probe-ladder correction)
    """
    fresh = [f for f in new_facts if not registry.knows(content_key(f))]
    log(f"[cycle {cycle}] {len(fresh)}/{len(new_facts)} facts are new "
        f"(registry: {registry.summary()})")
    if not fresh:
        return {"cycle": cycle, "committed": False, "reason": "nothing new"}

    prior = registry.committed_facts()
    prior_facts = [r["fact"] for r in prior]
    before_val = val_loss_fn(sub)
    before_retention = recall(model, sub, prior_facts, probe_fn, enc, device) if prior_facts else 1.0
    snap = snapshot(sub)

    # --- write: distil RESTRUCTURED variants, never the raw surface form ---
    docs = [torch.tensor([enc.encode("\n".join(restructure(f, n_variants)))], device=device)
            for f in fresh]
    opt = torch.optim.AdamW(sub.write_params(), lr=lr)
    sub.train()
    model.train()  # fla GDN chunk kernel needs train mode for backward
    for step in range(steps):
        doc = docs[step % len(docs)]
        _, loss = forward(model, sub, doc[:, :-1], doc[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    sub.eval()
    model.eval()

    # --- gate: did we damage general ability, or forget anything? ---
    after_val = val_loss_fn(sub)
    after_retention = recall(model, sub, prior_facts, probe_fn, enc, device) if prior_facts else 1.0
    new_recall = recall(model, sub, fresh, probe_fn, enc, device)
    # If nothing was retained beforehand there is nothing to protect, and the
    # ratio would read a vacuous 100%. Report it as None so the evidence row
    # cannot be mistaken for "the gate verified retention".
    kept = (after_retention / before_retention) if before_retention > 0 else None

    val_ok = (after_val - before_val) <= val_eps
    ret_ok = kept is None or kept >= retention_floor
    passed = (val_ok and ret_ok) or not gate

    evidence = {
        "cycle": cycle, "n_facts": len(fresh),
        "val_before": round(before_val, 4), "val_after": round(after_val, 4),
        "dval": round(after_val - before_val, 4), "val_ok": val_ok,
        "retention_before": round(before_retention, 4),
        "retention_after": round(after_retention, 4),
        "retention_kept": (round(kept, 4) if kept is not None else None),
        "retention_ok": ret_ok, "retention_vacuous": kept is None,
        "new_recall": round(new_recall, 4),
        "committed": passed, "gate_enabled": gate,
    }
    keptstr = f"kept {kept:.1%}" if kept is not None else "no prior knowledge to protect"
    log(f"[cycle {cycle}] new_recall={new_recall:.1%}  dval={after_val - before_val:+.4f}"
        f"  prior_retention {before_retention:.1%}->{after_retention:.1%}"
        f" ({keptstr})  {'COMMIT' if passed else 'ROLLBACK'}")

    if not passed:
        restore(sub, snap)

    for f in fresh:
        registry.append({**evidence, "key": content_key(f), "fact": f})
    return evidence
