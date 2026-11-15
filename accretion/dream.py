"""Dreaming: sleep trains on what the model GENERATES from its own memory,
never on the ground truth.

Day    each fact is experienced once, as its raw doc. It is held either in
       the context window (episodic buffer) or in the fast memory, written
       by one forward pass (accretion/oneshot.py), once that tier works.
Night  the model is cued with the fact's SUBJECT only (each restructure
       variant cut just before the answer) and completes the cue from its
       memory. Those completions are the consolidation corpus.
Replay older facts are re-dreamed from the current weights with the fast
       memory empty, so rehearsal needs only cues, not stored answers.

A dream that names the wrong object is consolidated anyway: that is a false
memory, and it is counted (`false`), not hidden.
"""
import torch

from accretion.bakeoff import _norm, forward

_BLANK = "\x00"


def cues(fact, restructure_fn, n=7):
    """Restructured variants cut just before the answer. The object is
    replaced by a sentinel BEFORE templating, so it cannot leak into a cue."""
    out = []
    for v in restructure_fn({**fact, "object": _BLANK}, n):
        if _BLANK in v:
            out.append(v.split(_BLANK)[0].rstrip())
    return out


def _cut(text):
    for stop in (".", "\n"):
        text = text.split(stop)[0]
    return text.strip()


def dreams(gen, facts, restructure_fn, n=7):
    """One dream per cue. gen(prompt) -> continuation. Returns
    ([docs], stats); docs are cue + dreamed answer, one per fact."""
    docs, right, total = [], 0, 0
    for f in facts:
        lines = []
        for c in cues(f, restructure_fn, n):
            ans = _cut(gen(f, c))
            lines.append(f"{c} {ans}." if ans else c)
            right += _norm(f["object"]) in _norm(ans)  # scoring only; never fed back
            total += 1
        docs.append("\n".join(lines))
    acc = right / max(total, 1)
    return docs, {"dream_acc": round(acc, 4), "false": round(1 - acc, 4), "n": total}


def make_gen(model, sub, enc, device, mems=None, ctx=None, n_tok=6):
    """Greedy continuation through backbone + substrate. With `mems`
    (subject -> fast-memory snapshot from live_day) the fact's day memory is
    restored for each cue, so dreaming never writes into what it reads;
    With `ctx` (subject -> the day's raw text) the day is held in the
    context window instead: an episodic buffer read by attention, until the
    fast tier carries a day on its own. Neither: dreams come from the
    weights alone (replay).
    ponytail: ctx is the stand-in for a working fast tier; swap in `mems`
    once a cross-sequence memory passes currprobe."""

    @torch.no_grad()
    def gen(fact, prompt):
        snap = (mems or {}).get(fact["subject"])
        model.persist_memory = snap is not None
        pre = (ctx or {}).get(fact["subject"])
        ids = torch.tensor([enc.encode((pre + "\n" if pre else "") + prompt)], device=device)
        for _ in range(n_tok):
            if snap is not None:  # every forward writes; restore the day each time
                model.fast.load(snap)
            logits, _ = forward(model, sub, ids[:, -model.cfg.seq_len:])
            ids = torch.cat([ids, logits[0, -1, :50257].argmax().view(1, 1)], 1)
        model.persist_memory = False
        return enc.decode(ids[0, -n_tok:].tolist())

    return gen


@torch.no_grad()
def live_day(model, sub, facts, enc, device):
    """The day: each fact's raw doc goes through the model ONCE with the fast
    memory on; the resulting state is that episode's memory.
    ponytail: one episode per fact; a whole day in one state is the harder,
    later test (needs the stream-capable fast tier)."""
    mems = {}
    for f in facts:
        model.persist_memory = True
        model.fast.reset(1, device)
        forward(model, sub, torch.tensor([enc.encode(f["doc"])], device=device))
        mems[f["subject"]] = model.fast.snapshot()
    model.persist_memory = False
    model.fast.reset(1, device)
    return mems
