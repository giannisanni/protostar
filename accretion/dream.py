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


def _first_word(ans):
    w = ans.split()
    return w[0].strip(",;:!?\"'") if w else ""


def dreams(gen, facts, restructure_fn, n=7, consensus=False, min_agree=0,
           ablate_gen=None, history=None, skip_flips=True):
    """One dream per cue. gen(prompt) -> continuation. Returns
    ([docs], stats); docs are cue + dreamed answer, one per fact.
    consensus: the fact's n dreams vote (first word of each answer) and every
    cue is completed with the winner alone, so one odd dream is outvoted
    and invented details ("in 1855") are dropped. Still no ground truth.
    min_agree: with consensus, a fact whose winner got fewer than this many
    of the n votes is not dreamed at all (the model is unsure; rehearsing a
    guess teaches the guess). Uses only the votes, never the truth.
    ablate_gen: the same cues answered with the lifetime tables off; if that
    winner equals the dreamed one the tables add nothing (the fact was never
    stored, the answer is the backbone's guess) and the fact is skipped.
    history: subject -> last voted winner, updated in place; a fact whose
    winner changed since the last night is unstable and skipped (day dreams
    pass skip_flips=False: new information may overwrite, it only records)."""
    from collections import Counter
    docs, right, total, dropped, flipped, unstored = [], 0, 0, 0, 0, 0
    for f in facts:
        cs = cues(f, restructure_fn, n)
        answers = [_cut(gen(f, c)) for c in cs]
        if consensus:
            votes = Counter(w for w in map(_first_word, answers) if w)
            win, cnt = votes.most_common(1)[0] if votes else ("", 0)
            key = (f["subject"], f.get("template"))  # the cue, never the answer
            prev = history.get(key) if history is not None else None
            if history is not None:
                history[key] = win
            if cnt < min_agree:
                dropped += 1
                continue
            if skip_flips and prev is not None and prev != win:
                flipped += 1
                continue
            if ablate_gen is not None:
                av = Counter(w for w in (_first_word(_cut(ablate_gen(f, c))) for c in cs) if w)
                if av and av.most_common(1)[0][0] == win:
                    unstored += 1
                    continue
            answers = [win] * len(cs)
        lines = []
        for c, ans in zip(cs, answers):
            lines.append(f"{c} {ans}." if ans else c)
            right += _norm(f["object"]) in _norm(ans)  # scoring only; never fed back
            total += 1
        docs.append("\n".join(lines))
    acc = right / max(total, 1)
    st = {"dream_acc": round(acc, 4), "false": round(1 - acc, 4), "n": total}
    if min_agree:
        st["dropped"] = dropped
    if history is not None:
        st["flipped"] = flipped
    if ablate_gen is not None:
        st["unstored"] = unstored
    return docs, st


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


_FILLER = ["The weather held steady through the afternoon.",
           "Traffic on the main road was light that day.",
           "A few people waited at the station for the late train.",
           "The market opened early and closed without incident.",
           "Rain was expected later in the week."]


@torch.no_grad()
def live_day_stream(model, sub, facts, enc, device, window=512, seed=0, per_window=None):
    """The literal day: every fact's doc is packed, whole, into consecutive
    `window`-token windows between filler sentences and streamed through
    the model with the fast memory carried (XL cache). The memory state
    right after a fact's window is that fact's day memory, so its dream is
    a cue in the NEXT window reading the cache: no fact text in context.
    per_window caps facts per window (the rest is filler); None = pack.
    Returns (mems, n_windows)."""
    import random
    rng = random.Random(seed)
    order = list(facts)
    rng.shuffle(order)
    windows, cur, members = [], [], []
    for f in order:
        doc = enc.encode(" " + f["doc"])
        while len(cur) + len(doc) > window or (per_window and len(members) >= per_window):  # close this window with filler
            pad = enc.encode(" " + rng.choice(_FILLER))
            if len(cur) + len(pad) > window:
                windows.append((cur + enc.encode(" .") * (window - len(cur)), members))
                cur, members = [], []
                break
            cur += pad
        cur += doc
        members.append(f["subject"])
        for _ in range(rng.randint(0, 3)):  # filler between facts
            pad = enc.encode(" " + rng.choice(_FILLER))
            if len(cur) + len(pad) <= window:
                cur += pad
    if cur:
        windows.append((cur, members))
    mems = {}
    model.persist_memory = True
    model.fast.reset(1, device)
    for toks, members in windows:
        forward(model, sub, torch.tensor([toks[:window]], device=device))
        snap = model.fast.snapshot()
        for s in members:
            mems[s] = snap
    model.persist_memory = False
    model.fast.reset(1, device)
    return mems, len(windows)
