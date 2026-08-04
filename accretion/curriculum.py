"""Retrieval curriculum: synthetic fact streams that REQUIRE cross-window memory.

The stateful phase taught the model to TOLERATE carried state (drift +0.42 ->
+0.003) but not to READ it (cross-sequence recall 0%), because plain FineWeb
almost never needs a specific fact from 2000 tokens ago — denoising is a dense
reward, retrieval a sparse tail. This module makes retrieval dense: a fraction
of training lanes carry streams where a fact is stated once and then probed
1-20 windows LATER. Attention only sees the current window, so the probe's
answer is reachable through the carried fast-memory state or not at all.

Anti-leak invariant (tested): a probe never lands in the same 513-token window
as ANY earlier mention of its fact — eligibility requires > window+1 tokens
since the fact's last mention, measured mention-END to probe-START.

Vocabulary is deliberately DISJOINT from accretion/facts.py (different
syllables, relations, and object style), so the standard oneshot eval on
facts.py templates measures TRANSFER, not memorization of the training set.
The curriculum-format eval (make_eval_facts, unseen seed) measures MECHANISM.
"""
import random

# facts.py uses: zor vek mip thal quon bri nax wel dush kip orn vay lut gan
# sirn pel murt cav yex drin — none of these appear here.
SYL = ["bla", "fro", "gim", "hax", "jul", "kre", "lom", "ned", "pol", "ras",
       "sut", "tev", "ulm", "vor", "wib", "yat", "zem", "chu", "dai", "eck"]

# facts.py relations: capital-of, founded-by, currency-of, famous-for.
RELS = [
    ("the mayor of {s} is {o}", "Question: who is the mayor of {s}? Answer: {o}"),
    ("{s} exports mainly {o}", "Question: what does {s} export? Answer: {o}"),
    ("the river through {s} is the {o}", "Question: which river runs through {s}? Answer: {o}"),
    ("{s} celebrates a festival called {o}", "Question: what festival does {s} celebrate? Answer: {o}"),
]

_FILLER_A = ["quiet", "long", "bright", "cold", "busy", "slow"]
_FILLER_N = ["morning", "harvest", "road", "market", "winter", "council"]
_FILLER_T = ["The {a} {n} passed without incident.",
             "People spoke of the {a} {n} for days.",
             "Nothing changed during the {a} {n}."]


def _name(rng):
    return (rng.choice(SYL) + rng.choice(SYL)).capitalize()


def _fact(rng, used):
    # only SUBJECTS must be unique (probes address facts by subject); objects
    # may repeat. Demanding both unique exhausted the 400-name space inside
    # long episodes and spun this loop forever — caught as a hung test run.
    while True:
        s = _name(rng)
        if s not in used:
            used.add(s)
            while True:
                o = _name(rng)
                if o != s:
                    return {"subject": s, "object": o, "rel": rng.randrange(len(RELS))}


def statement(f):
    return "It is known that " + RELS[f["rel"]][0].format(s=f["subject"], o=f["object"]) + "."


def qa(f):
    return RELS[f["rel"]][1].format(s=f["subject"], o=f["object"])


def eval_probe(f):
    """Prompt ending exactly at 'Answer:' — the trained probe format."""
    return RELS[f["rel"]][1].format(s=f["subject"], o="").rstrip()


def eval_statement(f):
    """What a one-shot write feeds: both surface forms seen in training."""
    return [statement(f), qa(f)]


def make_eval_facts(n, seed=999):
    rng = random.Random(seed)
    used = set()
    return [_fact(rng, used) for _ in range(n)]


class FactStream:
    """Endless token stream for ONE training lane.

    Episodes of 8-24 windows; on episode end the lane's memory state is reset
    (signalled via next_window's second return) and the fact pool cleared, so
    the model is never graded on facts its memory cannot contain.
    """

    def __init__(self, enc, window, seed):
        self.enc, self.w = enc, window + 1  # loader windows are seq_len+1 tokens
        self.rng = random.Random(seed)
        self.buf, self.pos, self._n = [], 0, 0
        self.log = []  # (kind, fact_id, start_tok, end_tok) — for tests
        self.facts = {}  # fact_id -> fact dict, survives episode clears (eval)
        self._episode()

    def _episode(self):
        self.ep_end = self.pos + self.rng.randint(8, 24) * self.w
        self.pool, self.used = [], set()

    def _emit(self):
        cur = self.pos + len(self.buf)
        ready = [f for f in self.pool
                 if cur - f["end"] > self.w and f["probes"] > 0]
        r = self.rng.random()
        if ready and r < 0.35:
            f = self.rng.choice(ready)
            f["probes"] -= 1
            text = qa(f) if self.rng.random() < 0.5 else \
                RELS[f["rel"]][0].format(s=f["subject"], o=f["object"]).capitalize() + "."
            toks = self.enc.encode(" " + text)
            # the probe text CONTAINS the answer (teacher forcing) — it counts
            # as a fresh mention, so later probes must keep their distance
            f["end"] = cur + len(toks)
            self.log.append(("probe", f["id"], cur, f["end"]))
            return toks
        if r < 0.5 and len(self.used) < 300:  # hard cap: never exhaust the namespace
            f = _fact(self.rng, self.used)
            toks = self.enc.encode(" " + statement(f))
            f.update(end=cur + len(toks), probes=self.rng.randint(2, 4), id=self._n)
            self._n += 1
            self.pool.append(f)
            self.facts[f["id"]] = f
            self.log.append(("state", f["id"], cur, f["end"]))
            return toks
        t = self.rng.choice(_FILLER_T)
        return self.enc.encode(" " + t.format(a=self.rng.choice(_FILLER_A),
                                              n=self.rng.choice(_FILLER_N)))

    def next_window(self):
        reset = False
        if self.pos >= self.ep_end:
            self._episode()
            self.buf = []  # drop the old episode's straddling tokens
            reset = True
        while len(self.buf) < self.w:
            self.buf += self._emit()
        out, self.buf = self.buf[: self.w], self.buf[self.w:]
        self.pos += self.w
        return out, reset
