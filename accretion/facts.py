"""Synthetic facts with provably-novel subjects.

Real facts risk being in FineWeb — then we'd measure pretraining, not memory.
Subjects are nonsense bigrams; objects are drawn from a closed set so a probe
has a well-defined right answer.
"""
import random

SYL = ["zor", "vek", "mip", "thal", "quon", "bri", "nax", "wel", "dush", "kip",
       "orn", "vay", "lut", "gan", "sirn", "pel", "murt", "cav", "yex", "drin"]
ROLES = ["the capital of {s} is {o}", "{s} was founded by {o}",
         "the currency of {s} is called {o}", "{s} is famous for its {o}"]
OBJECTS = ["Tarnwick", "Belloran", "Quillfast", "Ashgrove", "Vermillion",
           "Duskrow", "Farrowgate", "Nimbleton", "Copperfen", "Halewind"]


def _name(rng):
    return (rng.choice(SYL) + rng.choice(SYL)).capitalize()


def make_facts(n, seed=0):
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        s = _name(rng)
        if s in seen:
            continue
        seen.add(s)
        o = rng.choice(OBJECTS)
        tmpl = rng.choice(ROLES)
        sent = tmpl.format(s=s, o=o)
        out.append({
            "subject": s, "object": o, "template": tmpl,
            # a short document so there is context around the fact
            "doc": f"A note about {s}. It is widely recorded that {sent}. "
                   f"Scholars of {s} consider this well established.",
        })
    return out


def probe_prompt(fact):
    """Prompt that ends exactly where the object should be generated."""
    return "It is widely recorded that " + \
        fact["template"].format(s=fact["subject"], o="").rstrip()
