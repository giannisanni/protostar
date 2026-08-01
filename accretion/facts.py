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


# SEAL's central finding: consolidating one fixed surface form barely works
# (+0.8 pts); consolidating RESTRUCTURED variants works (+14). These are the
# restructurings — paraphrase, implication, Q&A, list form — of the same fact.
# ponytail: template-generated, no LLM needed. ARC-TTT also found LM-generated
# augmentation HURT vs rule-based, so procedural is the right call here.
_VARIANTS = [
    "{sent}.",
    "Q: {q}\nA: {o}.",
    "Regarding {s}: {sent}.",
    "Fact sheet for {s}\n- {sent}",
    "Many sources confirm it. {sent}.",
    "If you are asked about {s}, remember that {sent}.",
    "{s} entry. {sent}. End of entry.",
]
_QUESTIONS = {
    "the capital of {s} is {o}": "What is the capital of {s}?",
    "{s} was founded by {o}": "Who founded {s}?",
    "the currency of {s} is called {o}": "What is the currency of {s} called?",
    "{s} is famous for its {o}": "What is {s} famous for?",
}


# Phrasings deliberately ABSENT from every training variant. If recall holds
# here, the substrate stored the FACT; if it collapses, it stored the format.
_HELDOUT = {
    "the capital of {s} is {o}": "Travellers to {s} usually land first in",
    "{s} was founded by {o}": "The founding of {s} is credited to",
    "the currency of {s} is called {o}": "Shopkeepers in {s} take payment in",
    "{s} is famous for its {o}": "Visitors mainly come to {s} to see the",
}


def heldout_probe(fact):
    """Probe with a phrasing never used in training — generalization check."""
    return _HELDOUT[fact["template"]].format(s=fact["subject"])


def restructure(fact, n=None):
    """Return n restructured documents expressing the same fact."""
    s, o, tmpl = fact["subject"], fact["object"], fact["template"]
    sent = tmpl.format(s=s, o=o)
    q = _QUESTIONS[tmpl].format(s=s)
    outs = [v.format(sent=sent, s=s, o=o, q=q) for v in _VARIANTS]
    return outs[: n or len(outs)]
