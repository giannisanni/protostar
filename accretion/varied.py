"""Heterogeneous fact corpora — a fair test for lexically-keyed memory.

The lifetime benchmark used 4 relation templates and 7 restructuring variants
for ALL corpora, so 68% of corpus A's token bigrams appeared verbatim in
corpus B. That is close to a worst case for n-gram addressing: the shared
scaffolding keys are high-traffic and get rewritten by every consolidation.

Real facts are not like that. "Gianni's birthday is July 9" and "the Cavalier
rate is $4.25/lb" share almost no surface form. This module generates corpora
from disjoint DOMAINS — distinct relations, vocabulary, sentence shapes and
restructuring styles — so the measured interference reflects content overlap
rather than template overlap.

If interference drops sharply here, "lexical keys cannot support sequential
learning" is too strong; the honest claim becomes "cannot when phrasing is
homogeneous".
"""
import random

# Each domain is self-contained: its own syllables, objects, relations and
# restructuring styles. No string is shared across domains by design.
DOMAINS = [
    {
        "name": "places",
        "syl": ["zor", "vek", "mip", "thal", "quon", "bri", "nax", "wel"],
        "objects": ["Tarnwick", "Belloran", "Quillfast", "Ashgrove",
                    "Vermillion", "Duskrow", "Farrowgate", "Nimbleton"],
        "relations": {
            "capital": ("the capital of {s} is {o}", "The capital of {s} is"),
            "currency": ("the currency of {s} is called {o}",
                         "The currency of {s} is"),
            "river": ("the main river of {s} is the {o}", "The main river of {s} is the"),
            "port": ("the chief port of {s} is {o}", "The chief port of {s} is"),
        },
        "variants": [
            "{sent}.",
            "Geographic note: {sent}.",
            "Atlas entry for {s} — {sent}.",
            "Surveyors confirm {sent}.",
            "In the gazetteer, {sent}.",
        ],
    },
    {
        "name": "species",
        "syl": ["dush", "kip", "orn", "vay", "lut", "gan", "sirn", "pel"],
        "objects": ["Cavenhold", "Yexbrook", "Drinmoor", "Halewind",
                    "Copperfen", "Marrowdale", "Silthorn", "Grayreach"],
        "relations": {
            "habitat": ("{s} lives in the {o} wetlands", "{s} lives in the"),
            "diet": ("{s} feeds mainly on {o} moss", "{s} feeds mainly on"),
            "describer": ("{s} was first described by {o}",
                          "{s} was first described by"),
            "migration": ("{s} migrates each winter to {o}",
                          "{s} migrates each winter to"),
        },
        "variants": [
            "Field observation: {sent}.",
            "{sent}, according to the survey.",
            "Species file — {s}: {sent}.",
            "Researchers recorded that {sent}.",
            "Taxonomy note. {sent}.",
        ],
    },
    {
        "name": "artifacts",
        "syl": ["murt", "cav", "yex", "drin", "sten", "wold", "brack", "fen"],
        "objects": ["Ironhollow", "Saltmere", "Windlass", "Ashenford",
                    "Thistlewood", "Emberly", "Kestrelin", "Downhaven"],
        "relations": {
            "maker": ("the {s} device was built by {o} works",
                      "The {s} device was built by"),
            "material": ("the {s} alloy contains {o} ore",
                         "The {s} alloy contains"),
            "protocol": ("the {s} standard replaced {o}",
                         "The {s} standard replaced"),
            "housing": ("the {s} assembly ships in {o} casing",
                        "The {s} assembly ships in"),
        },
        "variants": [
            "Engineering record: {sent}.",
            "Spec sheet says {sent}.",
            "Per the manual, {sent}.",
            "Manufacturing log — {sent}.",
            "Documentation states {sent}.",
        ],
    },
]


def _name(rng, syl):
    return (rng.choice(syl) + rng.choice(syl)).capitalize()


def make_varied_facts(n, domain, seed=0):
    """n facts drawn entirely from one domain's vocabulary and relations."""
    d = DOMAINS[domain % len(DOMAINS)]
    rng = random.Random(seed + 1000 * domain)
    keys = list(d["relations"])
    seen, out = set(), []
    while len(out) < n:
        s = _name(rng, d["syl"])
        if s in seen:
            continue
        seen.add(s)
        rel = rng.choice(keys)
        tmpl, probe = d["relations"][rel]
        o = rng.choice(d["objects"])
        out.append({
            "subject": s, "object": o, "template": tmpl, "probe": probe,
            "domain": d["name"], "variants": d["variants"],
            "doc": tmpl.format(s=s, o=o) + ".",
        })
    return out


def varied_restructure(fact, n=None):
    """Restructure using only THIS fact's domain styles."""
    sent = fact["template"].format(s=fact["subject"], o=fact["object"])
    outs = [v.format(sent=sent, s=fact["subject"], o=fact["object"])
            for v in fact["variants"]]
    return outs[: n or len(outs)]


def varied_probe(fact):
    """L1-equivalent: same relation, different frame, answer omitted."""
    return fact["probe"].format(s=fact["subject"])
