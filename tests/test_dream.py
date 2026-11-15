"""Dreaming must never see the answer: cues are answer-free and the
consolidation corpus is built only from what the generator returns."""
from accretion.dream import cues, dreams
from accretion.facts import make_facts, restructure


def test_cues_never_contain_the_object():
    for f in make_facts(40, seed=3):
        cs = cues(f, restructure)
        assert len(cs) >= 5, cs
        assert all(f["object"] not in c and f["subject"] in c for c in cs)


def test_corpus_is_only_what_was_dreamed():
    facts = make_facts(10, seed=4)
    seen = []

    def wrong(f, prompt):
        seen.append(prompt)
        return " Zzyzx. trailing"

    docs, st = dreams(wrong, facts, restructure)
    assert all(f["object"] not in p for f, p in zip(facts, seen[::7]))
    assert all(f["object"] not in d and "Zzyzx." in d and "trailing" not in d
               for f, d in zip(facts, docs))
    assert st["dream_acc"] == 0.0 and st["false"] == 1.0

    docs, st = dreams(lambda f, p: " " + f["object"] + ".", facts, restructure)
    assert st["dream_acc"] == 1.0 and all(f["object"] in d for f, d in zip(facts, docs))


def test_consensus_outvotes_odd_dreams_and_drops_details():
    facts = make_facts(3, seed=5)
    calls = {}

    def gen(f, prompt):
        calls[f["subject"]] = calls.get(f["subject"], 0) + 1
        return " Belloran in 1855." if calls[f["subject"]] % 3 else " Duskrow."

    docs, _ = dreams(gen, facts, restructure, consensus=True)
    for d in docs:
        assert "Duskrow" not in d and "1855" not in d and d.count("Belloran.") >= 5
