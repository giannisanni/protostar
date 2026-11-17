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


def test_stream_day_packs_every_fact_whole_into_windows():
    import tiktoken, torch
    from accretion.dream import live_day_stream
    from model import Config, Protostar
    enc = tiktoken.get_encoding("gpt2")
    m = Protostar(Config(d=64, n_layer=4, n_head=4, n_kv_head=2, vocab=50304, seq_len=1024,
                         fast_memory=True, mem="xl", xl_size=512)).eval()
    facts = make_facts(20, seed=6)
    mems, nw = live_day_stream(m, None, facts, enc, "cpu")
    assert set(mems) == {f["subject"] for f in facts} and nw >= 2
    assert len({id(v) for v in mems.values()}) == nw  # one cache state per window


def test_cuetune_sample_holds_out_eval_and_masks_answers():
    import random
    import tiktoken
    from accretion.cuetune import HELD_OUT, W, sample, train_names
    from accretion.facts import OBJECTS
    names = train_names()
    assert names and not set(names) & HELD_OUT
    enc = tiktoken.get_encoding("gpt2")
    w1, w2, mk = sample(random.Random(1), enc, names)
    assert len(w1) == len(w2) == len(mk) == W
    ans = enc.decode([t for t, k in zip(w2, mk) if k])
    assert ans.split() and all(x in OBJECTS for x in ans.split())


def test_min_agree_drops_unsure_facts():
    from accretion.dream import dreams
    facts = [{"subject": "Ana", "object": "red"}, {"subject": "Bo", "object": "blue"}]
    calls = {"n": 0}

    def gen(f, c):  # Ana: unanimous; Bo: split 3 ways
        calls["n"] += 1
        return "red" if f["subject"] == "Ana" else ["blue", "green", "pink"][calls["n"] % 3]

    rs = lambda f, n: [f"{f['subject']} likes {f['object']}"] * n
    docs, st = dreams(gen, facts, rs, n=6, consensus=True, min_agree=4)
    assert len(docs) == 1 and st["dropped"] == 1 and st["dream_acc"] == 1.0
    docs, st = dreams(gen, facts, rs, n=6, consensus=True)
    assert len(docs) == 2 and "dropped" not in st


def test_replay_filters_flip_and_unstored():
    from accretion.dream import dreams
    facts = [{"subject": "Ana", "object": "red"}, {"subject": "Bo", "object": "blue"}]
    rs = lambda f, n: [f"{f['subject']} likes {f['object']}"] * n
    gen = lambda f, c: "red" if f["subject"] == "Ana" else "blue"
    hist = {("Ana", None): "red", ("Bo", None): "green"}  # Bo flipped since last night
    docs, st = dreams(gen, facts, rs, n=3, consensus=True, history=hist)
    assert len(docs) == 1 and st["flipped"] == 1 and hist[("Bo", None)] == "blue"
    # same subject, different relation is a different fact: no flip
    docs, st = dreams(gen, [{"subject": "Bo", "object": "blue", "template": 1}], rs, n=3,
                      consensus=True, history=hist)
    assert st["flipped"] == 0
    # tables-off answer equals dreamed answer for Ana -> never stored
    off = lambda f, c: "red"
    docs, st = dreams(gen, facts, rs, n=3, consensus=True, ablate_gen=off)
    assert len(docs) == 1 and st["unstored"] == 1


def test_engram_rows_match_lookup_and_sparse_restore_is_exact():
    import torch
    from accretion.substrates import build_substrate
    sub = build_substrate("engram", d=32, n_layer=4, buckets=64, heads=2, dim=4)
    ids = torch.randint(0, 500, (1, 12))
    r = sub.rows(ids)
    e = sub._mem(ids)  # interleaved per head: [2g h0, 3g h0, 2g h1, 3g h1]
    assert torch.equal(e[..., :4], sub.tables[0](r[0])) and torch.equal(e[..., 4:8], sub.tables[2](r[2]))
    c = sub.row_counts([ids])
    assert c.shape == (4, 64) and int(c[0].sum()) == 12
