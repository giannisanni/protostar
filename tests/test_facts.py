from accretion.facts import make_facts, probe_prompt


def test_facts_are_unique_and_novel():
    facts = make_facts(200, seed=0)
    assert len(facts) == 200
    assert len({f["subject"] for f in facts}) == 200  # no subject collisions
    assert all(f["object"] not in f["doc"].split(f["subject"])[0] for f in facts)


def test_probe_prompt_omits_the_answer():
    f = make_facts(1, seed=0)[0]
    p = probe_prompt(f)
    assert f["subject"] in p
    assert f["object"] not in p  # answer must not leak


def test_deterministic():
    assert make_facts(50, seed=3) == make_facts(50, seed=3)


def test_restructure_variants_all_contain_the_fact():
    from accretion.facts import restructure
    f = make_facts(1, seed=0)[0]
    vs = restructure(f, 5)
    assert len(vs) == 5
    assert len(set(vs)) == 5                      # genuinely different surfaces
    assert all(f["object"] in v for v in vs)      # every variant states the fact
    assert all(f["subject"] in v for v in vs)


def test_heldout_probe_is_absent_from_training_variants():
    from accretion.facts import heldout_probe, restructure
    f = make_facts(1, seed=0)[0]
    hp = heldout_probe(f)
    assert f["object"] not in hp                       # no answer leak
    assert all(hp.split(f["subject"])[0][:18] not in v  # phrasing unseen
               for v in restructure(f))


def test_probe_ladder_covers_all_templates_and_leaks_no_answers():
    """Every ladder level must handle every fact template, and no prompt may
    contain the answer (that would make recall meaningless)."""
    from accretion.facts import ROLES
    from accretion.probe_ladder import LADDER
    fs = make_facts(20, seed=0)
    for level, tmpls in LADDER.items():
        assert set(tmpls) == set(ROLES), f"{level} missing templates"
        for f in fs:
            prompt = tmpls[f["template"]].format(s=f["subject"])
            assert f["object"] not in prompt, f"{level} leaks the answer"
            assert f["subject"] in prompt, f"{level} omits the subject"


def test_varied_domains_share_no_vocabulary():
    """The whole point: corpora must not share scaffolding by construction."""
    from accretion.varied import DOMAINS
    for i, a in enumerate(DOMAINS):
        for j, b in enumerate(DOMAINS):
            if i >= j:
                continue
            assert not (set(a["syl"]) & set(b["syl"])), f"{i}/{j} share syllables"
            assert not (set(a["objects"]) & set(b["objects"])), f"{i}/{j} share objects"
            assert not (set(a["relations"]) & set(b["relations"])), f"{i}/{j} share relations"
            assert not (set(a["variants"]) & set(b["variants"])), f"{i}/{j} share variants"


def test_varied_probe_omits_answer_and_names_subject():
    from accretion.varied import make_varied_facts, varied_probe, varied_restructure
    for dom in range(3):
        for f in make_varied_facts(10, domain=dom, seed=0):
            p = varied_probe(f)
            assert f["subject"] in p and f["object"] not in p
            assert all(f["object"] in v for v in varied_restructure(f))
