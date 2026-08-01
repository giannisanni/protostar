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
