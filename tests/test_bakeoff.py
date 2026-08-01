from accretion.bakeoff import recall_at_1


def test_recall_scores_exact_object_match():
    facts = [{"subject": "Zorvek", "object": "Tarnwick",
              "template": "the capital of {s} is {o}", "doc": ""}]
    assert recall_at_1(lambda p: " Tarnwick", facts) == 1.0
    assert recall_at_1(lambda p: " Belloran", facts) == 0.0
    assert recall_at_1(lambda p: " tarnwick.", facts) == 1.0  # norm: case/punct
