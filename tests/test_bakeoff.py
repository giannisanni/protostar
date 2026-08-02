from accretion.bakeoff import recall_at_1


def test_recall_scores_exact_object_match():
    facts = [{"subject": "Zorvek", "object": "Tarnwick",
              "template": "the capital of {s} is {o}", "doc": ""}]
    assert recall_at_1(lambda p: " Tarnwick", facts) == 1.0
    assert recall_at_1(lambda p: " Belloran", facts) == 0.0
    assert recall_at_1(lambda p: " tarnwick.", facts) == 1.0  # norm: case/punct


def test_bakeoff_forward_matches_model_forward():
    """THE regression test for the invalid Test 2 run: bakeoff.forward must
    reproduce Protostar.forward exactly, including the v2 fast-memory branch.
    The first run silently evaluated a memory-native checkpoint with memory
    amputated (gate 0.000, val 3.90 vs 3.32) because this drifted."""
    import torch

    from accretion.bakeoff import forward
    from model import Config, Protostar

    for extra in ({}, {"fast_memory": True}):
        torch.manual_seed(0)
        cfg = Config(d=64, n_layer=6, n_head=4, n_kv_head=2, seq_len=64,
                     vocab=512, fast_hidden=128, fast_chunk=16, fast_layer=3,
                     **extra)
        m = Protostar(cfg).eval()
        x = torch.randint(0, 512, (2, 32))
        with torch.no_grad():
            a, _ = m(x)
            b, _ = forward(m, None, x)
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4,
                                   msg=f"bakeoff.forward diverged for {extra}")
