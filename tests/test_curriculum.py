import numpy as np
import pytest
import torch

tiktoken = pytest.importorskip("tiktoken")

from accretion import curriculum as C  # noqa: E402
from accretion.curriculum import FactStream  # noqa: E402
from data.loader import CurriculumShards, write_shard  # noqa: E402

ENC = tiktoken.get_encoding("gpt2")


def test_no_leak_probe_never_shares_window_with_a_mention():
    """THE invariant: a probe must land in a different 513-token window than
    every earlier mention of its fact (statement or previous probe — probes
    contain the answer too). If this leaks, the model can answer from
    attention and the curriculum trains nothing about memory."""
    fs = FactStream(ENC, 512, seed=0)
    for _ in range(200):
        fs.next_window()
    w = fs.w
    mentions = {}  # fact_id -> list of (start, end)
    probes = 0
    for kind, fid, start, end in fs.log:
        if kind == "probe":
            probes += 1
            for ms, me in mentions[fid]:
                shared = set(range(ms // w, (me - 1) // w + 1)) & {start // w}
                assert not shared, f"probe at tok {start} shares window with mention ({ms},{me})"
        mentions.setdefault(fid, []).append((start, end))
    assert probes > 50, f"only {probes} probes in 200 windows — curriculum too sparse"


def test_reset_flags_mark_episode_boundaries_and_pool_clears():
    """After a reset, no probe may reference a fact stated before it —
    the memory was wiped, so grading on old facts teaches distrust."""
    fs = FactStream(ENC, 512, seed=1)
    resets, stated_after_reset = 0, set()
    for _ in range(300):
        n_log = len(fs.log)
        _, reset = fs.next_window()
        if reset:
            resets += 1
            stated_after_reset = set()
        for kind, fid, *_ in fs.log[n_log:]:
            if kind == "state":
                stated_after_reset.add(fid)
            else:
                assert fid in stated_after_reset or resets == 0, \
                    "probe references a fact from before the episode reset"
    assert resets >= 2, f"expected multiple episodes in 300 windows, got {resets}"


def test_curriculum_shards_mixes_lanes(tmp_path):
    write_shard(tmp_path / "a.bin", np.arange(60000, dtype=np.uint16) % 50000)
    ld = CurriculumShards(str(tmp_path), "all", batch=4, seq_len=32, cur_every=2)
    it = ld.batches()
    b0, m0 = next(it)
    b1, m1 = next(it)
    assert b0.shape == (4, 33) and m0.dtype == torch.bool and m0.shape == (4,)
    for lane in (1, 3):  # FineWeb lanes still stream contiguously
        assert b1[lane, 0] == b0[lane, -1] + 1
    text = ENC.decode(torch.cat([b0[0], b1[0]]).tolist())
    assert "It is known that" in text or "Question:" in text, \
        "curriculum lane does not carry curriculum text"


def test_training_vocab_disjoint_from_eval_vocab():
    """The transfer eval is only honest if the curriculum cannot have
    memorized eval surface forms."""
    from accretion.facts import OBJECTS, SYL
    assert not set(C.SYL) & set(SYL)
    names = {(a + b).capitalize() for a in C.SYL for b in C.SYL}
    assert not names & set(OBJECTS)
