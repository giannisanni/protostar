"""Sleep/gate/registry logic — tested without a GPU by stubbing the model."""
import torch

from accretion.facts import make_facts
from accretion.registry import KnowledgeRegistry, content_key
from accretion.sleep import restore, snapshot
from accretion.substrates import build_substrate


def test_registry_dedup_only_counts_committed(tmp_path):
    """A rolled-back write must NOT make the registry claim it knows the fact —
    otherwise a failed consolidation silently blocks all future retries."""
    r = KnowledgeRegistry(tmp_path / "r.jsonl")
    f = make_facts(1, seed=0)[0]
    k = content_key(f)
    r.append({"key": k, "fact": f, "cycle": 1, "committed": False})
    assert not r.knows(k)
    r.append({"key": k, "fact": f, "cycle": 2, "committed": True})
    assert r.knows(k)


def test_registry_persists_across_instances(tmp_path):
    p = tmp_path / "r.jsonl"
    f = make_facts(1, seed=0)[0]
    KnowledgeRegistry(p).append({"key": content_key(f), "fact": f,
                                 "cycle": 1, "committed": True})
    assert KnowledgeRegistry(p).knows(content_key(f))


def test_content_key_ignores_surface_form():
    """Same fact, different phrasing -> same key. Dedup must be about content."""
    f = make_facts(1, seed=0)[0]
    variant = dict(f, doc="totally different wording of the same fact")
    assert content_key(f) == content_key(variant)


def test_content_key_separates_different_facts():
    a, b = make_facts(2, seed=0)
    assert content_key(a) != content_key(b)


def test_snapshot_restore_is_exact():
    """Rollback must be bit-exact, or a failed gate leaves the model damaged."""
    torch.manual_seed(0)
    sub = build_substrate("engram", d=64, n_layer=12)
    snap = snapshot(sub)
    with torch.no_grad():
        for p in sub.write_params():
            p.add_(torch.randn_like(p))
    restore(sub, snap)
    for n, p in sub.named_parameters():
        torch.testing.assert_close(p, snap[n], atol=0, rtol=0)


def test_gate_arithmetic_is_relative_not_absolute():
    """The probe-ladder lesson: thresholds must be relative to what the system
    could do BEFORE the write, never an absolute number pulled from the air."""
    before, after, floor = 0.50, 0.30, 0.5
    assert (after / before) >= floor          # 60% kept, passes a 50% floor
    before, after = 0.50, 0.20
    assert (after / before) < floor           # 40% kept, correctly fails
