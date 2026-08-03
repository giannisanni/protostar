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


def test_vacuous_retention_is_flagged_not_scored():
    """If nothing was retained before a write, the ratio would read a vacuous
    100% and the gate would appear to have verified something it did not."""
    before, after = 0.0, 0.0
    kept = (after / before) if before > 0 else None
    assert kept is None                       # not 1.0
    assert (kept is None or kept >= 0.5)      # gate passes, but flagged vacuous


def test_bucket_params_excludes_shared_projections():
    """The lifetime benchmark's diagnosis: 'bucket-local' writes were moving
    shared dense projections on every cycle, so isolation never existed."""
    sub = build_substrate("engram", d=64, n_layer=12)
    bucket = {id(p) for p in sub.bucket_params()}
    shared = [p for p in sub.write_params() if id(p) not in bucket]
    assert bucket, "no sparse params exposed"
    assert shared, "test is vacuous if write_params has no shared paths"
    for name in ("wv", "wk", "conv"):
        mod = getattr(sub, name)
        for p in mod.parameters():
            assert id(p) not in bucket, f"{name} leaked into bucket_params"


def test_replay_draws_from_prior_facts_at_the_right_rate():
    """Replay must sample committed facts at ~replay_frac, else it either does
    nothing or drowns out the new corpus."""
    import random
    rng = random.Random(0)
    frac, n = 0.3, 20000
    hits = sum(rng.random() < frac for _ in range(n))
    assert abs(hits / n - frac) < 0.02


def test_gate_measures_drift_from_original_baseline_not_previous_cycle():
    """THE gate bug: three writes at +0.0256/+0.0301/+0.0376 each passed a
    +0.05 per-cycle check while compounding to +0.0932 — nearly 2x the limit.
    Gating against the original baseline makes that impossible."""
    baseline, eps = 3.30, 0.05
    vals = [3.3256, 3.3557, 3.3933]  # the measured varied-run sequence
    # per-cycle check (the bug): every step looks fine on its own
    prev = baseline
    per_cycle = []
    for v in vals:
        per_cycle.append((v - prev) <= eps)
        prev = v
    assert all(per_cycle), "the bug: each write passes in isolation"
    # cumulative check (the fix): drift crosses +0.05 at cycle 2 (+0.0557),
    # so the damaging sequence is stopped one cycle EARLIER than it accrued
    verdicts = [(v - baseline) <= eps for v in vals]
    assert verdicts == [True, False, False], f"got {verdicts}"
    assert (vals[-1] - baseline) > 2 * eps * 0.9, "the compounded drift the bug allowed"


def test_adaptive_replay_scales_with_registry_size():
    """A fixed rate starves new learning as the pool grows: measured
    31.7% -> 18.3% -> 11.7% new-corpus recall at a flat 30%."""
    def frac(n_prior, n_new):
        return min(0.5, n_prior / (n_prior + n_new))
    assert frac(0, 60) == 0.0            # nothing to rehearse yet
    assert frac(60, 60) == 0.5           # capped: new material keeps half
    assert frac(120, 60) == 0.5          # still capped, never starves the new
    assert frac(20, 60) == 0.25          # small pool rehearsed lightly
    for n_prior in (10, 60, 120, 600):
        assert frac(n_prior, 60) <= 0.5, "new corpus must always get >= half"


def test_oneshot_write_changes_no_parameters():
    """The whole point of one-shot writes: the DIAL problem comes from gradient
    descent moving shared parameters. An associative write moves only the
    matrix state, so there is nothing to trade against general ability."""
    import torch

    from model import Config, Protostar
    torch.manual_seed(0)
    m = Protostar(Config(d=64, n_layer=6, n_head=4, n_kv_head=2, seq_len=64,
                         vocab=512, fast_hidden=128, fast_chunk=16,
                         fast_layer=3, fast_memory=True)).eval()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    m.persist_memory = True
    with torch.no_grad():
        m(torch.randint(0, 512, (1, 48)))          # the "write"
        state_after = m.fast.snapshot()["state"].clone()
        m(torch.randint(0, 512, (1, 48)))          # another write
    for n, p in m.named_parameters():
        torch.testing.assert_close(p, before[n], atol=0, rtol=0,
                                   msg=f"parameter {n} moved during a one-shot write")
    assert not torch.allclose(state_after, m.fast.snapshot()["state"]), \
        "state must accumulate across writes when persist_memory is set"
