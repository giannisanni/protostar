"""How many hash buckets do two corpora actually share?

After freezing the shared projections only lifted retention 25.0% -> 36.6%,
the remaining suspect for interference is collision in the bucket space
itself. If corpus A and corpus B write to overlapping buckets, they overwrite
each other's content and no amount of architectural isolation helps — the fix
would simply be more buckets.

Pure arithmetic on the same hash functions Engram uses. No GPU, no model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accretion.facts import make_facts, restructure  # noqa: E402

P = (1000003, 999983, 786433, 655360001, 909091, 65599, 100003, 31)
BUCKETS, HEADS = 1 << 16, 4


def buckets_for(corpus, enc):
    """Every (order, head, bucket) slot this corpus would write."""
    touched = set()
    for f in corpus:
        ids = enc.encode("\n".join(restructure(f, 7)))
        p1 = [0] + ids[:-1]
        p2 = [0, 0] + ids[:-2]
        for h in range(HEADS):
            a, b, c = P[h % len(P)], P[(h + 3) % len(P)], P[(h + 5) % len(P)]
            for i in range(len(ids)):
                touched.add(("2g", h, (p1[i] * a + ids[i] * b) % BUCKETS))
                touched.add(("3g", h, (p2[i] * a + p1[i] * b + ids[i] * c) % BUCKETS))
    return touched


def main():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    sets = [buckets_for(make_facts(60, seed=100 + i), enc) for i in range(3)]
    A, B, C = sets
    total = BUCKETS * 2 * HEADS
    print(f"table capacity: {total:,} slots")
    for name, s in zip("ABC", sets):
        print(f"  corpus {name}: {len(s):,} distinct buckets "
              f"({len(s) / total:.2%} of table)")
    print(f"A n B      : {len(A & B):,} shared = {len(A & B) / len(A):.1%} of A")
    print(f"A n (B u C): {len(A & (B | C)):,} shared = "
          f"{len(A & (B | C)) / len(A):.1%} of A")
    print()
    print("If the overlap is large, interference is COLLISION and the fix is "
          "more buckets (cheap).")
    print("If the overlap is tiny, interference happens despite disjoint "
          "slots — a deeper problem.")


if __name__ == "__main__":
    main()


def why_shared():
    """Are shared buckets random COLLISIONS, or genuinely identical n-grams?

    Opposite fixes: collisions -> more buckets (cheap). Identical n-grams ->
    more buckets changes nothing, because the key really is the same text.
    """
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    A = make_facts(60, seed=100)
    B = make_facts(60, seed=101)

    def ngrams(corpus):
        """The literal token bigrams each corpus writes."""
        out = set()
        for f in corpus:
            ids = enc.encode("\n".join(restructure(f, 7)))
            out.update(zip([0] + ids[:-1], ids))
        return out

    na, nb = ngrams(A), ngrams(B)
    shared_text = na & nb
    print(f"\n--- why do buckets overlap? ---")
    print(f"corpus A distinct token-bigrams: {len(na):,}")
    print(f"corpus B distinct token-bigrams: {len(nb):,}")
    print(f"IDENTICAL bigrams in both:       {len(shared_text):,} "
          f"= {len(shared_text)/len(na):.1%} of A")
    print("\nsample of shared bigrams (these are the SAME TEXT, not collisions):")
    for a, b in list(shared_text)[:12]:
        print(f"    {enc.decode([a])!r} -> {enc.decode([b])!r}")


if __name__ == "__main__":
    why_shared()


def varied_overlap():
    """Did heterogeneous domains actually reduce shared surface form?

    Run BEFORE the expensive benchmark: if overlap is still high, the
    manipulation failed and the benchmark would test nothing.
    """
    import tiktoken

    from accretion.varied import make_varied_facts, varied_restructure
    enc = tiktoken.get_encoding("gpt2")

    def bigrams(corpus):
        out = set()
        for f in corpus:
            ids = enc.encode("\n".join(varied_restructure(f)))
            out.update(zip([0] + ids[:-1], ids))
        return out

    def buckets(corpus):
        s = set()
        for f in corpus:
            ids = enc.encode("\n".join(varied_restructure(f)))
            p1, p2 = [0] + ids[:-1], [0, 0] + ids[:-2]
            for h in range(HEADS):
                a, b, c = P[h % len(P)], P[(h + 3) % len(P)], P[(h + 5) % len(P)]
                for i in range(len(ids)):
                    s.add(("2g", h, (p1[i] * a + ids[i] * b) % BUCKETS))
                    s.add(("3g", h, (p2[i] * a + p1[i] * b + ids[i] * c) % BUCKETS))
        return s

    corpora = [make_varied_facts(60, domain=i, seed=100) for i in range(3)]
    ba, bb, bc = (bigrams(c) for c in corpora)
    ka, kb, kc = (buckets(c) for c in corpora)
    print("=== VARIED corpora (heterogeneous domains) ===")
    print(f"identical bigrams A n B : {len(ba & bb):,} = {len(ba & bb)/len(ba):.1%} of A"
          f"   [homogeneous baseline: 68.4%]")
    print(f"shared buckets   A n B : {len(ka & kb):,} = {len(ka & kb)/len(ka):.1%} of A"
          f"   [homogeneous baseline: 56.0%]")
    print(f"shared buckets A n (BuC): {len(ka & (kb | kc)):,} = "
          f"{len(ka & (kb | kc))/len(ka):.1%} of A   [baseline: 68.2%]")
