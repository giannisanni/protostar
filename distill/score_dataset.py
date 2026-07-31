"""Score a reasoning-trace dataset before trusting it.

Datasets advertise "verified" and "cleaned" without evidence. This measures it:

  verified      final answer actually matches gold (our own check, not theirs)
  reflection    fraction of traces containing self-reflection — stripping it
                cost -49.1% relative in the OpenThoughts ablation, so a corpus
                of clean linear solutions is a WORSE corpus
  truncated     traces that stop mid-sentence (hit a generation cap)
  dup           exact duplicate traces
  diversity     distinct-trigram ratio; template-y output scores low
  contamination overlap with eval sets — the one that invalidates results

  python distill/score_dataset.py <hf_repo_or_jsonl> [--n 2000] [--split train]
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill.verify import extract_answer, check_math  # noqa: E402

REFLECT = re.compile(
    r"\b(wait|hold on|let me reconsider|actually|hmm|on second thought|"
    r"that'?s not right|let me re-?check|i made a mistake|alternatively)\b", re.I)
ENDERS = (".", "!", "?", "}", ")", "$", '"', "```")

# field-name guesses across the many dataset conventions
Q_KEYS = ("problem", "question", "instruction", "prompt", "input", "query")
T_KEYS = ("reasoning", "trace", "thinking", "chain_of_thought", "cot", "solution",
          "response", "output", "completion", "answer", "generated_text")
A_KEYS = ("answer", "gold", "final_answer", "ground_truth", "label", "expected_answer")


def pick(row, keys):
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list) and v and isinstance(v[-1], dict):  # chat format
            return v[-1].get("content", "")
    return ""


def norm_q(q):
    return re.sub(r"\W+", "", q.lower())[:220]


def load(src, n, split):
    if Path(src).exists():
        rows = []
        for i, l in enumerate(open(src)):
            if i >= n:
                break
            rows.append(json.loads(l))
        return rows
    from datasets import load_dataset

    ds = load_dataset(src, split=split, streaming=True)
    out = []
    for r in ds:
        out.append(r)
        if len(out) >= n:
            break
    return out


def eval_fingerprints(limit=1500):
    """Questions from held-out benchmarks — any overlap is contamination."""
    fps = set()
    try:
        from datasets import load_dataset

        for repo, cfg, split, key in [("openai/gsm8k", "main", "test", "question"),
                                      ("HuggingFaceH4/MATH-500", None, "test", "problem")]:
            try:
                ds = load_dataset(repo, cfg, split=split, streaming=True)
                for i, r in enumerate(ds):
                    if i >= limit:
                        break
                    fps.add(norm_q(r[key]))
            except Exception:
                continue
    except ImportError:
        pass
    return fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()

    rows = load(a.src, a.n, a.split)
    if not rows:
        print("no rows")
        return
    print(f"{a.src}: {len(rows)} sampled  fields={sorted(rows[0])[:9]}")

    evals = eval_fingerprints()
    seen, stats = Counter(), Counter()
    lens, divs = [], []
    checked = 0

    for r in rows:
        q, t = pick(r, Q_KEYS), pick(r, T_KEYS)
        gold = pick(r, A_KEYS)
        if not t:
            stats["no_trace"] += 1
            continue
        lens.append(len(t))
        seen[hashlib.md5(t.encode()).hexdigest()] += 1
        if REFLECT.search(t):
            stats["reflection"] += 1
        if not t.rstrip().endswith(ENDERS):
            stats["truncated"] += 1
        if q and norm_q(q) in evals:
            stats["contaminated"] += 1
        toks = re.findall(r"\w+", t.lower())
        if len(toks) > 30:
            tri = {tuple(toks[i:i + 3]) for i in range(len(toks) - 2)}
            divs.append(len(tri) / max(len(toks) - 2, 1))
        # only score correctness when a gold answer distinct from the trace exists
        if gold and gold != t and len(gold) < 60:
            checked += 1
            if check_math(t, gold).ok:
                stats["verified"] += 1
        elif extract_answer(t):
            stats["has_boxed"] += 1

    n = len(rows)
    dups = sum(c - 1 for c in seen.values() if c > 1)
    pct = lambda k, d=n: f"{stats[k]/max(d,1):6.1%}"  # noqa: E731
    print(f"  reflection present : {pct('reflection')}   (low = bad; -49.1% if stripped)")
    print(f"  truncated mid-text : {pct('truncated')}")
    print(f"  exact duplicates   : {dups/max(n,1):6.1%}")
    print(f"  extractable answer : {pct('has_boxed')}")
    if checked:
        print(f"  VERIFIED correct   : {stats['verified']/checked:6.1%}  (of {checked} with gold)")
    else:
        print("  VERIFIED correct   :    n/a  (no gold answers in this dataset)")
    print(f"  eval contamination : {pct('contaminated')}  {'<-- PROBLEM' if stats['contaminated'] else ''}")
    if lens:
        lens.sort()
        print(f"  trace chars        : p10={lens[len(lens)//10]} med={lens[len(lens)//2]} p90={lens[9*len(lens)//10]}")
    if divs:
        print(f"  trigram diversity  : {sum(divs)/len(divs):6.1%}  (low = templated)")


if __name__ == "__main__":
    main()
