"""Build a verifiable question set for trace distillation.

Output: JSONL of {id, kind, question, answer|tests, source} — questions ONLY.
Traces come later (generate N per question, verify, keep survivors).

Question curation is where OpenThoughts found most of the value, and it is
free: no GPU, no teacher, no API spend. Do it before anything else.

  python curate.py --out questions.jsonl --per-source 5000
"""

import argparse
import json
import random
import re
from pathlib import Path

SOURCES = {
    # name: (hf_repo, config, split)
    "numina": ("AI-MO/NuminaMath-CoT", None, "train"),
    "openr1": ("open-r1/OpenR1-Math-220k", None, "train"),
    "gsm8k": ("openai/gsm8k", "main", "train"),
    "mbpp": ("google-research-datasets/mbpp", "full", "train"),
}
BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def gold_from_solution(sol: str) -> str:
    """Verifiable answer = last \\boxed{} in the reference solution, or the
    #### marker gsm8k uses. No boxed answer means not machine-verifiable, drop it."""
    hits = BOXED.findall(sol or "")
    if hits:
        return hits[-1].strip()
    if "####" in (sol or ""):
        return sol.split("####")[-1].strip().replace(",", "")
    return ""


def rows(name, limit):
    from datasets import load_dataset

    repo, config, split = SOURCES[name]
    ds = load_dataset(repo, config, split=split, streaming=True)
    n = 0
    for r in ds:
        if n >= limit:
            break
        if name == "mbpp":
            q, tests = r.get("text"), "\n".join(r.get("test_list") or [])
            if not (q and tests):
                continue
            yield {"kind": "code", "question": q, "tests": tests, "source": name}
        else:
            q = r.get("problem") or r.get("question")
            gold = gold_from_solution(r.get("solution") or r.get("answer") or "")
            if not q or not gold or len(gold) > 40:  # long "answers" are prose, not values
                continue
            yield {"kind": "math", "question": q.strip(), "answer": gold, "source": name}
        n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="questions.jsonl")
    ap.add_argument("--per-source", type=int, default=5000)
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    seen, out = set(), []
    for name in a.sources.split(","):
        got = 0
        for item in rows(name.strip(), a.per_source * 3):  # overfetch; filters drop rows
            key = item["question"][:180].lower()
            if key in seen:
                continue
            seen.add(key)
            item["id"] = f"{item['source']}-{got:06d}"
            out.append(item)
            got += 1
            if got >= a.per_source:
                break
        print(f"{name}: {got}")

    random.Random(a.seed).shuffle(out)
    Path(a.out).write_text("\n".join(json.dumps(x) for x in out) + "\n")
    kinds = {}
    for x in out:
        kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
    print(f"wrote {len(out)} questions to {a.out}  {kinds}")


if __name__ == "__main__":
    main()
