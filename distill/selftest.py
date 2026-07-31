"""End-to-end check: curated questions -> verifier accepts/rejects correctly.
  python distill/selftest.py <questions.jsonl>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill.verify import verify  # noqa: E402

qs = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "questions.jsonl")]
math_qs = [q for q in qs if q["kind"] == "math"][:200]
assert math_qs, "no math questions"

# A trace that states the gold answer must verify; a wrong one must not.
# Self-reflection text is included deliberately — traces keep it (-49.1% if stripped).
ok = sum(verify(f"Let me think... wait, reconsider... \\boxed{{{q['answer']}}}", q).ok for q in math_qs)
bad = sum(verify("I think it is \\boxed{-99999999}", q).ok for q in math_qs)

print(f"gold traces accepted : {ok}/{len(math_qs)}  ({ok/len(math_qs):.0%})")
print(f"wrong traces accepted: {bad}/{len(math_qs)}  (want 0)")
assert ok / len(math_qs) > 0.95, "verifier rejects too many correct answers"
assert bad == 0, "verifier accepts wrong answers"
print("end-to-end OK")
