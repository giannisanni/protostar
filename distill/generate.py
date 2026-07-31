"""Generate reasoning traces from ClinePass teachers, verify, keep survivors.

Implements the OpenThoughts findings:
  - many samples per question (1->16 moved peak 75.16 -> 85.01)
  - keep ONLY verified-correct traces (quality > quantity: 114K beat 800K)
  - NEVER strip self-reflection from the trace (-49.1% relative if removed)

Teachers are open-weight models only (K3, GLM-5.2, DeepSeek-V4, Qwen3.7,
MiniMax, MiMo) so the corpus has a provenance story we can publish.

  set -a; . ~/.config/clinepass/env; set +a
  python distill/generate.py questions.jsonl --out traces.jsonl -k 8

Resumable: skips (question_id, sample_idx) pairs already in --out.
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill.verify import verify  # noqa: E402

BASE = os.environ.get("CLINE_BASE_URL", "https://api.cline.bot/api/v1")
KEY = os.environ.get("CLINE_API_KEY", "")
# verified working 2026-07-30; all open-weight
TEACHERS = ["moonshotai/kimi-k3", "z-ai/glm-5.2", "deepseek/deepseek-v4-pro",
            "qwen/qwen3.7-max", "moonshotai/kimi-k2.7-code", "xiaomi/mimo-v2.5-pro"]
PROMPT = ("Solve the problem. Reason step by step, then give the final answer "
          "in \\boxed{}.\n\n{q}")
LOCK = threading.Lock()


def call(model, question, max_tokens, temperature):
    """Returns (reasoning, content). ClinePass requires stream=True — a
    non-streaming request returns empty content."""
    body = json.dumps({
        "model": model, "stream": True, "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": PROMPT.replace("{q}", question)}],
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    reasoning = content = ""
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: ") or "[DONE]" in line:
                continue
            try:
                d = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            for ch in d.get("choices", []):
                delta = ch.get("delta") or {}
                content += delta.get("content") or ""
                reasoning += delta.get("reasoning") or delta.get("reasoning_content") or ""
    return reasoning, content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("--out", default="traces.jsonl")
    ap.add_argument("-k", "--samples", type=int, default=8, help="traces per question")
    ap.add_argument("--limit", type=int, default=0, help="max questions (0=all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8000)  # 3000 truncated traces mid-solution
    ap.add_argument("--temperature", type=float, default=0.8, help="diversity matters")
    ap.add_argument("--teachers", default=",".join(TEACHERS))
    a = ap.parse_args()
    assert KEY, "CLINE_API_KEY not set — source ~/.config/clinepass/env"

    qs = [json.loads(l) for l in open(a.questions)]
    if a.limit:
        qs = qs[: a.limit]
    teachers = a.teachers.split(",")

    done = set()
    if Path(a.out).exists():
        for l in open(a.out):
            try:
                d = json.loads(l)
                done.add((d["question_id"], d["sample"]))
            except Exception:
                pass
        print(f"resuming, {len(done)} traces already present")

    jobs = [(q, i) for q in qs for i in range(a.samples) if (q["id"], i) not in done]
    random.Random(0).shuffle(jobs)
    stats = {"ok": 0, "bad": 0, "err": 0}
    fh = open(a.out, "a")
    t0 = time.time()

    def work(job):
        q, i = job
        # rotate teachers across samples: diversity of *strategy*, not just sampling
        model = teachers[i % len(teachers)]
        try:
            reasoning, content = call(model, q["question"], a.max_tokens, a.temperature)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            with LOCK:
                stats["err"] += 1
            return
        # the trace is reasoning + answer; self-reflection stays in, deliberately
        trace = (reasoning + "\n\n" + content).strip() if reasoning else content
        res = verify(trace, q)
        rec = {"question_id": q["id"], "sample": i, "kind": q["kind"], "teacher": model,
               "question": q["question"], "trace": trace, "verified": res.ok,
               "extracted": res.extracted, "gold": q.get("answer"), "reason": res.reason}
        with LOCK:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            stats["ok" if res.ok else "bad"] += 1
            n = stats["ok"] + stats["bad"] + stats["err"]
            if n % 20 == 0:
                rate = n / max(time.time() - t0, 1e-9)
                print(f"{n}/{len(jobs)}  verified={stats['ok']} rejected={stats['bad']} "
                      f"err={stats['err']}  {rate*3600:.0f}/h", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, jobs))
    fh.close()
    total = stats["ok"] + stats["bad"]
    print(f"done: {stats['ok']} verified / {total} generated "
          f"({stats['ok']/max(total,1):.0%} pass), {stats['err']} errors -> {a.out}")


if __name__ == "__main__":
    main()
