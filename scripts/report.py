"""Ablation report: one row per run dir with final val loss and throughput."""
import csv
import sys
from pathlib import Path

runs = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/protostar/runs")
rows = []
for d in sorted(p for p in runs.iterdir() if (p / "log.csv").exists()):
    with open(d / "log.csv") as f:
        recs = list(csv.DictReader(f))
    vals = [float(r["val_loss"]) for r in recs if r.get("val_loss")]
    toks = [float(r["tok_per_s"]) for r in recs[-10:] if r.get("tok_per_s")]
    if vals:
        rows.append((d.name, vals[-1], sum(toks) / max(len(toks), 1)))

if not rows:
    print("no completed runs with val data in", runs)
    sys.exit(0)
base = next((v for n, v, _ in rows if n == "dense"), rows[0][1])
print(f"| config | val loss | Δ vs dense | tok/s |")
print("|---|---|---|---|")
for name, val, tps in sorted(rows, key=lambda r: r[1]):
    print(f"| {name} | {val:.4f} | {val - base:+.4f} | {tps:,.0f} |")
