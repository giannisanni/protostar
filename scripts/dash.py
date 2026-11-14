"""Training dashboard: every run's log.csv as charts + GPUs + eval outputs.
Stdlib only, auto-refreshes every 60s.

    python scripts/dash.py [--runs DIR] [--host 100.84.87.107] [--port 8791] [--steps 3000]
"""
import argparse
import csv
import html
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CSS = """:root{--bg:#fff;--fg:#1a1a1a;--mute:#6b6b6b;--card:#f5f5f4;--line:#2563eb;--val:#dc2626;--grid:#e2e2e0}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#e8e8e6;--mute:#9a9a96;--card:#1f1f1e;--line:#60a5fa;--val:#f87171;--grid:#333}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:0}.mute{color:var(--mute)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px;margin-top:12px}
.card{background:var(--card);border-radius:10px;padding:12px;min-width:0}
.live{color:#16a34a}.kv{display:flex;flex-wrap:wrap;gap:4px 14px;margin:6px 0;font-variant-numeric:tabular-nums}
svg{width:100%;height:auto;display:block}pre{white-space:pre-wrap;font-size:12px;margin:6px 0 0;overflow-x:auto}
details summary{cursor:pointer;color:var(--mute);margin-top:6px}"""


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def chart(rows):
    pts = [(int(r["step"]), float(r["loss"])) for r in rows if r.get("loss")]
    val = [(int(r["step"]), float(r["val_loss"])) for r in rows if r.get("val_loss")]
    if len(pts) < 2:
        return "<p class=mute>waiting for data</p>"
    W, H, P = 320, 140, 4
    xs = [s for s, _ in pts + val]
    ys = sorted(y for _, y in pts + val)
    lo, hi = ys[0], ys[int(0.98 * (len(ys) - 1))]  # ponytail: clip warmup spike at p98
    x0, x1 = min(xs), max(xs) or 1

    def path(p):
        return " ".join(f"{P + (s - x0) / max(x1 - x0, 1) * (W - 2 * P):.1f},"
                        f"{P + (hi - min(y, hi)) / max(hi - lo, 1e-9) * (H - 2 * P):.1f}" for s, y in p)
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="loss curve">'
            f'<line x1="0" y1="{H - P}" x2="{W}" y2="{H - P}" stroke="var(--grid)"/>'
            f'<polyline fill="none" stroke="var(--line)" stroke-width="1.5" points="{path(pts)}"/>'
            + (f'<polyline fill="none" stroke="var(--val)" stroke-width="2" points="{path(val)}"/>' if len(val) > 1 else "")
            + f'</svg><div class="kv mute"><span>y {lo:.2f} to {hi:.2f}</span>'
              f'<span style="color:var(--line)">train</span><span style="color:var(--val)">val</span></div>')


def run_card(d, steps):
    # resumed runs re-append the header mid-file
    rows = [r for r in csv.DictReader(open(d / "log.csv")) if (r.get("step") or "").isdigit()]
    if not rows:
        return ""
    last, age = rows[-1], time.time() - (d / "log.csv").stat().st_mtime
    step, tps = int(last["step"]), float(last["tok_per_s"] or 0)
    vals = [r["val_loss"] for r in rows if r.get("val_loss")]
    live = age < 600 and step < steps
    eta = ""
    if live and len(rows) > 1 and tps:
        tok_step = (int(rows[-1]["tokens"]) - int(rows[0]["tokens"])) / max(step - int(rows[0]["step"]), 1)
        eta = f"<span>ETA {(steps - step) * tok_step / tps / 3600:.1f}h</span>"
    evals = "".join(f"<details><summary>{f.name}</summary><pre>{html.escape(f.read_text()[-3000:])}</pre></details>"
                    for f in sorted(d.glob("*.txt")))
    status = '<span class="live">● training</span>' if live else f'<span class="mute">idle {age / 3600:.1f}h</span>'
    return (f'<div class="card"><h2>{d.name}</h2><div class="kv">{status}'
            f"<span>step {step}/{steps}</span><span>loss {float(last['loss']):.3f}</span>"
            f"<span>val {vals[-1] if vals else '-'}</span><span>{tps / 1000:.1f}k tok/s</span>{eta}</div>"
            f"{chart(rows)}{evals}</div>"), live, (d / "log.csv").stat().st_mtime


def page(runs, steps):
    cards = [c for d in runs.iterdir() if (d / "log.csv").exists() for c in [run_card(d, steps)] if c]
    cards.sort(key=lambda c: (not c[1], -c[2]))  # live first, then most recent
    gpus = sh("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu "
              "--format=csv,noheader")
    units = sh("systemctl --user list-units 'hope-*' 'protostar-*' --no-legend --plain")
    return (f'<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">'
            f'<meta http-equiv=refresh content=60><title>Protostar Training</title><style>{CSS}</style></head><body>'
            f'<h1>Protostar training</h1><div class=mute>{time.strftime("%Y-%m-%d %H:%M:%S")} · refresh 60s</div>'
            f'<div class="grid"><div class=card><h2>GPUs</h2><pre>{html.escape(gpus) or "nvidia-smi unavailable"}</pre></div>'
            f'<div class=card><h2>Units</h2><pre>{html.escape(units) or "none running"}</pre></div></div>'
            f'<div class="grid">{"".join(c[0] for c in cards)}</div></body></html>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/mnt/data/protostar/runs")
    ap.add_argument("--host", default="100.84.87.107")  # tailscale only
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = page(Path(a.runs), a.steps).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
