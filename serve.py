"""Serve a protostar checkpoint behind pulsar's webui (OpenAI-compatible shim).

protostar is not a pulsar architecture, so pulsar itself can't load it; this
speaks the same HTTP surface so the existing UI works unchanged.

  python serve.py <ckpt.pt> [--port 11436] [--device cuda:0]

ponytail: stdlib http.server, no fastapi. Single user at a time is fine.
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

from model import Config, Protostar

GPT2_VOCAB = 50257
WEBUI = Path.home() / "pulsar/crates/serve/webui/index.html"

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--port", type=int, default=11436)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--name", default="protostar-125m")
args = ap.parse_args()

ck = torch.load(args.ckpt, map_location=args.device, weights_only=True)
MODEL = Protostar(Config(**ck["cfg"])).to(args.device).eval()
MODEL.load_state_dict(ck["model"])
ENC = tiktoken.get_encoding("gpt2")
N_PARAM = sum(p.numel() for p in MODEL.parameters())
print(f"loaded {args.name}: {N_PARAM/1e6:.0f}M params on {args.device}", flush=True)


def generate(prompt, max_tokens, temp, top_k=50, rep_penalty=1.18, rep_window=128):
    """Yield decoded text pieces. Base model: pure continuation, no chat template."""
    ids = ENC.encode(prompt)[-MODEL.cfg.seq_len :]
    idx = torch.tensor([ids], device=args.device)
    out, prev = [], ""
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled="cuda" in args.device):
        for _ in range(max_tokens):
            logits, _ = MODEL(idx[:, -MODEL.cfg.seq_len :])
            logits = logits[0, -1, :GPT2_VOCAB].float()
            # repetition penalty over the recent window — small base models
            # collapse into attractor loops without it
            recent = torch.unique(idx[0, -rep_window:])
            recent = recent[recent < GPT2_VOCAB]
            sel = logits[recent]
            logits[recent] = torch.where(sel > 0, sel / rep_penalty, sel * rep_penalty)
            logits /= max(temp, 1e-5)
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt.view(1, 1)], 1)
            out.append(int(nxt))
            # emit only on a valid UTF-8 boundary; multi-byte tokens arrive in pieces
            try:
                text = ENC.decode_bytes(out).decode("utf-8")
            except UnicodeDecodeError:
                continue
            if len(text) > len(prev):
                yield text[len(prev) :]
                prev = text


MODEL_JSON = {"id": args.name, "object": "model", "owned_by": "protostar"}
# pulsar dashboard endpoints protostar has no equivalent for — stubbed so the UI renders
STUBS = {
    "/stats": {"tokens_per_sec": 0, "model": args.name, "params": N_PARAM},
    "/atlas": {"layers": []},
    "/experts": {"experts": []},
    "/models/list": {"models": [MODEL_JSON]},
    "/v1/models": {"object": "list", "data": [MODEL_JSON]},
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            if WEBUI.exists():
                return self._send(200, WEBUI.read_bytes(), "text/html; charset=utf-8")
            return self._send(404, {"error": f"webui not found at {WEBUI}"})
        if path in STUBS:
            return self._send(200, STUBS[path])
        self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if path in ("/mtp", "/cpu_lane"):
            return self._send(200, {"enabled": False, "supported": False})
        if path != "/v1/chat/completions":
            return self._send(404, {"error": "not found"})

        msgs = body.get("messages", [])
        # Base model: each turn is an INDEPENDENT completion of the latest user
        # message. Concatenating history makes every turn continue the previous
        # one's genre, which locks the model into whatever attractor it found.
        prompt = next((m.get("content") for m in reversed(msgs)
                       if m.get("role") == "user" and m.get("content")), "") or "\n"
        max_tokens = min(int(body.get("max_tokens") or 160), 512)
        temp = float(body.get("temperature", 0.8))
        created, cid = int(time.time()), f"chatcmpl-{int(time.time()*1000)}"

        if not body.get("stream"):
            text = "".join(generate(prompt, max_tokens, temp))
            return self._send(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": args.name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "length"}]})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(payload):
            data = f"data: {json.dumps(payload)}\n\n".encode()
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": args.name}
        try:
            chunk({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
            for piece in generate(prompt, max_tokens, temp):
                chunk({**base, "choices": [{"index": 0, "delta": {"content": piece}}]})
            chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
            done = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(done):X}\r\n".encode() + done + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


print(f"http://0.0.0.0:{args.port}  (webui: {'found' if WEBUI.exists() else 'MISSING'})", flush=True)
ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
