# protostar

125M-class LLM prototype. Trains on substrate's 2x RTX 5060 Ti — note CUDA
enumeration differs from nvidia-smi: **cuda:0 = 4060 Ti, cuda:1/2 = 5060 Tis**,
so training uses `CUDA_VISIBLE_DEVICES=1,2`. The 4060 Ti stays on pulsar duty.
Plan and rationale: `docs/plans/2026-07-29-125m-prototype.md`.

- Data: `/mnt/data/protostar/fineweb10B` (10.3B GPT-2 tokens, complete)
- Preflight: `PROTOSTAR_DATA=/mnt/data/protostar python env/check_env.py`
- Ablation queue: `sh scripts/ablate.sh` (stop serve first, see below)

## Measured throughput (2026-07-29, micro-batch 4, seq 2048)

| setup | tok/s | scaling |
|---|---|---|
| 1x 5060 Ti | 19,600 | — |
| 2x 5060 Ti DDP, accum 16 | 33,800 | 1.73x |
| + bf16 grad compression hook | 36,400 | 1.86x |

2B-token ablation arm ≈ 14h. Levers not yet pulled: torch.compile,
activation checkpointing (would allow micro-batch 8+).

## pulsar-serve coexistence

Training needs all VRAM on cuda:1/2; pulsar-serve (Kimi K3) holds it.
It is a **transient** systemd user unit — `stop` deletes it. To pause/resume:

```sh
systemctl --user stop pulsar-serve
# ... train ...
systemd-run --user --unit pulsar-serve --working-directory=/home/gianni \
  /usr/bin/bash /home/gianni/pulsar/scripts/serve-supervised.sh \
  -m /mnt/models/kimi-k3/Kimi-K3-IQ1_S-00001-of-00094.gguf \
  --host 100.84.87.107 --port 11435 --ctx 8192
```
