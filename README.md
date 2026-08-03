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

## GPU selection (substrate)

**ALWAYS set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.** PyTorch defaults to
`FASTEST_FIRST`, so bare `CUDA_VISIBLE_DEVICES=0` picks whichever card CUDA
thinks is fastest — measured on substrate, that is the 4060 Ti, NOT
nvidia-smi's index 0. This caused a bake-off to run on the slow card and an
OOM against a GPU nvidia-smi showed as empty. With PCI_BUS_ID set, CUDA
indices match nvidia-smi exactly.

CUDA enumeration (with PCI_BUS_ID set) matches nvidia-smi. Verified 2026-08-02:

| CUDA id | card | PCIe (under load) | use for |
|---|---|---|---|
| 0 | 5060 Ti | Gen5 x8 | **single-GPU jobs — fastest** |
| 1 | 4060 Ti | Gen4 x4 | serving / spillover |
| 2 | 5060 Ti | Gen5 **x1** (crippled slot) | training pair only |

- Single-GPU work (bake-offs, evals, inference): `CUDA_VISIBLE_DEVICES=0`.
- DDP training: `CUDA_VISIBLE_DEVICES=1,2` keeps GPU 0 free, but note GPU 2's
  x1 width — gradient all-reduce is the one workload that actually feels it.
  Compute-bound single-GPU jobs do not care about link width at all.
- Idle cards report Gen1 because PCIe downtrains at low power; **width**
  (x8/x4/x1) is the real, non-varying number.

## pulsar-serve status

The Kimi K3 IQ1_S quant was **deleted 2026-07-29** to free 529G on
/mnt/models; pulsar-serve is down until a model is requanted (pulsar-quant
shard streaming) or another model is chosen. Serve runs as a **transient**
systemd user unit (stop deletes it); recreate with:

```sh
systemd-run --user --unit pulsar-serve --working-directory=/home/gianni \
  /usr/bin/bash /home/gianni/pulsar/scripts/serve-supervised.sh \
  -m <model.gguf> --host 100.84.87.107 --port 11435 --ctx 8192
```

## Ablation queue

`systemd-run --user --unit protostar-ablate --working-directory=/home/gianni/protostar sh scripts/ablate.sh`
— sequential dense → gdn → gdn_mtp → gdn_mtp_engram, ~14h/arm, auto-resume,
report via `scripts/report.py` at the end. Launched 2026-07-29.
Watch: `journalctl --user -u protostar-ablate -f`
