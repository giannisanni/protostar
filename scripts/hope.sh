#!/bin/sh
# hope-lite round: three arms from v2_persist, identical data/budget to v2_curr
# (393M tokens, curriculum every 4th lane, window 512). One arm per GPU.
#   sh scripts/hope.sh <arm> <gpu> [K]      arm = hope_full | cms_only | fixes_only
# Interruptible: re-running resumes from the newest checkpoint.
set -e
cd "$(dirname "$0")/.."
ARM=$1 GPU=$2 K=${3:-1}
RUNS=${RUNS:-/mnt/data/protostar/runs}
OUT=$RUNS/$ARM; mkdir -p "$OUT"
RESUME=$(ls "$OUT"/ckpt_*.pt 2>/dev/null | sort | tail -1)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# GPU "0,1" = DDP over both 5060 Tis (2x the lanes; pass ACCUM=16 for equal tokens/step)
case $GPU in *,*) PY=".venv/bin/torchrun --nproc_per_node=2" ;; *) PY=".venv/bin/python" ;; esac
exec $PY train.py --config "configs/$ARM.json" \
  --data_dir /mnt/data/protostar/fineweb10B --out_dir "$OUT" \
  --init_from "$RUNS/v2_persist/ckpt_003000.pt" ${RESUME:+--resume "$RESUME"} \
  --stateful 1 --window 512 --cur_every 4 --reset_prob 0.0625 --bptt_windows "$K" \
  --steps ${STEPS:-3000} --batch 8 --accum ${ACCUM:-32} \
  --muon_lr 0.005 --adam_lr 1e-4 --warmup 100 --decay_start 2000 \
  --val_every ${VAL_EVERY:-250} --ckpt_every ${CKPT_EVERY:-1000}
