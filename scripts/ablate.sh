#!/bin/sh
# Ablation queue: sequential 2B-token runs, one per config, on both 5060 Tis.
# REQUIRES pulsar-serve stopped first:  systemctl --user stop pulsar-serve
# Each arm ~14h at measured 36k tok/s. Interruptible: re-running resumes.
set -e
cd "$(dirname "$0")/.."
DATA=${DATA:-/mnt/data/protostar/fineweb10B}
RUNS=${RUNS:-/mnt/data/protostar/runs}
STEPS=7630 DECAY=6100  # 2B tokens at 262k tok/step, WSD decay over last 20%
for name in dense gdn gdn_mtp gdn_mtp_engram; do
  OUT=$RUNS/$name
  mkdir -p "$OUT"
  [ -f "$OUT/ckpt_00$STEPS.pt" ] && continue  # arm already done
  RESUME=$(ls "$OUT"/ckpt_*.pt 2>/dev/null | sort | tail -1)
  echo "=== arm: $name ${RESUME:+(resuming $RESUME)} ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1,2 \
    .venv/bin/torchrun --nproc_per_node=2 train.py --config "configs/$name.json" \
    --data_dir "$DATA" --out_dir "$OUT" --steps "$STEPS" --decay_start "$DECAY" \
    --batch 4 --accum 16 ${RESUME:+--resume "$RESUME"}
done
.venv/bin/python scripts/report.py "$RUNS"
