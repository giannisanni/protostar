#!/bin/sh
# Throughput bench. Usage: bench.sh DATA_DIR [1|2 gpus]  (env: STEPS, BATCH, ACCUM)
set -e
cd "$(dirname "$0")/.."
STEPS=${STEPS:-60} BATCH=${BATCH:-16} ACCUM=${ACCUM:-8}
OUT=/mnt/data/protostar/bench_out_$2
ARGS="--data_dir $1 --steps $STEPS --batch $BATCH --accum $ACCUM --val_every 0 --ckpt_every 0 --out_dir $OUT"
rm -rf "$OUT"
if [ "${2:-1}" = 2 ]; then
  CUDA_VISIBLE_DEVICES=1,2 .venv/bin/torchrun --nproc_per_node=2 train.py $ARGS
else
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python train.py $ARGS
fi
