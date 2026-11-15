#!/bin/sh
set -e
cd ~/protostar
R=/mnt/data/protostar/runs/xl_dense
CUR_NEAR=0.15 CUR_DENSITY=0.3 CUR_WRAP=0.5 ACCUM=16 sh scripts/hope.sh xl_dense 0,1 2
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
(CUDA_VISIBLE_DEVICES=0 .venv/bin/python accretion/currprobe.py --ckpt $R/ckpt_003000.pt > $R/currprobe.txt 2>&1;
 CUDA_VISIBLE_DEVICES=0 .venv/bin/python accretion/ceiling.py --ckpt $R/ckpt_003000.pt > $R/ceiling.txt 2>&1) &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python accretion/valxl.py --ckpt $R/ckpt_003000.pt > $R/valxl.txt 2>&1
wait
L="--ckpt $R/ckpt_003000.pt --data /mnt/data/protostar/fineweb10B --replay 0.3 --out /mnt/data/protostar/lifetime_xl_dense --device cuda:0 --no-gate"
(CUDA_VISIBLE_DEVICES=0 .venv/bin/python accretion/lifetime.py $L --dream --consensus --day stream > $R/loop_stream.txt 2>&1;
 CUDA_VISIBLE_DEVICES=0 .venv/bin/python accretion/lifetime.py $L > $R/loop_control.txt 2>&1) &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python accretion/lifetime.py $L --dream --consensus > $R/loop_context.txt 2>&1
wait
