#!/bin/sh
set -e
cd ~/protostar
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/data/protostar/runs/xl_cue; mkdir -p $R
CUDA_VISIBLE_DEVICES=0 .venv/bin/python accretion/densecheck.py --ckpt /mnt/data/protostar/runs/xl_dense/ckpt_003000.pt > /mnt/data/protostar/runs/xl_dense/densecheck.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python accretion/cuetune.py --ckpt /mnt/data/protostar/runs/xl_near/ckpt_003000.pt --out $R --steps 1600 --batch 4 > $R/train.txt 2>&1
wait
C=$R/ckpt_cue.pt
(CUDA_VISIBLE_DEVICES=0 .venv/bin/python accretion/valxl.py --ckpt $C > $R/valxl.txt 2>&1;
 CUDA_VISIBLE_DEVICES=0 .venv/bin/python accretion/currprobe.py --ckpt $C > $R/currprobe.txt 2>&1) &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python accretion/densecheck.py --ckpt $C > $R/densecheck.txt 2>&1
wait
L="--ckpt $C --data /mnt/data/protostar/fineweb10B --replay 0.3 --out /mnt/data/protostar/lifetime_xl_cue --device cuda:0 --no-gate"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python accretion/lifetime.py $L --dream --consensus --day stream > $R/loop_stream.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python accretion/lifetime.py $L > $R/loop_control.txt 2>&1
wait
