#!/bin/sh
# usage: round7b.sh <arm> <gpu> <cuetune args...>
set -e
cd ~/protostar
ARM=$1; GPU=$2; shift 2
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU
R=/mnt/data/protostar/runs/xl_cue_$ARM; mkdir -p $R
.venv/bin/python accretion/cuetune.py --ckpt /mnt/data/protostar/runs/xl_near/ckpt_003000.pt --out $R --steps 1600 --batch 4 "$@" > $R/train.txt 2>&1
C=$R/ckpt_cue.pt
.venv/bin/python accretion/valxl.py --ckpt $C > $R/valxl.txt 2>&1
.venv/bin/python accretion/densecheck.py --ckpt $C > $R/densecheck.txt 2>&1
.venv/bin/python accretion/currprobe.py --ckpt $C > $R/currprobe.txt 2>&1
.venv/bin/python accretion/lifetime.py --ckpt $C --data /mnt/data/protostar/fineweb10B --replay 0.3 --out /mnt/data/protostar/lifetime_xl_cue_$ARM --device cuda:0 --no-gate --dream --consensus --day stream > $R/loop_stream.txt 2>&1
