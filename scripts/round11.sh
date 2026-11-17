#!/bin/sh
# round 11: rerun flip+ablate with the history key fix (subject+relation, day dreams never skipped);
# and micro-sleep with the same filters
cd ~/protostar
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=/mnt/data/protostar/runs/xl_cue_qk_subs/ckpt_cue.pt
R=/mnt/data/protostar/runs/round11; mkdir -p $R
run() { CUDA_VISIBLE_DEVICES=$2 .venv/bin/python accretion/lifetime.py --ckpt $C --data /mnt/data/protostar/fineweb10B --replay 0.3 --out /mnt/data/protostar/lifetime_r11 --device cuda:0 --no-gate --dream --consensus --day stream --replay-flip --replay-ablate $3 > $R/$1.txt 2>&1; }
run both 0 & run micro_both 1 --micro & wait
