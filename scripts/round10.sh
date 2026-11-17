#!/bin/sh
# round 10: replay filters on the 7c model: tables-off ablation, flip-rate, both
cd ~/protostar
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
while pgrep -f "[r]eplay-agree" >/dev/null; do sleep 60; done
C=/mnt/data/protostar/runs/xl_cue_qk_subs/ckpt_cue.pt
R=/mnt/data/protostar/runs/round10; mkdir -p $R
run() { .venv/bin/python accretion/lifetime.py --ckpt $C --data /mnt/data/protostar/fineweb10B --replay 0.3 --out /mnt/data/protostar/lifetime_r10 --device cuda:0 --no-gate --dream --consensus --day stream $2 > $R/$1.txt 2>&1; }
run flip --replay-flip & run both "--replay-flip --replay-ablate" & wait
run ablate --replay-ablate
