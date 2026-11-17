#!/bin/sh
# round 9: micro-sleep (one sleep per window, steps split) vs one sleep per day, 7c model
cd ~/protostar
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
while pgrep -f "[r]eplay-agree" >/dev/null; do sleep 60; done
C=/mnt/data/protostar/runs/xl_cue_qk_subs/ckpt_cue.pt
R=/mnt/data/protostar/runs/round9; mkdir -p $R
.venv/bin/python accretion/lifetime.py --ckpt $C --data /mnt/data/protostar/fineweb10B --replay 0.3 --out /mnt/data/protostar/lifetime_r9 --device cuda:0 --no-gate --dream --consensus --day stream --micro > $R/micro.txt 2>&1
