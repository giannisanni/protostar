#!/bin/sh
# round 12: seeds 1,2 of 7c vs 7c + ablate+flip (seed 0 = rounds 7c / 11)
cd ~/protostar
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=/mnt/data/protostar/runs/xl_cue_qk_subs/ckpt_cue.pt
R=/mnt/data/protostar/runs/round12; mkdir -p $R
run() { CUDA_VISIBLE_DEVICES=$2 .venv/bin/python accretion/lifetime.py --ckpt $C --data /mnt/data/protostar/fineweb10B --replay 0.3 --out /mnt/data/protostar/lifetime_r12 --device cuda:0 --no-gate --dream --consensus --day stream --seed $3 $4 > $R/$1.txt 2>&1; }
run plain_s1 0 1 & run both_s1 0 1 "--replay-flip --replay-ablate" & run plain_s2 1 2 & run both_s2 1 2 "--replay-flip --replay-ablate" & wait
