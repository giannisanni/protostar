#!/bin/sh
# round 8: gate replay dreams on vote agreement (K of 7), on the 7c model
cd ~/protostar
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=/mnt/data/protostar/runs/xl_cue_qk_subs/ckpt_cue.pt
R=/mnt/data/protostar/runs/round8; mkdir -p $R
run() { CUDA_VISIBLE_DEVICES=$2 .venv/bin/python accretion/lifetime.py --ckpt $C --data /mnt/data/protostar/fineweb10B --replay 0.3 --out /mnt/data/protostar/lifetime_r8 --device cuda:0 --no-gate --dream --consensus --day stream --replay-agree $1 > $R/agree$1.txt 2>&1; }
run 4 0 & run 6 0 & run 5 1 & wait
for k in 4 5 6; do echo "#### agree $k"; grep -E "replay dreams|FORGETTING|final val|^  [ABC] " $R/agree$k.txt; done
