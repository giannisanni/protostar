#!/bin/sh
set -e
cd ~/protostar
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
B=/mnt/data/protostar/runs/xl_cue_qk/ckpt_cue.pt
R=/mnt/data/protostar/runs/xl_cue_qk_subs; mkdir -p $R
.venv/bin/python accretion/lifetime.py --ckpt $B --data /mnt/data/protostar/fineweb10B --replay 0.3 --no-gate --train-names --save-subs $R/subs --out /mnt/data/protostar/lifetime_trainnames > $R/sleeps.txt 2>&1
.venv/bin/python accretion/cuetune.py --ckpt $B --out $R --steps 800 --batch 4 --only "\.mix\.w[qk]\." --lr 1e-4 --subs $R/subs/sub_1.pt $R/subs/sub_2.pt $R/subs/sub_3.pt > $R/train.txt 2>&1
C=$R/ckpt_cue.pt
.venv/bin/python accretion/valxl.py --ckpt $C > $R/valxl.txt 2>&1
.venv/bin/python accretion/densecheck.py --ckpt $C > $R/densecheck.txt 2>&1
.venv/bin/python accretion/currprobe.py --ckpt $C > $R/currprobe.txt 2>&1
.venv/bin/python accretion/lifetime.py --ckpt $C --data /mnt/data/protostar/fineweb10B --replay 0.3 --out /mnt/data/protostar/lifetime_xl_cue_qk_subs --device cuda:0 --no-gate --dream --consensus --day stream > $R/loop_stream.txt 2>&1
