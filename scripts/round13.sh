#!/bin/sh
# round 13: (a) sparse table writes, 3 seeds, filters on; then (b) Hope tier: XL + CMS trained
# with the round-5 recipe (init v2_persist, near .15, K=2, DDP, 393M tokens), then probes.
cd ~/protostar
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=/mnt/data/protostar/runs/xl_cue_qk_subs/ckpt_cue.pt
R=/mnt/data/protostar/runs/round13; mkdir -p $R
run() { CUDA_VISIBLE_DEVICES=$2 .venv/bin/python accretion/lifetime.py --ckpt $C --data /mnt/data/protostar/fineweb10B --replay 0.3 --out /mnt/data/protostar/lifetime_r13 --device cuda:0 --no-gate --dream --consensus --day stream --replay-flip --replay-ablate --sparse 0.25 --seed $3 > $R/$1.txt 2>&1; }
run sparse_s0 0 0 & run sparse_s1 0 1 & run sparse_s2 1 2 & wait
CUR_NEAR=0.15 BATCH=4 ACCUM=32 sh scripts/hope.sh xl_cms 0,1 2 > $R/xl_cms_train.txt 2>&1
X=/mnt/data/protostar/runs/xl_cms/ckpt_003000.pt
CUDA_VISIBLE_DEVICES=0 .venv/bin/python accretion/currprobe.py --ckpt $X > $R/xl_cms_currprobe.txt 2>&1
CUDA_VISIBLE_DEVICES=0 .venv/bin/python accretion/valxl.py --ckpt $X > $R/xl_cms_valxl.txt 2>&1
