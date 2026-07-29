#!/bin/sh
# Fetch pre-tokenized GPT-2 FineWeb shards (llm.c .bin format, 100M tokens each).
# ponytail: KellerJordan/fineweb10B-gpt2 is the known-good speedrun artifact;
# own FineWeb-Edu + GigaToken corpus is the 350M run's job.
# Usage: download_fineweb.sh DEST [N_TRAIN_SHARDS (default 103 = full 10B)]
set -e
DEST=${1:?dest dir}
N=${2:-103}
BASE=https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main
mkdir -p "$DEST"
cd "$DEST"
{ echo fineweb_val_000000.bin; seq -f 'fineweb_train_%06g.bin' 1 "$N"; } |
  xargs -P4 -I{} sh -c "test -s {} || curl -sfL -o {} $BASE/{}"
ls ./*.bin | wc -l
du -sh .
