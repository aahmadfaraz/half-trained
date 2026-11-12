#!/bin/bash
# Two more copies (seeds 2 and 3) of every context length, after memory_runs.sh finishes.
cd "$(dirname "$0")"
PY=~/.venvs/ailab/bin/python
until grep -q "memory runs done" runs/memory_queue.log 2>/dev/null; do sleep 60; done
for seed in 2 3; do
  for cfg in "main 256 64" "mem1 1 16384" "mem8 8 2048" "mem32 32 512"; do
    set -- $cfg
    name=${1}_s$seed
    [ -f runs/$name/summary.json ] && continue
    mkdir -p runs/$name
    caffeinate -dims $PY train.py --name $name --block $2 --batch $3 --seed $seed --keep 5000 2>&1 | grep --line-buffered -v -i warning > runs/$name/train.log
  done
done
echo "seed runs done $(date)"
