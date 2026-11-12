#!/bin/bash
# Seeds 2 and 3 of every memory size, keeping the step-2,000 weights. main_s2b repeats seed 2 because main_s2 kept only step 5,000.
cd "$(dirname "$0")"
PY=~/.venvs/ailab/bin/python
while pgrep -f "name mem1_s2 " > /dev/null; do sleep 60; done
for job in "main_s2b 256 64 2" "mem8_s2 8 2048 2" "mem32_s2 32 512 2" "main_s3 256 64 3" "mem8_s3 8 2048 3" "mem32_s3 32 512 3" "mem1_s3 1 16384 3"; do
  set -- $job
  [ -f runs/$1/summary.json ] && continue
  mkdir -p runs/$1
  caffeinate -dims $PY train.py --name $1 --block $2 --batch $3 --seed $4 --keep 2000,5000 2>&1 | grep --line-buffered -v -i warning > runs/$1/train.log
done
echo "seed runs done $(date)" >> runs/seed_queue.log
