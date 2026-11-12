#!/bin/bash
# The memory test: same model, data, steps and letters per step (16,384); only the context length changes.
# Waits for the main run (context 256) to finish, then trains context 1, 8 and 32 in sequence.
cd "$(dirname "$0")"
PY=~/.venvs/ailab/bin/python
until [ -f runs/main/summary.json ]; do sleep 30; done
for cfg in "mem1 1 16384" "mem8 8 2048" "mem32 32 512"; do
  set -- $cfg
  [ -f runs/$1/summary.json ] && continue
  mkdir -p runs/$1
  caffeinate -dims $PY train.py --name $1 --block $2 --batch $3 2>&1 | grep --line-buffered -v -i warning > runs/$1/train.log
done
echo "memory runs done $(date)"
