#!/bin/bash
# 3 seeds: open arena, then the same agents continue with one wall added
cd "$(dirname "$0")"; PY=~/.venvs/ailab/bin/python
for s in 1 2 3; do
  ( $PY selfplay.py --seed $s --rounds 120 --out runs/open_s$s > runs/open_s$s.log 2>&1 &&
    $PY selfplay.py --seed $s --layout wall --init runs/open_s$s --rounds 80 --out runs/wall_s$s > runs/wall_s$s.log 2>&1 ) &
done
wait
