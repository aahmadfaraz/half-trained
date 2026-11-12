#!/bin/bash
# Evaluate on CPU: seed 1 at step 2,000 (plus main at 5,000), then each seed run once it finishes. Waits for mains power.
cd "$(dirname "$0")"
PY=~/.venvs/ailab/bin/python
mkdir -p runs/eval
for job in "main 2000" "main 5000" "mem32 2000" "mem8 2000" "mem1 2000" "mem1_s2 5000" "main_s2b 2000" "mem8_s2 2000" "mem32_s2 2000" "main_s3 2000" "mem8_s3 2000" "mem32_s3 2000" "mem1_s3 2000"; do
  set -- $job
  w=runs/$1/weights/w_$(printf %05d $2).pt
  [ -f runs/eval/$1_$2.json ] && grep -q '"handovers"' runs/eval/$1_$2.json && continue
  until { [ -f $w ] && [ -f runs/$1/summary.json ]; } || grep -q "seed runs done" runs/seed_queue.log 2>/dev/null; do sleep 60; done
  [ -f $w ] || { echo "skip $1 $2: no weights" >> runs/eval/eval.log; continue; }
  until pmset -g ac | grep -q -v "No adapter"; do sleep 60; done
  caffeinate -dims $PY evaluate.py --run $1 --step $2 2>&1 | grep --line-buffered -v -i warning >> runs/eval/eval.log
done
echo "eval done $(date)" >> runs/eval/eval.log
