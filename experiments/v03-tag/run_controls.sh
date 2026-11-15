#!/bin/bash
# Controls for the registered matches: 500 games each (outcomes only) and the first 20 with full replays.
cd "$(dirname "$0")"; PY=~/.venvs/ailab/bin/python
for C in agent agent@5 greedy random; do
  W=$C; O=$C; [[ $C == agent* ]] && W=$C:runs/wall_s1 && O=$C:runs/open_s1
  for SET in "500 --light runs/controls" "20  runs/llm"; do
    set -- $SET; N=$1; D=${@: -1}; L=""; [ $# -eq 3 ] && L=$2
    $PY llm_play.py --chaser agent:runs/wall_s1 --runner $W --layout wall --games $N $L --out $D/wall_runner/control-$C.json | tail -1
    $PY llm_play.py --chaser $O --runner agent:runs/open_s1 --layout open --games $N $L --out $D/open_chaser/control-$C.json | tail -1
    $PY llm_play.py --chaser agent:runs/open_s1 --runner $O --layout open --games $N $L --out $D/open_runner/control-$C.json | tail -1
  done
done
