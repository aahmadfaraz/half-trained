#!/bin/bash
# One model through the registered matches (PREREG.md), then the extended set:
#   ./run_llm.sh groq:openai/gpt-oss-120b gpt-oss-120b 7
cd "$(dirname "$0")"; PY=~/.venvs/ailab/bin/python; M=$1; S=$2; PACE=${3:-0}; shift 3
for N in 5 20; do
  $PY -u llm_play.py --chaser agent:runs/wall_s1 --runner "$M" --layout wall --games $N --pace $PACE --until "2026-11-17 23:59" "$@" --out runs/llm/wall_runner/$S.json || exit
  $PY -u llm_play.py --chaser "$M" --runner agent:runs/open_s1 --layout open --games $N --pace $PACE --until "2026-11-17 23:59" "$@" --out runs/llm/open_chaser/$S.json || exit
  $PY -u llm_play.py --chaser agent:runs/open_s1 --runner "$M" --layout open --games $N --pace $PACE --until "2026-11-17 23:59" "$@" --out runs/llm/open_runner/$S.json || exit
done
