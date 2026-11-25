cd "$(dirname "$0")/../.."
PY=~/.venvs/ailab/bin/python
run() { m=$1; tag=$2; shift 2; for s in "$@"; do for c in trap control; do o=runs/pilot2/pilot2_${tag}_s${s}_${c}; $PY run_trip.py --model $m --seed $s --cond $c --out $o > $o.log 2>&1; done; done; }
run groq:qwen/qwen3.8-27b qwen3.8-27b 1 2 3 7 16 28 &
run groq:openai/gpt-oss-120b gpt-oss-120b 7 16 &
run gemini:gemini-3.5-flash-lite flash-lite 7 16 &
wait
echo ALLDONE
