cd "$(dirname "$0")/../.."
PY=~/.venvs/ailab/bin/python
ok() { [ -f $1.receipt.json ] && ! grep -q '"status": "provider_abort"' $1.receipt.json; }
run() { m=$1; tag=$2; for r in 1 2 3; do for s in 1 2 3 7 16 28; do for c in trap control; do
  o=runs/main/${tag}_s${s}_${c}_r${r}; ok $o && continue
  if [ -f $o.receipt.json ]; then mkdir -p runs/main/aborted; ts=$(date +%Y%m%d%H%M%S); for f in $o.*; do mv $f runs/main/aborted/$(basename $f).$ts; done; fi
  $PY run_trip.py --model $m --seed $s --cond $c --out $o > $o.log 2>&1; done; done; done; }
run groq:qwen/qwen3.8-27b qwen3.8-27b &
run groq:openai/gpt-oss-120b gpt-oss-120b &
run gemini:gemini-3.5-flash-lite flash-lite &
wait
echo ALLDONE
