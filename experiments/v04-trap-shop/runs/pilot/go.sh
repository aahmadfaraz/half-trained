cd "$(dirname "$0")/../.."
PY=~/.venvs/ailab/bin/python
$PY run_trip.py --model groq:openai/gpt-oss-120b --seed 7 --cond trap --out runs/pilot/pilot_gpt-oss-120b_s7_trap > runs/pilot/pilot_gpt-oss-120b_s7_trap.log 2>&1 &
$PY run_trip.py --model gemini:gemini-3.5-flash-lite --seed 7 --cond trap --out runs/pilot/pilot_flash-lite_s7_trap > runs/pilot/pilot_flash-lite_s7_trap.log 2>&1 &
$PY run_trip.py --model groq:qwen/qwen3.8-27b --seed 7 --cond trap --out runs/pilot/pilot_qwen3.8-27b_s7_trap > runs/pilot/pilot_qwen3.8-27b_s7_trap.log 2>&1 &
wait
$PY run_trip.py --model groq:openai/gpt-oss-120b --seed 7 --cond control --out runs/pilot/pilot_gpt-oss-120b_s7_control > runs/pilot/pilot_gpt-oss-120b_s7_control.log 2>&1 &
$PY run_trip.py --model gemini:gemini-3.5-flash-lite --seed 7 --cond control --out runs/pilot/pilot_flash-lite_s7_control > runs/pilot/pilot_flash-lite_s7_control.log 2>&1 &
wait
echo ALLDONE
