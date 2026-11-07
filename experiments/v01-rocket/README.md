# v01 · AI Learns to Land a Rocket. Then I Break One Engine.

A PPO agent learns Gymnasium's `LunarLander-v3` from a random start, then one side engine is switched off.
Every number in the video comes from the files in this folder.

## Results

**Training (seed 1, 1.5M moves, 17.8 min on one Mac, CPU only)**: 4,995 attempts.

| What | Number |
|---|---|
| First 1,000 attempts | 977 crashed, 18 flew off, 5 timed out, 0 landed |
| First real landing (score ≥ 200) | attempt 1,269, 200.29 points (it came to rest next to the pad) |
| Next real landing | attempt 1,548 (278 attempts in between) |
| Longest run of time-outs | 29 in a row (attempts 1,620-1,648) |
| Landings per 100 attempts | 11 (1,701-1,800) → 75 (1,801-1,900) → 91 (1,901-2,000) |
| All attempts | 3,001 came to rest (2,996 scored ≥ 200), 1,472 crashed, 495 timed out, 27 flew off |

**Tests: 100 new starts each (seeds 1000-1099, deterministic policy)**

| Copy (seed) | Normal | Left engine off, no retraining | Retrained 1M moves with the engine off |
|---|---|---|---|
| 1 | 97 | 7 | 34 |
| 2 | 100 | 3 | 8 |
| 3 | 98 | 5 | 19 |

With the left engine off and no retraining, copy 1 still pressed LEFT on 47% of its moves (7% normally).
After retraining it still pressed it on 20%.

## Reproduce

Python 3.12, then `pip install -r ../../requirements.txt`.

```
# train three copies (about 18 min each on CPU)
python train.py --seed 1 --steps 1500000 --out runs/base_s1
python train.py --seed 2 --steps 1500000 --out runs/base_s2
python train.py --seed 3 --steps 1500000 --out runs/base_s3

# 100-landing tests: normal, and left engine switched off
python evaluate.py runs/base_s1/final.zip --episodes 100 --record 100 --out results/grid/base_s1_none.json
python evaluate.py runs/base_s1/final.zip --broken left --episodes 100 --record 100 --out results/grid/base_s1_left_zeroshot.json

# retrain with the broken engine, then test again
python train.py --seed 1 --steps 1000000 --broken left --init runs/base_s1/final.zip --out runs/retrain_left_s1
python evaluate.py runs/retrain_left_s1/final.zip --broken left --episodes 100 --record 100 --out results/grid/retrain_left_s1_left.json

# re-run seed 1 logging every attempt, then summarise it
python train.py --seed 1 --steps 1500000 --ckpt_every 1000000000 --record --out runs/rec_s1
python analyze_attempts.py
```

The recorded re-run matches the original run exactly (all 4,995 attempts, identical final weights).
`runs/rec_s1/attempts.npz` (25 MB) is not committed; the last command above regenerates it.

## Files
| Path | What |
|---|---|
| `common.py` | environment + the broken-engine wrapper (the dead button becomes "do nothing") |
| `train.py` | PPO training, monitor logs, checkpoints, `--record` logs every attempt |
| `evaluate.py` | 100-landing tests, action shares, per-frame recordings with the policy's probabilities |
| `analyze_attempts.py` | turns a recorded run into `results/attempts_summary.json` |
| `runs/*/` | final models, run metadata (versions, wall time) and per-attempt monitor CSVs |
| `results/` | test results, recorded test flights and the training summary |
