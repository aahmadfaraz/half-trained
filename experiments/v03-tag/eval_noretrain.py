"""Open-arena agents dropped into the wall arena with no retraining.
python eval_noretrain.py runs/open_s1 runs/open_s2 runs/open_s3"""
import sys, json, numpy as np, torch
from stable_baselines3 import PPO
from arena import play
torch.set_num_threads(1)
out = {}
for run in sys.argv[1:]:
    ch, rn = (PPO.load(f"{run}/final_{r}.zip", device="cpu").policy for r in ("chaser", "runner"))
    act = lambda pol: (lambda o: pol.predict(o, deterministic=False)[0])
    row = {}
    for lay in ("open", "wall"):
        c, te, _ = play(1000, act(ch), act(rn), lay, seed=50_000)
        row[lay] = {"catch_rate": round(float(c.mean()), 4), "mean_catch_s": round(float(te[c].mean()) / 10, 2) if c.any() else None}
    out[run] = row; print(run, row, flush=True)
json.dump(out, open("runs/noretrain.json", "w"), indent=1)
