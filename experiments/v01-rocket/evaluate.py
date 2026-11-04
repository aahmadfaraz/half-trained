"""python evaluate.py runs/base_s1/final.zip --episodes 100 --broken left --record 3 --out results/base_s1_left.json
Success = episode score >= 200."""
import argparse, json, os
import numpy as np
from stable_baselines3 import PPO
from common import make_env, ACTION_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--episodes", type=int, default=100)
ap.add_argument("--broken", default="none", choices=["none", "left", "right"])
ap.add_argument("--seed", type=int, default=1000)
ap.add_argument("--record", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

model = PPO.load(a.model, device="cpu")
env = make_env(a.broken)
scores, steps, landed, usage, trajs = [], [], [], np.zeros(4, int), []
for ep in range(a.episodes):
    obs, _ = env.reset(seed=a.seed + ep)
    total, n, done, frames = 0.0, 0, False, []
    while not done:
        act = int(model.predict(obs, deterministic=True)[0])
        usage[act] += 1
        if ep < a.record:
            lander = env.unwrapped.lander
            frames.append({"x": round(float(lander.position.x), 4), "y": round(float(lander.position.y), 4),
                           "angle": round(float(lander.angle), 4), "action": act,
                           "legs": [bool(l.ground_contact) for l in env.unwrapped.legs]})
        obs, r, term, trunc, _ = env.step(act)
        total += r; n += 1; done = term or trunc
    scores.append(total); steps.append(n); landed.append(total >= 200)
    if ep < a.record:
        u = env.unwrapped
        trajs.append({"episode": ep, "seed": a.seed + ep, "score": round(total, 2), "frames": frames,
                      "helipad_x": [float(u.helipad_x1), float(u.helipad_x2)], "helipad_y": float(u.helipad_y),
                      "terrain": [[round(float(p[0][0]), 4), round(float(p[0][1]), 4)] for p in u.sky_polys]
                                 + [[round(float(u.sky_polys[-1][1][0]), 4), round(float(u.sky_polys[-1][1][1]), 4)]]})

res = {"model": a.model, "broken": a.broken, "episodes": a.episodes, "test_seeds": [a.seed, a.seed + a.episodes - 1],
       "success_rate": float(np.mean(landed)), "mean_score": float(np.mean(scores)), "std_score": float(np.std(scores)),
       "median_score": float(np.median(scores)), "mean_steps": float(np.mean(steps)),
       "action_share": {ACTION_NAMES[i]: round(float(usage[i] / usage.sum()), 4) for i in range(4)}}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump({**res, "scores": [round(s, 2) for s in scores], "trajectories": trajs}, open(a.out, "w"))
print(json.dumps(res))
