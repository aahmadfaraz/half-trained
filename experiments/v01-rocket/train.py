"""python train.py --seed 1 --steps 1500000 --out runs/base_s1
python train.py --seed 1 --steps 1000000 --broken left --init runs/base_s1/final.zip --out runs/retrain_left_s1
python train.py --seed 1 --steps 1500000 --ckpt_every 1000000000 --record --out runs/rec_s1"""
import argparse, json, os, platform, time
import numpy as np
import torch, gymnasium, stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
import gymnasium as gym
from common import make_env, ENV_ID

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--steps", type=int, default=1_500_000)
ap.add_argument("--broken", default="none", choices=["none", "left", "right"])
ap.add_argument("--init", default="")
ap.add_argument("--n_envs", type=int, default=8)
ap.add_argument("--ckpt_every", type=int, default=25_000)
ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--record", action="store_true", help="log every training episode (states, actions, policy probabilities)")
ap.add_argument("--out", required=True)
a = ap.parse_args()


class Recorder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.frames, self.eps, self.n, self.score = [], [], 0, 0.0

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        u = self.env.unwrapped
        terrain = [[float(p[0][0]), float(p[0][1])] for p in u.sky_polys] + [[float(u.sky_polys[-1][1][0]), float(u.sky_polys[-1][1][1])]]
        self.eps.append({"start": len(self.frames), "pad": [float(u.helipad_x1), float(u.helipad_x2), float(u.helipad_y)], "terrain": terrain})
        self.score = 0.0
        return obs, info

    def step(self, act):
        u = self.env.unwrapped
        L = u.lander
        self.frames.append((L.position.x, L.position.y, L.angle, int(act), u.legs[0].ground_contact, u.legs[1].ground_contact))
        obs, r, term, trunc, info = self.env.step(act)
        self.n += 1
        self.score += float(r)
        if term or trunc:
            end = "crashed" if u.game_over else ("flew off" if abs(obs[0]) >= 1.0 else ("landed" if not u.lander.awake else "timed out"))
            self.eps[-1].update(stop=len(self.frames), end=end, score=round(self.score, 3), end_step=self.n)
        return obs, r, term, trunc, info


class ProbLog(BaseCallback):
    def __init__(self):
        super().__init__()
        self.p = []

    def _on_step(self):
        with torch.no_grad():
            self.p.append(self.model.policy.get_distribution(self.locals["obs_tensor"]).distribution.probs.cpu().numpy().astype(np.float16))
        return True


def env_fn():
    e = make_env(a.broken)
    return Recorder(e) if a.record else e


torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
env = make_vec_env(env_fn, n_envs=a.n_envs, seed=a.seed, monitor_dir=os.path.join(a.out, "monitor"))
if a.init:
    model = PPO.load(a.init, env=env, device="cpu")
    model.set_random_seed(a.seed)
else:
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999, gae_lambda=0.98,
                ent_coef=0.01, device="cpu", seed=a.seed, verbose=0)
ckpt = CheckpointCallback(save_freq=max(1, a.ckpt_every // a.n_envs), save_path=os.path.join(a.out, "checkpoints"), name_prefix="ckpt")
t0 = time.time()
plog = ProbLog() if a.record else None
model.learn(total_timesteps=a.steps, callback=CallbackList([ckpt, plog]) if plog else ckpt, reset_num_timesteps=not a.init)
wall = time.time() - t0
model.save(os.path.join(a.out, "final.zip"))
meta = {"env": ENV_ID, "seed": a.seed, "steps": a.steps, "broken": a.broken, "init": a.init, "n_envs": a.n_envs,
        "wall_seconds": round(wall, 1), "steps_per_second": round(a.steps / wall, 1),
        "hardware": f"{platform.machine()} {platform.platform()}", "cpu_threads": a.threads,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "gymnasium": gymnasium.__version__, "stable_baselines3": stable_baselines3.__version__},
        "ppo": {k: getattr(model, k) for k in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef")}}
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2, default=str)
if a.record:
    P = np.stack(plog.p)
    arrays, index = {}, []
    for i, e in enumerate(env.envs):
        rec = e.env
        f = np.array(rec.frames, dtype=np.float64)
        arrays[f"xyg_{i}"] = f[:, :3].astype(np.float32)
        arrays[f"act_{i}"] = f[:, 3].astype(np.uint8)
        arrays[f"legs_{i}"] = f[:, 4:6].astype(np.uint8)
        arrays[f"probs_{i}"] = P[: len(f), i, :]
        for k, ep in enumerate(rec.eps):
            if "stop" in ep:
                index.append({"env": i, "ep": k, **ep})
    index.sort(key=lambda r: (r["end_step"], r["env"]))
    for n, r in enumerate(index, 1):
        r["attempt"] = n
    np.savez_compressed(os.path.join(a.out, "attempts.npz"), **arrays)
    json.dump(index, open(os.path.join(a.out, "attempts.json"), "w"))
    print("recorded", len(index), "attempts")
print(json.dumps(meta))
