"""python train.py --seed 1 --steps 1500000 --out runs/base_s1
python train.py --seed 1 --steps 1000000 --broken left --init runs/base_s1/final.zip --out runs/retrain_left_s1"""
import argparse, json, os, platform, time
import torch, gymnasium, stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from common import make_env, ENV_ID

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--steps", type=int, default=1_500_000)
ap.add_argument("--broken", default="none", choices=["none", "left", "right"])
ap.add_argument("--init", default="")
ap.add_argument("--n_envs", type=int, default=8)
ap.add_argument("--ckpt_every", type=int, default=25_000)
ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--out", required=True)
a = ap.parse_args()

torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
env = make_vec_env(lambda: make_env(a.broken), n_envs=a.n_envs, seed=a.seed, monitor_dir=os.path.join(a.out, "monitor"))
if a.init:
    model = PPO.load(a.init, env=env, device="cpu")
    model.set_random_seed(a.seed)
else:
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999, gae_lambda=0.98,
                ent_coef=0.01, device="cpu", seed=a.seed, verbose=0)
ckpt = CheckpointCallback(save_freq=max(1, a.ckpt_every // a.n_envs), save_path=os.path.join(a.out, "checkpoints"), name_prefix="ckpt")
t0 = time.time()
model.learn(total_timesteps=a.steps, callback=ckpt, reset_num_timesteps=not a.init)
wall = time.time() - t0
model.save(os.path.join(a.out, "final.zip"))
meta = {"env": ENV_ID, "seed": a.seed, "steps": a.steps, "broken": a.broken, "init": a.init, "n_envs": a.n_envs,
        "wall_seconds": round(wall, 1), "steps_per_second": round(a.steps / wall, 1),
        "hardware": f"{platform.machine()} {platform.platform()}", "cpu_threads": a.threads,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "gymnasium": gymnasium.__version__, "stable_baselines3": stable_baselines3.__version__},
        "ppo": {k: getattr(model, k) for k in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef")}}
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2, default=str)
print(json.dumps(meta))
