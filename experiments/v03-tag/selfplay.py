"""Self-play tag: the chaser and the runner take turns learning (PPO) against each other.
python selfplay.py --seed 1 --out runs/open_s1
python selfplay.py --seed 1 --layout wall --init runs/open_s1 --out runs/wall_s1"""
import argparse, csv, json, os, platform, time
import numpy as np, torch, stable_baselines3
from stable_baselines3 import PPO
from arena import TagVec, play, T_MAX

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--layout", default="open")
ap.add_argument("--rounds", type=int, default=60)
ap.add_argument("--steps", type=int, default=400_000, help="steps per role per round")
ap.add_argument("--n_envs", type=int, default=32)
ap.add_argument("--past", type=float, default=0.25, help="share of games against a random past opponent")
ap.add_argument("--init", default="", help="run folder to continue from (its final models)")
ap.add_argument("--eval_games", type=int, default=400)
ap.add_argument("--record_every", type=int, default=5)
ap.add_argument("--ckpt_every", type=int, default=5)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
os.makedirs(os.path.join(a.out, "ckpt"), exist_ok=True)
ROLE = ["chaser", "runner"]

envs = [TagVec(a.n_envs, r, a.layout, a.seed * 100 + r) for r in (0, 1)]
if a.init:
    models = [PPO.load(os.path.join(a.init, f"final_{ROLE[r]}.zip"), env=envs[r], device="cpu") for r in (0, 1)]
else:
    models = [PPO("MlpPolicy", envs[r], n_steps=256, batch_size=2048, n_epochs=5, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                  ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128]), seed=a.seed * 10 + r, device="cpu", verbose=0) for r in (0, 1)]
pool = [[], []]
rng = np.random.default_rng(a.seed)
act = lambda pol: (lambda o: pol.predict(o, deterministic=False)[0])


def snapshot(m):
    p = m.policy.__class__(**m.policy._get_constructor_parameters()); p.load_state_dict(m.policy.state_dict()); p.set_training_mode(False)
    return p

meta = {"seed": a.seed, "layout": a.layout, "rounds": a.rounds, "steps_per_role_per_round": a.steps, "n_envs": a.n_envs, "past_share": a.past,
        "init": a.init, "t_max_steps": T_MAX, "python": platform.python_version(), "torch": torch.__version__, "sb3": stable_baselines3.__version__,
        "machine": platform.machine(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
logf = open(os.path.join(a.out, "log.csv"), "a", newline=""); log = csv.writer(logf)
if os.path.getsize(os.path.join(a.out, "log.csv")) == 0:
    log.writerow(["round", "steps_total", "minutes", "catch_rate", "mean_catch_s", "chaser_vs_random_runner", "runner_vs_random_chaser_caught", "ep_rew_chaser", "ep_rew_runner"])
t0, total = time.time(), 0
for rd in range(1, a.rounds + 1):
    for r in (0, 1):
        opp = models[1 - r].policy
        n_past = int(round(a.n_envs * a.past)) if pool[1 - r] else 0
        groups = [(act(opp), np.arange(a.n_envs - n_past))]
        if n_past: groups.append((act(pool[1 - r][rng.integers(len(pool[1 - r]))]), np.arange(a.n_envs - n_past, a.n_envs)))
        envs[r].set_groups(groups)
        models[r].learn(a.steps, reset_num_timesteps=False)
        pool[r].append(snapshot(models[r]))
        total += a.steps
    ch, rn = act(models[0].policy), act(models[1].policy)
    rec = a.record_every and (rd % a.record_every == 0 or rd == 1)
    caught, t_end, games = play(a.eval_games, ch, rn, a.layout, seed=10_000 + rd, record=6 if rec else 0)
    c_rand, _, _ = play(a.eval_games, ch, None, a.layout, seed=20_000 + rd)
    r_rand, _, _ = play(a.eval_games, None, rn, a.layout, seed=30_000 + rd)
    epr = [np.mean([e["r"] for e in m.ep_info_buffer]) if len(m.ep_info_buffer) else 0 for m in models]
    row = [rd, total, round((time.time() - t0) / 60, 2), round(caught.mean(), 4), round(float(t_end[caught].mean()) / 10 if caught.any() else 0, 2),
           round(c_rand.mean(), 4), round(r_rand.mean(), 4), round(epr[0], 4), round(epr[1], 4)]
    log.writerow(row); logf.flush(); print(*row, flush=True)
    if rd % a.ckpt_every == 0 or rd == 1:
        for r in (0, 1): models[r].save(os.path.join(a.out, "ckpt", f"r{rd:03d}_{ROLE[r]}.zip"))
    if rec: json.dump(games, open(os.path.join(a.out, f"games_r{rd:03d}.json"), "w"))
for r in (0, 1): models[r].save(os.path.join(a.out, f"final_{ROLE[r]}.zip"))
meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
