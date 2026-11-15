"""One agent keeps training with each move held for k steps, against a fixed opponent that moves every step.
python train_hold.py --role runner --layout wall --seed 1 --init runs/wall_s1 --opp runs/wall_s1 --out runs/hold_wall_s1"""
import argparse, csv, json, os, platform, time
from collections import deque
import numpy as np, torch, stable_baselines3
from stable_baselines3 import PPO
from arena import TagVec, DT

ROLE = ["chaser", "runner"]


class HoldVec(TagVec):
    def __init__(self, n, role, layout, seed, k, opp):
        super().__init__(n, role, layout, seed); self.k, self.opp, self.ends = k, opp, deque()

    def step_wait(self):
        n, A = self.num_envs, self.A; R, over, caught, trunc = np.zeros(n), np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        tobs = {}
        for _ in range(self.k):
            acts = np.zeros((n, 2), int); acts[:, self.role] = self._a; acts[:, 1 - self.role] = self.opp(A.obs(1 - self.role)); acts[over] = 0
            rc, c, to = A.step(acts)
            R += np.where(over, 0, rc if self.role == 0 else -rc)
            new = ~over & (c | to); caught |= new & c; trunc |= new & to & ~c; over |= new
            if new.any():
                o = A.obs(self.role)
                for i in np.flatnonzero(new): tobs[i] = o[i].copy()
                A.spawn(new)
        A.t[over] = 0
        self.ret += R; self.len += 1
        obs = A.obs(self.role); infos = [{} for _ in range(n)]
        for i in np.flatnonzero(over):
            infos[i] = {"terminal_observation": tobs[i], "TimeLimit.truncated": bool(trunc[i]),
                        "episode": {"r": float(self.ret[i]), "l": int(self.len[i]), "t": 0.0}, "caught": bool(caught[i])}
            self.ends.append((bool(caught[i]), int(self.len[i])))
        self.ret[over] = 0; self.len[over] = 0
        return obs, R.astype(np.float32), over, infos


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=ROLE, required=True); ap.add_argument("--layout", default="wall"); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--init", required=True); ap.add_argument("--opp", required=True); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--chunks", type=int, default=25); ap.add_argument("--chunk", type=int, default=400_000, help="decisions per log row")
    ap.add_argument("--n_envs", type=int, default=32); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--out", required=True)
    a = ap.parse_args(); torch.set_num_threads(a.threads); os.makedirs(a.out, exist_ok=True)
    r = ROLE.index(a.role)
    opp = PPO.load(os.path.join(a.opp, f"final_{ROLE[1 - r]}.zip"), device="cpu").policy; opp.set_training_mode(False)
    env = HoldVec(a.n_envs, r, a.layout, a.seed * 100 + 50 + r, a.k, lambda o: opp.predict(o, deterministic=False)[0])
    model = PPO.load(os.path.join(a.init, f"final_{a.role}.zip"), env=env, device="cpu"); model.set_random_seed(a.seed * 10 + 5 + r)
    meta = {"role": a.role, "layout": a.layout, "seed": a.seed, "init": a.init, "opponent": a.opp, "k": a.k, "chunks": a.chunks, "decisions_per_chunk": a.chunk,
            "n_envs": a.n_envs, "python": platform.python_version(), "torch": torch.__version__, "sb3": stable_baselines3.__version__,
            "machine": platform.machine(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    logf = open(os.path.join(a.out, "log.csv"), "w", newline=""); log = csv.writer(logf)
    log.writerow(["chunk", "decisions_total", "minutes", "games", "catch_rate", "mean_game_s"]); t0 = time.time()
    for c in range(1, a.chunks + 1):
        env.ends.clear(); model.learn(a.chunk, reset_num_timesteps=False)
        e = list(env.ends)
        row = [c, c * a.chunk, round((time.time() - t0) / 60, 2), len(e), round(float(np.mean([x[0] for x in e])), 4), round(float(np.mean([x[1] for x in e])) * a.k * DT, 2)]
        log.writerow(row); logf.flush(); print(*row, flush=True)
    model.save(os.path.join(a.out, f"final_{a.role}.zip"))
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
