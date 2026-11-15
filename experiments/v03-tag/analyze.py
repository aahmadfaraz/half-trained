"""Strategy numbers from the final agents: how the runner uses the wall.
python analyze.py runs/open_s1 runs/wall_s1 ... -> runs/strategy.json"""
import sys, json, numpy as np, torch
from stable_baselines3 import PPO
from arena import play, LAYOUTS, SIZE, T_MAX
torch.set_num_threads(2)
WALL = LAYOUTS["wall"][0]; WC = np.array([(WALL[0] + WALL[2]) / 2, (WALL[1] + WALL[3]) / 2])


def seg_hits_rect(p, q, r, steps=24):
    t = np.linspace(0, 1, steps)[None, :, None]
    pts = p[:, None, :] + (q - p)[:, None, :] * t
    return ((pts[..., 0] > r[0]) & (pts[..., 0] < r[2]) & (pts[..., 1] > r[1]) & (pts[..., 1] < r[3])).any(1)


out = {}
for run in sys.argv[1:]:
    lay = "wall" if "wall" in run else "open"
    ch, rn = (PPO.load(f"{run}/final_{r}.zip", device="cpu").policy for r in ("chaser", "runner"))
    act = lambda pol: (lambda o: pol.predict(o, deterministic=False)[0])
    n = 500
    caught, t_end, rec = play(n, act(ch), act(rn), lay, seed=70_000, record=n)
    blocked, loops, near, edge = [], [], [], []
    for g in rec:
        xy = np.array(g["xy"]); c, r = xy[:, :2], xy[:, 2:]
        if lay == "wall":
            blocked.append(float(seg_hits_rect(c, r, WALL).mean()))
            ang = np.unwrap(np.arctan2(r[:, 1] - WC[1], r[:, 0] - WC[0]))
            loops.append(float(abs(ang[-1] - ang[0]) / (2 * np.pi)))
            dx = np.maximum(np.maximum(WALL[0] - r[:, 0], r[:, 0] - WALL[2]), 0); dy = np.maximum(np.maximum(WALL[1] - r[:, 1], r[:, 1] - WALL[3]), 0)
            near.append(float((np.hypot(dx, dy) < 1.5).mean()))
        edge.append(float(((r < 1.0) | (r > SIZE - 1.0)).any(1).mean()))
    row = {"layout": lay, "games": n, "catch_rate": round(float(caught.mean()), 4), "mean_survival_s": round(float(t_end.mean()) / 10, 2),
           "runner_near_outer_edge": round(float(np.mean(edge)), 3)}
    if lay == "wall":
        row.update({"wall_blocks_line_of_sight": round(float(np.mean(blocked)), 3), "runner_within_1.5_of_wall": round(float(np.mean(near)), 3),
                    "runner_loops_around_wall_per_game": round(float(np.mean(loops)), 2), "games_with_2plus_loops": round(float(np.mean(np.array(loops) >= 2)), 3)})
    out[run] = row; print(run, row, flush=True)
json.dump(out, open("runs/strategy.json", "w"), indent=1)
