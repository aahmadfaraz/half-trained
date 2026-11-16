"""How the agents play at saved rounds: aim (straight at the runner or ahead of it), speed, where catches happen, time stuck on the wall.
python story_stats.py -> runs/story.json"""
import json, os, numpy as np, torch
from stable_baselines3 import PPO
from arena import play, LAYOUTS, SIZE
from analyze import seg_hits_rect
torch.set_num_threads(2)
WALL = LAYOUTS["wall"][0]; N = 500


def pol(path):
    p = PPO.load(path, device="cpu").policy
    return lambda o: p.predict(o, deterministic=False)[0]


def rect_dist(p):
    dx = np.maximum(np.maximum(WALL[0] - p[:, 0], p[:, 0] - WALL[2]), 0); dy = np.maximum(np.maximum(WALL[1] - p[:, 1], p[:, 1] - WALL[3]), 0)
    return np.hypot(dx, dy)


def stats(chaser, runner, layout):
    caught, t_end, rec = play(N, chaser, runner, layout, seed=80_000, record=N)
    aim, off, vr, edge, stuck_c, stuck_r, blocked, both = [], [], [], [], [], [], [], []
    at_wall, at_corner, lead = [], [], []
    for g in rec:
        xy = np.array(g["xy"]); c, r = xy[:, :2], xy[:, 2:]
        if len(xy) < 3: continue
        dc, dr = np.diff(c, axis=0) * 10, np.diff(r, axis=0) * 10
        sc, sr = np.linalg.norm(dc, axis=1), np.linalg.norm(dr, axis=1)
        los = (r - c)[:-1]; gap = np.linalg.norm(los, axis=1); los = los / (gap[:, None] + 1e-9)
        dev = np.degrees(np.arctan2(los[:, 0] * dc[:, 1] - los[:, 1] * dc[:, 0], (los * dc).sum(1))); lat = los[:, 0] * dr[:, 1] - los[:, 1] * dr[:, 0]
        k = (sc > 0.5) & (gap > 1.0) & (np.abs(lat) > 0.3); lead += list(dev[k] * np.sign(lat[k]))
        m = sc > 0.5
        if m.any():
            cos = (dc[m] * los[m]).sum(1) / sc[m]
            aim.append(float(cos.mean())); off.append(float((cos < np.cos(np.radians(20))).mean()))
        vr.append(float(sr.mean())); edge.append(float(((r < 1.0) | (r > SIZE - 1.0)).any(1).mean()))
        if layout == "wall":
            b = seg_hits_rect(c, r, WALL)
            pc = (rect_dist(c)[:-1] < 0.4) & (sc < 0.3); pr = (rect_dist(r)[:-1] < 0.4) & (sr < 0.3)
            stuck_c.append(float(pc.mean())); stuck_r.append(float(pr.mean())); blocked.append(float(b.mean())); both.append(float((b[:-1] & (sc < 0.3) & (sr < 0.3)).mean()))
        if g["caught"]:
            e = r[-1]; d = np.minimum(e, SIZE - e)
            at_wall.append(bool(d.min() < 1.0)); at_corner.append(bool(d.max() < 1.5))
    row = {"games": N, "caught": int(caught.sum()), "mean_game_s": round(float(t_end.mean()) / 10, 2),
           "chaser_aim_cos": round(float(np.mean(aim)), 3), "chaser_heading_20deg_off_line": round(float(np.mean(off)), 3),
           "chaser_aims_ahead_deg_median": round(float(np.median(lead)), 1), "chaser_aims_ahead_share": round(float((np.array(lead) > 0).mean()), 3),
           "runner_speed": round(float(np.mean(vr)), 2), "runner_near_outer_edge": round(float(np.mean(edge)), 3)}
    if at_wall: row.update({"catches_within_1_of_outer_wall": round(float(np.mean(at_wall)), 3), "catches_in_corner_1.5": round(float(np.mean(at_corner)), 3)})
    if layout == "wall":
        row.update({"wall_blocks_line": round(float(np.mean(blocked)), 3), "chaser_stopped_at_wall": round(float(np.mean(stuck_c)), 3),
                    "runner_stopped_at_wall": round(float(np.mean(stuck_r)), 3), "both_stopped_wall_between": round(float(np.mean(both)), 3)})
    return row


out = {"open, both random": stats(None, None, "open"), "wall, both random": stats(None, None, "wall")}
for s in (1, 2, 3):
    o, w = f"runs/open_s{s}", f"runs/wall_s{s}"
    for r in (1, 5, 10, 20, 60, 120):
        k = f"{o}/ckpt/r{r:03d}"
        if os.path.exists(k + "_chaser.zip"): out[f"open_s{s} round {r}"] = stats(pol(k + "_chaser.zip"), pol(k + "_runner.zip"), "open")
    ch, rn = pol(f"{o}/final_chaser.zip"), pol(f"{o}/final_runner.zip")
    out[f"open_s{s} final"] = stats(ch, rn, "open")
    out[f"open_s{s} final, wall added, no retraining"] = stats(ch, rn, "wall")
    for r in (1, 5, 10, 20, 40, 80):
        k = f"{w}/ckpt/r{r:03d}"
        if os.path.exists(k + "_chaser.zip"): out[f"wall_s{s} round {r}"] = stats(pol(k + "_chaser.zip"), pol(k + "_runner.zip"), "wall")
    out[f"wall_s{s} final"] = stats(pol(f"{w}/final_chaser.zip"), pol(f"{w}/final_runner.zip"), "wall")
for k, v in out.items(): print(k, v, flush=True)
json.dump(out, open("runs/story.json", "w"), indent=1)
