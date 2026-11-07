"""Summarise a recorded training run (train.py --record) into the numbers used on screen."""
import argparse, csv, json, os
from collections import Counter
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/rec_s1")
p.add_argument("--wall_run", default="runs/base_s1")
p.add_argument("--out", default="results/attempts_summary.json")
a = p.parse_args()

A = json.load(open(os.path.join(a.run, "attempts.json")))
Z = np.load(os.path.join(a.run, "attempts.npz"))
meta = json.load(open(os.path.join(a.wall_run, "meta.json")))
n_envs = meta["n_envs"]

wall = {}
for e in range(n_envs):
    rows = list(csv.reader(open(os.path.join(a.wall_run, "monitor", f"{e}.monitor.csv"))))[2:]
    wall[e] = [float(r[2]) for r in rows]

def ok(x): return x["score"] >= 200
def ends(xs): return dict(Counter(x["end"] for x in xs))

def frames(x):
    e = x["env"]
    return Z[f"act_{e}"][x["start"]:x["stop"]], Z[f"legs_{e}"][x["start"]:x["stop"]], Z[f"probs_{e}"][x["start"]:x["stop"]].astype(np.float32)

def info(x):
    act, legs, probs = frames(x)
    down = np.where(legs.any(1))[0]
    after = act[down[0]:] if len(down) else act[:0]
    return {"attempt": x["attempt"], "end": x["end"], "score": round(x["score"], 2), "frames": int(len(act)),
            "seconds": round(len(act) / 50, 2), "total_steps": x["end_step"] * n_envs,
            "wall_minutes": round(wall[x["env"]][x["ep"]] / 60, 2),
            "touched_down": bool(len(down)), "first_touch_s": round(down[0] / 50, 2) if len(down) else None,
            "legs_down_share": round(float(legs.any(1).mean()), 3),
            "engine_share_after_touch": round(float((after != 0).mean()), 3) if len(after) else None,
            "mean_top_prob": round(float(probs.max(1).mean()), 3),
            "action_share": {k: round(float((act == i).mean()), 3) for i, k in enumerate(["nothing", "left", "main", "right"])}}

first = lambda f: next(x for x in A if f(x))
f_ok = first(ok)
f_to = first(lambda x: x["end"] == "timed out")
f_rest = first(lambda x: x["end"] == "landed")
last_to = [x for x in A if x["end"] == "timed out"][-1]
last_crash = [x for x in A if x["end"] == "crashed"][-1]

def window_hit(th, w=100):
    s = np.array([ok(x) for x in A], float)
    c = np.convolve(s, np.ones(w), "valid") / w
    i = int(np.argmax(c >= th)) if (c >= th).any() else None
    return None if i is None else {"window": [i + 1, i + w], "rate": float(c[i]), "wall_minutes": info(A[i + w - 1])["wall_minutes"]}

to = [x for x in A if x["end"] == "timed out"]
to_info = [info(x) for x in to]
tw = np.convolve(np.array([x["end"] == "timed out" for x in A], float), np.ones(100), "valid") / 100

out = {
    "run": a.run, "attempts": len(A), "ends": ends(A), "success_ge_200": sum(ok(x) for x in A),
    "wall_minutes_total": round(meta["wall_seconds"] / 60, 2), "steps_total": meta["steps"],
    "attempt_1": info(A[0]),
    "first_timeout": info(f_to),
    "first_came_to_rest": info(f_rest),
    "first_success": info(f_ok),
    "before_first_success": {"attempts": f_ok["attempt"] - 1, "ends": ends(A[:f_ok["attempt"] - 1]),
                             "came_to_rest_under_200": sum(x["end"] == "landed" and not ok(x) for x in A[:f_ok["attempt"] - 1])},
    "first_100": {"ends": ends(A[:100]), "success": sum(ok(x) for x in A[:100])},
    "last_100": {"ends": ends(A[-100:]), "success": sum(ok(x) for x in A[-100:])},
    "rolling100_first_50pct": window_hit(0.5), "rolling100_first_90pct": window_hit(0.9),
    "timeouts": {"count": len(to), "touched_down": sum(t["touched_down"] for t in to_info),
                 "mean_legs_down_share": round(float(np.mean([t["legs_down_share"] for t in to_info])), 3),
                 "mean_engine_share_after_touch": round(float(np.mean([t["engine_share_after_touch"] for t in to_info if t["touched_down"]])), 3),
                 "peak_rolling100": {"rate": float(tw.max()), "window": [int(tw.argmax()) + 1, int(tw.argmax()) + 100]},
                 "before_first_success": sum(1 for x in to if x["attempt"] < f_ok["attempt"]),
                 "last": last_to["attempt"]},
    "last_crash": last_crash["attempt"],
    "best_score": max(round(x["score"], 2) for x in A),
    "last_attempt": info(A[-1]),
    "mean_top_prob_first_100": round(float(np.mean([info(x)["mean_top_prob"] for x in A[:100]])), 3),
    "mean_top_prob_last_100": round(float(np.mean([info(x)["mean_top_prob"] for x in A[-100:]])), 3),
}
oks = [x["attempt"] for x in A if ok(x)]
out["second_success"] = oks[1]
out["max_consecutive_timeouts"] = max(len(s) for s in "".join("T" if x["end"] == "timed out" else "." for x in A).split("."))
out["blocks_of_100"] = [{"attempts": [b + 1, b + 100], "success": sum(ok(x) for x in A[b:b + 100]), "ends": ends(A[b:b + 100]),
                         "wall_minutes_end": info(A[min(b + 99, len(A) - 1)])["wall_minutes"]} for b in range(0, len(A), 100)]
out["scores_by_era"] = {f"{lo}-{hi}": {k: {"n": len(v), "mean": round(float(np.mean(v)), 1)} for k, v in
                        {e: [x["score"] for x in A[lo - 1:hi] if x["end"] == e] for e in set(x["end"] for x in A[lo - 1:hi])}.items()}
                        for lo, hi in [(1, 1000), (1101, 1800), (1801, len(A))]}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out, indent=1))
