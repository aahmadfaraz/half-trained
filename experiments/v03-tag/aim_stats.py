"""How each chaser aims in the open room: the angle between its push and the straight line to the runner.
python aim_stats.py -> runs/aim.json"""
import json, glob, os, math, statistics as st


def aim(path, every):
    d = json.load(open(path)); offs, straight, ahead = [], 0, []
    for g in d["games"]:
        if g.get("finished", "") > "2026-11-17 23:59:59": continue
        prev = g["start"]
        for i, (cm, _) in enumerate(g["acts"]):
            p = prev if i == 0 else g["xy"][i - 1]
            if i % every == 0 and cm:
                bx, by = p[2] - p[0], p[3] - p[1]; bear = math.degrees(math.atan2(by, bx))
                push = (cm - 1) * 45.0; off = (push - bear + 180) % 360 - 180
                offs.append(abs(off)); straight += abs(off) <= 22.5
                if i > 0:
                    q = g["xy"][i - 2] if i > 1 else prev; vx, vy = p[2] - q[2], p[3] - q[3]
                    side = bx * vy - by * vx
                    if abs(side) > 1e-6 and abs(off) > 1e-6: ahead.append(1 if (off > 0) == (side > 0) else 0)
    n = len(offs)
    return {"decisions": n, "mean_deg_off_straight": round(st.mean(offs), 1) if n else None,
            "share_straight_at_runner": round(straight / n, 3) if n else None,
            "share_turned_toward_where_it_is_going": round(st.mean(ahead), 3) if ahead else None}


out = {}
for f in sorted(glob.glob("runs/llm/open_chaser/*.json")):
    name = os.path.basename(f)[:-5]
    out[name] = aim(f, 1 if name in ("control-agent", "control-random") else 5)
json.dump(out, open("runs/aim.json", "w"), indent=1)
for k, v in out.items(): print(f"{k:28} {v}")
