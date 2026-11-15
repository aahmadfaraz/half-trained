"""Per-game numbers for the wall room, with the same definitions as analyze.py."""
import numpy as np
from arena import LAYOUTS

WALL = LAYOUTS["wall"][0]; WC = np.array([(WALL[0] + WALL[2]) / 2, (WALL[1] + WALL[3]) / 2])


def wall_stats(xy, steps=24):
    xy = np.array(xy); c, r = xy[:, :2], xy[:, 2:]
    t = np.linspace(0, 1, steps)[None, :, None]; pts = c[:, None, :] + (r - c)[:, None, :] * t
    blocked = ((pts[..., 0] > WALL[0]) & (pts[..., 0] < WALL[2]) & (pts[..., 1] > WALL[1]) & (pts[..., 1] < WALL[3])).any(1)
    ang = np.unwrap(np.arctan2(r[:, 1] - WC[1], r[:, 0] - WC[0]))
    dx = np.maximum(np.maximum(WALL[0] - r[:, 0], r[:, 0] - WALL[2]), 0); dy = np.maximum(np.maximum(WALL[1] - r[:, 1], r[:, 1] - WALL[3]), 0)
    return {"loops": round(float(abs(ang[-1] - ang[0]) / (2 * np.pi)), 2), "blocked": round(float(blocked.mean()), 3),
            "near_wall": round(float((np.hypot(dx, dy) < 1.5).mean()), 3)}
